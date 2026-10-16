#!/usr/bin/env python3
# Copyright 2022-2026 Friedrich Miescher Institute for Biomedical Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Georg Kempf, Friedrich Miescher Institute for Biomedical Research
"""
evaluation_pipeline.py

Post‑processing / evaluation utilities for AlphaFold3‑style prediction output.

The module provides two pipelines:

* ``EvaluationPipeline``       - evaluates a single prediction directory
                                 (PAE plots, confidence metrics, HTML report).
* ``EvaluationPipelineBatch``  - aggregates ``best_scores.json`` files produced
                                 by many single evaluations into one summary
                                 report (bar plots / heatmaps).

The script can be run from the command line (see ``main``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import string
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

# Headless backend + single threaded BLAS must be configured before heavy imports.
matplotlib.use("Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import logging as logging_python  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402
from absl import logging  # noqa: E402
from Bio.PDB import MMCIFParser, PDBIO  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

try:  # ``pkg_resources`` is deprecated; fall back gracefully if unavailable.
    import pkg_resources
except Exception:  # pragma: no cover
    pkg_resources = None

logging_python.getLogger("matplotlib.font_manager").disabled = True


# --------------------------------------------------------------------------- #
# Metric configuration
# --------------------------------------------------------------------------- #
# Each entry: (metric_name, optimisation_direction, is_scalar)
#   - direction 'max' -> higher is better, 'min' -> lower is better.
#   - is_scalar       -> whether the value is a single number that can be
#                        sorted.  Array/matrix metrics (pae, chain_pair_*, ...)
#                        must NOT be sorted directly (would raise on comparison).
CONFIDENCE_METRICS: List[Tuple[str, str, bool]] = [
    ("atom_plddts", "max", False),
    ("pae", "min", False),
    ("chain_iptm", "max", False),
    ("chain_pair_iptm", "max", False),
    ("chain_pair_pae_min", "min", False),
    ("chain_ptm", "max", False),
    ("fraction_disordered", "min", True),
    ("has_clash", "min", True),
    ("iptm", "max", True),
    ("ptm", "max", True),
    ("ranking_score", "max", True),
]

CHAIN_COLORS = [
    "lime", "cyan", "magenta", "yellow",
    "salmon", "white", "blue", "orange",
]


def _resource_path(package: str, resource: str) -> str:
    """Return the absolute path to a packaged resource."""
    if pkg_resources is not None:
        return pkg_resources.resource_filename(package, resource)
    # Fallback: assume resource relative to this file.
    return os.path.join(os.path.dirname(__file__), resource)


# --------------------------------------------------------------------------- #
# Single prediction evaluation
# --------------------------------------------------------------------------- #
class EvaluationPipeline:
    """Evaluate a single prediction directory.

    Parameters
    ----------
    results_dir:
        Directory containing the model sub‑folders (``seed-x_sample-y``),
        the ``*summary_confidences.json`` / ``*confidences.json`` files and
        a ``chain_id_name_mapping.json`` file.
    continue_from_existing_results:
        If ``True`` the PAE plots are not regenerated.
    custom_spacing:
        Custom spacing (in residues) for the major tick labels of PAE plots.
    custom_start_residue_list:
        Comma‑separated list of starting residue numbers (one per subunit)
        used to restore the real residue numbering in the plots.
    custom_axis_label_list:
        Comma‑separated list of custom subunit labels.
    server:
        Reserved flag (kept for interface compatibility).
    chain_id_name_mapping_file:
        Explicit path to the chain‑id → protein‑name mapping file.  If not
        given, the file is auto‑discovered inside ``results_dir``.
    """

    def __init__(
        self,
        results_dir: Optional[str] = None,
        continue_from_existing_results: bool = False,
        custom_spacing: Optional[str] = None,
        custom_start_residue_list: Optional[str] = None,
        custom_axis_label_list: Optional[str] = None,
        server: bool = False,
        chain_id_name_mapping_file: Optional[str] = None,
        pairwise: bool = False,
    ) -> None:
        if not results_dir or not os.path.exists(results_dir):
            error_msg = f"Results directory {results_dir} not found"
            logging.error(error_msg)
            raise SystemExit(error_msg)

        self.results_dir = results_dir
        self.continue_from_existing_results = continue_from_existing_results
        self.custom_spacing = custom_spacing
        self.custom_start_residue_list = custom_start_residue_list
        self.custom_axis_label_list = custom_axis_label_list
        self.server = server
        self.chain_id_name_mapping_file = chain_id_name_mapping_file
        self.pairwise = pairwise

        self.confidence_metrics = CONFIDENCE_METRICS

        # Populated during the run.
        self.cif_dict: Dict[str, str] = {}
        self.json_full_data_dict: Dict[str, str] = {}
        self.json_summary_confidences_dict: Dict[str, str] = {}
        self.chain_id_name_mapping: Dict[str, str] = {}
        self.subunit_list: List[str] = []
        self.seq_titles: List[str] = []

    # ----------------------------------------------------------------- #
    # File collection
    # ----------------------------------------------------------------- #
    def collect_files_local(self, dir_name: str) -> None:
        """Discover model files inside ``dir_name``.

        Fills ``cif_dict``, ``json_full_data_dict`` and
        ``json_summary_confidences_dict`` keyed by model name
        (``seed-x_sample-y``).  Raises if any required file type is missing.
        """
        self.cif_dict = {}
        self.json_full_data_dict = {}
        self.json_summary_confidences_dict = {}

        regex = r"seed-(\d+)_sample-(\d+)"
        for entry in os.listdir(dir_name):
            path = os.path.join(dir_name, entry)

            match = re.search(regex, entry)
            if match and os.path.isdir(path):
                seed, sample = match.group(1), match.group(2)
                model = f"seed-{seed}_sample-{sample}"
                for sub in os.listdir(path):
                    sub_path = os.path.join(path, sub)
                    if sub.endswith("summary_confidences.json"):
                        self.json_summary_confidences_dict[model] = sub_path
                    elif sub.endswith("confidences.json"):
                        self.json_full_data_dict[model] = sub_path
                    elif sub.endswith("_model.cif"):
                        self.cif_dict[model] = sub_path

            # Auto‑discover the chain mapping file if not supplied.
            if entry.endswith("chain_id_name_mapping.json") and not self.chain_id_name_mapping_file:
                self.chain_id_name_mapping_file = path

        if not self.pairwise and not self.cif_dict:
            raise FileNotFoundError("No cif files found")
        if not self.json_full_data_dict:
            raise FileNotFoundError("No confidence files found")
        if not self.json_summary_confidences_dict:
            raise FileNotFoundError("No summary confidence files found")
        if not self.pairwise and not self.chain_id_name_mapping_file:
            raise FileNotFoundError("No chain_id_name_mapping.json found and no pairwise option.")

    def get_chain_id_name_mapping(self) -> Dict[str, str]:
        """Load the chain‑id → protein‑name mapping JSON."""
        with open(self.chain_id_name_mapping_file) as fh:
            return json.load(fh)

    # ----------------------------------------------------------------- #
    # Small helpers
    # ----------------------------------------------------------------- #
    @staticmethod
    def find_sequence_in_dict(d: Any, sequences: List[str]) -> List[str]:
        """Recursively collect every value stored under a ``"sequence"`` key."""
        if isinstance(d, dict):
            for key, value in d.items():
                if key == "sequence":
                    sequences.append(value)
                elif isinstance(value, (dict, list)):
                    EvaluationPipeline.find_sequence_in_dict(value, sequences)
        elif isinstance(d, list):
            for item in d:
                EvaluationPipeline.find_sequence_in_dict(item, sequences)
        return sequences

    @staticmethod
    def create_alphabet_list(n: int) -> List[str]:
        """Return the first ``n`` lowercase ASCII letters as a list."""
        if n > len(string.ascii_lowercase):
            raise ValueError("Number of sequences is higher than the length of the alphabet")
        return list(string.ascii_lowercase[:n])

    def round_float(self, data: Any, decimal_places: int = 2) -> Any:
        """Recursively round floats contained in nested lists/tuples/dicts."""
        if isinstance(data, float):
            return round(data, decimal_places)
        if isinstance(data, tuple):
            return tuple(self.round_float(item, decimal_places) for item in data)
        if isinstance(data, list):
            return [self.round_float(item, decimal_places) for item in data]
        if isinstance(data, dict):
            return {k: self.round_float(v, decimal_places) for k, v in data.items()}
        return data

    def convert_to_list(self, obj: Any) -> Any:
        """Recursively convert jax/numpy arrays to plain python lists."""
        if isinstance(obj, dict):
            return {k: self.convert_to_list(v) for k, v in obj.items()}
        if isinstance(obj, jnp.ndarray):
            return jax.device_get(obj).tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def convert_to_np(self, obj: Any) -> Any:
        """Recursively convert jax arrays / sequences to numpy arrays."""
        if isinstance(obj, dict):
            return {k: self.convert_to_np(v) for k, v in obj.items()}
        if isinstance(obj, jnp.ndarray):
            return np.array(jax.device_get(obj).tolist())
        if isinstance(obj, (list, tuple)):
            return np.array(obj)
        return obj

    @staticmethod
    def convert_cif_to_pdb(cif_list: Sequence[str]) -> List[Tuple[str, str]]:
        """Convert ``.cif`` files to ``.pdb`` and return (path, atom_records)."""
        parser = MMCIFParser(QUIET=True)
        io = PDBIO()
        pdb_list: List[Tuple[str, str]] = []
        for cif_file in cif_list:
            structure = parser.get_structure("structure", cif_file)
            pdb_file = cif_file.replace(".cif", ".pdb")
            io.set_structure(structure)
            io.save(pdb_file)
            with open(pdb_file, "r") as fin:
                content = "".join(line for line in fin if line.startswith("ATOM"))
            pdb_list.append((pdb_file, content))
        return pdb_list

    @staticmethod
    def check_none(nested_list: Any) -> bool:
        """Return ``True`` if any element of a nested structure is ``None``."""
        if isinstance(nested_list, (list, tuple)):
            return any(EvaluationPipeline.check_none(x) for x in nested_list)
        return nested_list is None

    # ----------------------------------------------------------------- #
    # Sorting helpers
    # ----------------------------------------------------------------- #
    @staticmethod
    def sort_dict(results: Dict[str, Any], reverse: bool = False) -> List[Tuple[str, Any]]:
        """Sort a ``{model: scalar}`` dict by value."""
        return sorted(results.items(), key=lambda x: x[1], reverse=reverse)

    @staticmethod
    def sort_pae_results(results: Dict[str, Dict[str, float]]) -> List[Tuple[str, Dict[str, float]]]:
        """Sort PAE results by the overall average PAE (ascending)."""
        return sorted(results.items(), key=lambda x: x[1]["Overall"], reverse=False)

    @staticmethod
    def find_index_by_protein_name(data_list: Sequence[Dict[str, Any]], search_value: str) -> int:
        """Return index of the dict whose ``protein_names`` matches ``search_value``."""
        for index, item in enumerate(data_list):
            if item.get("protein_names") == search_value:
                return index
        return -1

    # ----------------------------------------------------------------- #
    # Scoring output
    # ----------------------------------------------------------------- #
    def save_scoring_results(self) -> None:
        """Write the best (ptm, iptm, pae, ranking) scores to ``best_scores.json``."""
        ptm_model, ptm = next(iter(self.ptm_by_model_dict.items()))
        iptm_model, iptm = next(iter(self.iptm_by_model_dict.items()))
        multimer_model, multimer_score = next(iter(self.ranking_score_by_model_dict.items()))

        # inter_pae dict is ``{model: value}`` sorted ascending.
        inter_pae_model = min(self.inter_pae_by_model_dict, key=self.inter_pae_by_model_dict.get)
        inter_pae = self.inter_pae_by_model_dict[inter_pae_model]

        best_scores = {
            "protein_names": "_".join(self.seq_titles),
            "subunits": self.subunit_list,
            "max_ptm_model_name": ptm_model,
            "max_ptm_value": ptm,
            "max_iptm_model_name": iptm_model,
            "max_iptm_value": iptm,
            "min_pae_model_name": inter_pae_model,
            "min_pae_value": inter_pae,
            "max_multimer_score_model_name": multimer_model,
            "max_multimer_score_value": multimer_score,
        }
        with open(os.path.join(self.results_dir, "best_scores.json"), "w") as f:
            json.dump(best_scores, f, default=float)

    # ----------------------------------------------------------------- #
    # Main driver
    # ----------------------------------------------------------------- #
    def run_pipeline(self) -> None:
        """Run the full single prediction evaluation."""
        logging.info(f"Running evaluation pipeline in {self.results_dir}.")

        self.collect_files_local(self.results_dir)
        if self.chain_id_name_mapping_file:
            self.chain_id_name_mapping = self.get_chain_id_name_mapping()
        elif self.pairwise:
            subunits = os.path.basename(self.results_dir).split('_')
            if len(subunits) != 2:
                raise ValueError(f"Number of subunits extracted from results folder name {self.results_dir} is {len(subunits)}. In pairwise mode two subunits need to be separated by \"_\" in the filename")
            self.chain_id_name_mapping = {'A': subunits[0], 'B': subunits[1]}

        logging.info("Extracting results.")

        # Prepare per‑metric result dicts.
        for name, _, _ in self.confidence_metrics:
            setattr(self, f"{name}_by_model_dict", {})

        # Load and merge the full + summary confidence JSONs per model.
        data_by_model_dict: Dict[str, Dict[str, Any]] = {}
        for model, json_full in self.json_full_data_dict.items():
            json_confidence = self.json_summary_confidences_dict[model]
            with open(json_full) as full_fh, open(json_confidence) as conf_fh:
                data_full = json.load(full_fh)
                data_full.update(json.load(conf_fh))
                data_by_model_dict[model] = data_full

        # Determine chain layout from any model (all share the same chains).
        chain_ids: List[str] = []
        last_data: Dict[str, Any] = {}
        for model, data in data_by_model_dict.items():
            self.pae_by_model_dict[model] = data["pae"]
            last_data = data
        # ``sorted`` keeps the ordering deterministic (set() alone is not).
        chain_ids = sorted(set(last_data["token_chain_ids"]))

        seq_titles = [self.chain_id_name_mapping[chain_id] for chain_id in chain_ids]
        self.subunit_list = self.seq_titles = seq_titles

        # Compute (start, end) token indices for every chain.
        indices: List[Tuple[int, int]] = []
        prev = 0
        token_chain_ids = last_data["token_chain_ids"]
        for chain_id in chain_ids:
            seq_len = token_chain_ids.count(chain_id) + prev
            indices.append((prev, seq_len))
            prev = seq_len

        # --- PAE plots --------------------------------------------------
        if not self.continue_from_existing_results:
            self.plot_pae(
                self.pae_by_model_dict,
                indices,
                seq_titles,
                self.custom_spacing,
                self.custom_start_residue_list,
                self.custom_axis_label_list,
            )

        # --- Average PAE per subunit combination -----------------------
        average_pae = self.analyse_pae(self.pae_by_model_dict, indices, seq_titles)
        # ``average_pae_by_model_dict`` becomes a *sorted list* of
        # (model, combination_dict) tuples.
        self.average_pae_by_model_dict = self.sort_pae_results(average_pae)

        # --- Collect raw confidence values -----------------------------
        for name, _, _ in self.confidence_metrics:
            dict_ = getattr(self, f"{name}_by_model_dict")
            for model, data in data_by_model_dict.items():
                dict_[model] = data[name]

        # --- Sort / round scalar metrics only --------------------------
        # Only scalar metrics can be sorted by value; array metrics are left
        # as plain dicts (sorting them would raise an ambiguity error).
        for name, direction, is_scalar in self.confidence_metrics:
            dict_ = getattr(self, f"{name}_by_model_dict")
            if is_scalar:
                reverse = direction == "max"
                sorted_items = self.sort_dict(dict_, reverse)
                # Keep a dict (insertion order preserved) so ``next(iter(...))``
                # yields the best entry.
                setattr(self, f"{name}_by_model_dict",
                        {k: self.round_float(v) for k, v in sorted_items})

        # --- Average plddt per model -----------------------------------
        self.plddt_by_model_dict = {}
        for model, plddt_by_atom in self.atom_plddts_by_model_dict.items():
            self.plddt_by_model_dict[model] = float(np.mean(plddt_by_atom))

        # --- Inter‑chain PAE (average over off‑diagonal combinations) --
        self.inter_pae_by_model_dict = {}
        for model, combination_pae_dict in self.average_pae_by_model_dict:
            inter_pae = []
            for combination, pae in combination_pae_dict.items():
                parts = combination.split(" vs ")
                if len(parts) > 1 and parts[0] != parts[1]:
                    inter_pae.append(pae)
            # Fall back to the overall PAE for a single‑chain prediction.
            self.inter_pae_by_model_dict[model] = (
                float(np.mean(inter_pae)) if inter_pae
                else float(combination_pae_dict["Overall"])
            )

        self.confidence_metrics_names = [x[0] for x in self.confidence_metrics]

        if self.cif_dict:
            chain_color = self.get_chain_color()
        else:
            chain_color = list(zip(chain_ids, CHAIN_COLORS))
        pdb_files = self.convert_cif_to_pdb(list(self.cif_dict.values()))

        # --- Render HTML report ----------------------------------------
        templates_path = _resource_path("guifold3", "templates")
        logging.info(templates_path)

        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template("results.html")

        render_attributes: Dict[str, Any] = {
            f"{name}_by_model_dict": getattr(self, f"{name}_by_model_dict")
            for name in self.confidence_metrics_names
        }
        render_attributes.update(
            pae_results=self.average_pae_by_model_dict,
            pae_examples_path=_resource_path("guifold3", "images/pae_examples.png"),
            pdb_path_list=[(os.path.basename(p), c) for p, c in pdb_files],
            chain_color=chain_color,
            images=self.get_image_path_list(self.average_pae_by_model_dict),
            plddt_by_model_dict=self.plddt_by_model_dict,
            use_model_viewer=True,
        )

        rendered = template.render(render_attributes)
        html_path = os.path.join(self.results_dir, "results_model_viewer.html")
        with open(html_path, "w") as f_out:
            f_out.write(rendered)

        plt.close("all")
        logging.info(f"Finished. Results written to {self.results_dir}.")

        self.save_scoring_results()

    # ----------------------------------------------------------------- #
    # PAE analysis / plotting
    # ----------------------------------------------------------------- #
    def get_major_tick_spacing(self, num_elements: int, len_seq: int,
                               custom_spacing: Optional[str]) -> int:
        """Return a sensible major tick spacing for the PAE axes."""
        if custom_spacing:
            return int(custom_spacing)
        if len_seq <= 50:
            return 10
        if num_elements < 4 and len_seq < 2000:
            return 100
        return 200

    def analyse_pae(
        self,
        pae_dict: Dict[str, Any],
        indices: Sequence[Tuple[int, int]],
        seq_titles: Sequence[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute the average PAE for every pair of subunits.

        Returns a mapping ``{model: {"Overall": v, "A vs B": v, ...}}``.
        """
        num_seqs = len(indices)
        x_labels, y_labels = self._build_axis_labels(num_seqs, seq_titles)

        average_pae_dict: Dict[str, Dict[str, float]] = {}
        for model_name, pae in pae_dict.items():
            pae = np.asarray(pae)
            average_pae_dict[model_name] = {"Overall": float(np.mean(pae))}
            count = 0
            for u in range(num_seqs):
                for v in range(num_seqs):
                    block = pae[indices[u][0]:indices[u][1],
                                indices[v][0]:indices[v][1]]
                    key = f"{x_labels[count]} vs {y_labels[count]}"
                    average_pae_dict[model_name][key] = float(np.mean(block))
                    count += 1
        return average_pae_dict

    @staticmethod
    def _build_axis_labels(num_seqs: int,
                           seq_titles: Sequence[str]) -> Tuple[List[str], List[str]]:
        """Build the flattened x/y label lists for an ``n x n`` grid."""
        y_labels: List[str] = []
        u = 0
        for i in range(num_seqs * num_seqs):
            y_labels.append(seq_titles[u])
            if (i + 1) % num_seqs == 0:
                u += 1
        x_labels = list(seq_titles[:num_seqs]) * num_seqs
        return x_labels, y_labels

    def plot_pae(
        self,
        pae_dict: Dict[str, Any],
        indices: Sequence[Tuple[int, int]],
        seq_titles: Sequence[str],
        custom_spacing: Optional[str] = None,
        custom_aa_start: Optional[str] = None,
        custom_labels: Optional[str] = None,
    ) -> None:
        """Plot PAE matrices for every model.

        Adapted from the ColabFold AlphaFold2 notebook.
        """
        num_seqs = len(indices)

        # Parse optional custom start residues.
        custom_aa_start_list: List[int] = []
        if custom_aa_start:
            raw = [int(x) for x in custom_aa_start.split(",")]
            custom_aa_start_list = raw * num_seqs
            if len(custom_aa_start_list) != num_seqs * num_seqs:
                raise ValueError(
                    f"Number of given starting residue values "
                    f"({len(custom_aa_start_list)}) does not match the number "
                    f"of protein subunits ({num_seqs * num_seqs})."
                )

        if custom_labels:
            seq_titles = custom_labels.split(",")

        x_labels, y_labels = self._build_axis_labels(num_seqs, seq_titles)
        plt.rc("xtick", labelsize=6)
        plt.rc("ytick", labelsize=6)
        plt.rc("font", **{"family": "sans-serif", "size": 8})

        for model_name, pae in pae_dict.items():
            pae = np.asarray(pae)

            # Compute relative heights (one entry per subunit row).
            heights = []
            for u in range(num_seqs):
                block = pae[indices[u][0]:indices[u][1], indices[0][0]:indices[0][1]]
                heights.append(block.shape[0])
            max_height = max(heights)
            heights = [h / max_height for h in heights]

            fig, axs = plt.subplots(
                num_seqs,
                num_seqs,
                squeeze=False,
                gridspec_kw={"height_ratios": heights, "width_ratios": heights},
                constrained_layout=True,
            )

            im = None
            count = 0
            for u in range(num_seqs):
                for v in range(num_seqs):
                    data = pae[indices[u][0]:indices[u][1],
                               indices[v][0]:indices[v][1]]
                    im = axs[u, v].im = axs[u, v].imshow(
                                    data,
                                    cmap="bwr",
                                    vmin=0,
                                    vmax=30,
                                    origin="upper",          # replaces invert_yaxis()
                                    interpolation="nearest",
                                    aspect="auto",
                                )

                    # Labels only on the outer edges.
                    if u == num_seqs - 1:
                        axs[u, v].set_xlabel(f"{x_labels[count]} [aa]")
                    if v % num_seqs == 0:
                        axs[u, v].set_ylabel(f"{y_labels[count]} [aa]")

                    # Tick positioning (optionally using custom start residues).
                    if custom_aa_start_list:
                        # X axis shows subunit v, y axis shows subunit u.
                        start_tick_x = custom_aa_start_list[count]
                        start_tick_y = custom_aa_start_list[v * num_seqs + u]
                        major_start_x = round(start_tick_x / 100) * 100
                        minor_start_x = round(start_tick_x / 10) * 10
                        major_start_y = round(start_tick_y / 100) * 100
                        minor_start_y = round(start_tick_y / 10) * 10
                    else:
                        start_tick_x = start_tick_y = 0
                        major_start_x = major_start_y = 0
                        minor_start_x = minor_start_y = 0

                    len_y, len_x = data.shape
                    major_x = self.get_major_tick_spacing(num_seqs, len_x, custom_spacing)
                    major_y = self.get_major_tick_spacing(num_seqs, len_y, custom_spacing)

                    axs[u, v].set_xticks(
                        np.arange(major_start_x - start_tick_x, len_x, major_x))
                    axs[u, v].set_xticks(
                        np.arange(minor_start_x - start_tick_x, len_x, major_x / 10),
                        minor=True)
                    axs[u, v].set_xticklabels(
                        np.arange(major_start_x, len_x + start_tick_x, major_x))

                    axs[u, v].set_yticks(
                        np.arange(major_start_y - start_tick_y, len_y, major_y))
                    axs[u, v].set_yticks(
                        np.arange(minor_start_y - start_tick_y, len_y, major_y / 10),
                        minor=True)
                    axs[u, v].set_yticklabels(
                        np.arange(major_start_y, len_y + start_tick_y, major_y))

                    count += 1

            if im is not None:
                cbar = plt.colorbar(im, ax=axs, shrink=0.8)
                cbar.set_label(
                    r"Predicted aligned error (PAE) [$\AA$]",
                    rotation=90, labelpad=10,
                )

            png_path = os.path.join(self.results_dir, f"pae_{model_name}.png")
            svg_path = os.path.join(self.results_dir, f"pae_{model_name}.svg")
            fig.savefig(png_path, dpi=300)
            #fig.savefig(svg_path, dpi=300)
            plt.close(fig)

    @staticmethod
    def get_image_path_list(results: Sequence[Tuple[str, Any]]) -> List[Tuple[str, str]]:
        """Return ``(model_name, png_filename)`` pairs for the template."""
        return [(item[0], f"pae_{item[0]}.png") for item in results]

    @staticmethod
    def get_pae_messages(results: Sequence[Tuple[str, Dict[str, float]]]) -> str:
        """Return a human readable confidence message for the best PAE."""
        best_result = float(results[0][1]["Overall"])
        if best_result >= 20.0:
            return (
                "The best overall PAE is higher than 20 A. This does not "
                "indicate a high confidence. The prediction is likely incorrect."
            )
        if 10.0 < best_result < 20.0:
            return (
                "The best overall PAE is between 10 and 20 A. This indicates an "
                "intermediate confidence and prediction might be close to reality."
            )
        return (
            "The best overall PAE is lower than 10 A. This indicates a high "
            "confidence and prediction is likely correct."
        )

    def get_chain_color(self) -> List[Tuple[str, str]]:
        """Return ``(chain_id, colour)`` pairs for the first model."""
        structure = list(self.cif_dict.values())[0]
        model = MMCIFParser(QUIET=True).get_structure("structure", structure)[0]
        chains = [str(c.get_id()) for c in model.get_chains()]
        return list(zip(chains, CHAIN_COLORS))


# --------------------------------------------------------------------------- #
# Batch evaluation
# --------------------------------------------------------------------------- #
class EvaluationPipelineBatch:
    """Aggregate many ``best_scores.json`` files into one summary report."""

    def __init__(self, results_dir: str, pairwise: bool = False) -> None:
        self.results_dir = results_dir
        self.pairwise = pairwise
        self.plotly_dict: Dict[str, str] = {}
        self.proteins: List[str] = []
        self.scores: List[Dict[str, Any]] = []

    def run(self) -> None:
        """Collect scores, build plots and write the HTML summary."""
        self.process_scores()
        self.write_html()
        plt.close("all")

    # ----------------------------------------------------------------- #
    # Name helpers
    # ----------------------------------------------------------------- #
    @staticmethod
    def remove_split(x: str) -> str:
        """Strip any ``_splitN`` suffixes from a protein name."""
        return re.sub(r"_split\d+", "", x)

    def split_by_protein_name(self, item: str, protein_num: int,
                              remove_split: bool) -> Optional[str]:
        """Determine which protein (0 or 1) ``item`` corresponds to.

        Used to build pairwise heat‑maps.  Returns ``None`` when fewer than
        two known proteins can be matched.
        """
        found_names = []
        for protein in self.proteins:
            m = re.search(r"(_(?:bait|prey)_(?:(?!_split\d+).)*)", protein)
            if m:
                protein = protein.replace(m.group(1), "")
            protein_name = re.sub(r"_split\d+", "", protein) if remove_split else protein
            if re.search(re.escape(protein_name), item):
                found_names.append(protein_name)

        # Deterministic order (set iteration order is not stable across runs).
        found_names = sorted(set(found_names))
        if len(found_names) < 2:
            return None
        if len(found_names) > 2:
            logging.warning(
                f"Ambiguous protein match for '{item}': {found_names}; "
                f"using the first two alphabetically."
            )

        first, second = found_names[0], found_names[1]
        if item.startswith(first):
            return first if protein_num == 0 else second
        elif item.startswith(second):
            return second if protein_num == 0 else first
        else:
            logging.error(f"Error in splitting {item}")
            return None
    # ----------------------------------------------------------------- #
    # Plot generation
    # ----------------------------------------------------------------- #
    def generate_summary_plots(self, df: pd.DataFrame, grouped: bool = False) -> None:
        """Create bar‑plots (and pairwise heat‑maps) for every score category."""
        logging.debug("Generating summary plots")

        height = max(0.2 * len(df["protein_names"]), 3)
        grouped_str = "_merged_" if grouped else "_"

        categories = [
            "max_multimer_score_value",
            "max_iptm_value",
            "max_ptm_value",
            "min_pae_value",
        ]

        for category in categories:
            # Sort / set range according to the score direction.
            if category == "min_pae_value":
                df = df.sort_values(category, ascending=False)
                range_ = [0, 35]
            else:
                df = df.sort_values(category)
                range_ = [0, 1]

            # --- Ranking bar plot --------------------------------------
            fig = go.Figure(
                go.Bar(
                    y=df["protein_names"],
                    x=df[category],
                    orientation="h",
                    marker=dict(color="skyblue", line=dict(color="black", width=0.5)),
                )
            )
            base_height = 300
            plot_height = base_height + df["protein_names"].size * 20
            fig.update_layout(
                xaxis=dict(title=category, range=range_, showticklabels=True,
                           nticks=10, showgrid=True),
                yaxis=dict(title="Protein Names", tickmode="linear", automargin=True),
                height=plot_height,
                margin=dict(b=50, t=50, l=100, r=50),
            )
            fig.update_yaxes(automargin=True)
            fig.update_xaxes(automargin=True)

            key = f"{category.replace('_value', '')}{grouped_str}barplot"
            self.plotly_dict[key] = fig.to_html(full_html=False)

            # --- Pairwise heat map -------------------------------------
            if "protein_1" not in df.columns and "protein_2" not in df.columns:
                df["protein_1"] = df["protein_names"].apply(
                    lambda x, rm=grouped: self.split_by_protein_name(x, 0, rm))
                df["protein_2"] = df["protein_names"].apply(
                    lambda x, rm=grouped: self.split_by_protein_name(x, 1, rm))

            if self.pairwise:
                df_swapped = df.copy()
                df_swapped["protein_1"], df_swapped["protein_2"] = (
                    df["protein_2"], df["protein_1"])
                df_combined = pd.concat([df, df_swapped], ignore_index=True)

                if category == "min_pae_value":
                    heatmap_data = df_combined.pivot_table(
                        index="protein_1", columns="protein_2",
                        values=category, aggfunc="min")
                else:
                    # Clamp negative scores to 0 for the affected column only.
                    df_combined.loc[df_combined[category] < 0, category] = 0
                    heatmap_data = df_combined.pivot_table(
                        index="protein_1", columns="protein_2",
                        values=category, aggfunc="max")

                fig = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns.tolist(),
                        y=heatmap_data.index.tolist(),
                        colorscale="Viridis",
                    )
                )
                fig.update_layout(
                    xaxis=dict(title="Protein 2", showgrid=True, gridwidth=1,
                               gridcolor="lightgray", tickmode="linear"),
                    yaxis=dict(title="Protein 1", showgrid=True, gridwidth=1,
                               gridcolor="lightgray", tickmode="linear"),
                    height=base_height + df["protein_names"].size * 10,
                    margin=dict(b=50, t=50, l=100, r=50),
                )
                key = f"{category.replace('_value', '')}{grouped_str}heatmap"
                self.plotly_dict[key] = fig.to_html(full_html=False)

    # ----------------------------------------------------------------- #
    # Score collection / processing
    # ----------------------------------------------------------------- #
    def collect_scores(self) -> List[Dict[str, Any]]:
        """Walk ``results_dir`` and load every ``best_scores.json`` found."""
        score_list = []
        for root, _folders, files in os.walk(self.results_dir, followlinks=True):
            for f in files:
                if f == "best_scores.json":
                    with open(os.path.join(root, f)) as h:
                        score_list.append(json.load(h))
        return score_list

    def process_scores(self) -> None:
        """Load scores, build data frames, mappings and all summary plots."""
        self.scores = self.collect_scores()
        if not self.scores:
            raise FileNotFoundError(
                f"No best_scores.json files found under {self.results_dir}")

        self.proteins = []
        for score_dict in self.scores:
            self.proteins.extend(score_dict["subunits"])

        df = pd.DataFrame(self.scores)
        value_columns = df.filter(like="value").columns
        df[value_columns] = df[value_columns].astype(float)

        self.plotly_dict = {}
        self.generate_summary_plots(df)

        found_split = any(
            re.search(r"_split\d+", score["protein_names"]) for score in self.scores
        )

        if found_split:
            df["protein_names"] = df["protein_names"].apply(self.remove_split)

            df_pae = (df.groupby("protein_names")["min_pae_value"].min()
                        .reset_index().sort_values("min_pae_value"))
            df_iptm = (df.groupby("protein_names")["max_iptm_value"].max()
                         .reset_index().sort_values("max_iptm_value", ascending=False))
            df_ptm = (df.groupby("protein_names")["max_ptm_value"].max()
                        .reset_index().sort_values("max_ptm_value", ascending=False))
            df_multimer = (df.groupby("protein_names")["max_multimer_score_value"].max()
                             .reset_index()
                             .sort_values("max_multimer_score_value", ascending=False))

            self.pae_list_grouped = df_pae.values.tolist()
            self.iptm_list_grouped = df_iptm.values.tolist()
            self.ptm_list_grouped = df_ptm.values.tolist()
            self.multimer_score_list_grouped = df_multimer.values.tolist()

            df_merged = df_pae.merge(df_iptm, on="protein_names", how="inner")
            df_merged = df_merged.merge(df_ptm, on="protein_names", how="inner")
            df_merged = df_merged.merge(df_multimer, on="protein_names", how="inner")
            df_merged.to_csv(
                os.path.join(self.results_dir,
                             "regrouped_pairwise_prediction_results.csv"),
                index=False,
            )
            self.generate_summary_plots(df_merged, grouped=True)
        else:
            self.pae_list_grouped = None
            self.iptm_list_grouped = None
            self.ptm_list_grouped = None
            self.multimer_score_list_grouped = None
            df.to_csv('pairwise_prediction_results.csv')

        # Build sorted {protein_names: (value, [model_name])} mappings.
        self.protein_pae_mapping = self._build_mapping(
            "min_pae_value", "min_pae_model_name", reverse=False)
        self.protein_iptm_mapping = self._build_mapping(
            "max_iptm_value", "max_iptm_model_name", reverse=True)
        self.protein_ptm_mapping = self._build_mapping(
            "max_ptm_value", "max_ptm_model_name", reverse=True)
        self.protein_multimer_score_mapping = self._build_mapping(
            "max_multimer_score_value", "max_multimer_score_model_name", reverse=True)

    def _build_mapping(self, value_key: str, model_key: str,
                       reverse: bool) -> Dict[str, Tuple[Any, List[str]]]:
        """Build a sorted ``{protein: (value, [model])}`` mapping."""
        mapping = {
            score["protein_names"]: (score[value_key], [score[model_key]])
            for score in self.scores
        }
        return dict(
            sorted(mapping.items(), key=lambda item: float(item[1][0]), reverse=reverse)
        )

    # ----------------------------------------------------------------- #
    # HTML output
    # ----------------------------------------------------------------- #
    def write_html(self) -> None:
        """Render the batch summary HTML report."""
        templates_path = _resource_path("guifold3", "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template("batch_summary.html")

        def _clean(mapping: Dict[str, Any]) -> Dict[str, Any]:
            return {k.replace("-", "_"): v for k, v in mapping.items()}

        rendered = template.render(
            protein_pae_mapping=_clean(self.protein_pae_mapping),
            protein_iptm_mapping=_clean(self.protein_iptm_mapping),
            protein_ptm_mapping=_clean(self.protein_ptm_mapping),
            protein_multimer_score_mapping=_clean(self.protein_multimer_score_mapping),
            pae_list_grouped=self.pae_list_grouped,
            iptm_list_grouped=self.iptm_list_grouped,
            ptm_list_grouped=self.ptm_list_grouped,
            multimer_score_list_grouped=self.multimer_score_list_grouped,
            plotly=self.plotly_dict,
            pairwise=self.pairwise,
        )

        html_path = os.path.join(self.results_dir, "batch_summary.html")
        with open(html_path, "w") as fout:
            fout.write(rendered)
        logging.info(f"Summary HTML written to {html_path}")


# --------------------------------------------------------------------------- #
# Stand‑alone helpers
# --------------------------------------------------------------------------- #
def get_score_dict(csv_file: str) -> List[Dict[str, Optional[str]]]:
    """Parse a scores CSV into a list of dicts (``'None'`` → ``None``)."""
    with open(csv_file) as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) <= 1:
        return []

    keys = lines[0].split(",")
    scores = []
    for line in lines[1:]:
        values = [None if v == "None" else v for v in line.split(",")]
        row = dict(zip(keys, values))
        if row.get("protein_names") is not None:
            scores.append(row)
    return scores


def run_evaluation(results_dir: str, args: argparse.Namespace) -> None:
    """Run a single evaluation, swallowing/logging exceptions (batch worker)."""
    try:
        if not os.path.exists(os.path.join(results_dir, "best_scores.json")) or args.overwrite:
            EvaluationPipeline(
                results_dir=results_dir,
                continue_from_existing_results=args.continue_from_existing_results,
                custom_spacing=args.custom_spacing,
                custom_start_residue_list=args.custom_start_residue_list,
                custom_axis_label_list=args.custom_axis_label_list,
                pairwise=args.pairwise,
            ).run_pipeline()
    except Exception as exc:  # noqa: BLE001 - batch workers must not crash the pool
        logging.error(f"Exception in process: {exc}")
        traceback.print_exc()

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaFold3-style prediction output."
    )
    parser.add_argument("--results_dir", default=None,
                        help="Path to results (predictions).")
    parser.add_argument("--run_batch_eval", action="store_true", default=False,
                        help="Evaluate multiple predictions.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="(Optional) Debug mode.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run evaluation even if best_scores.json exists.")
    parser.add_argument("--pairwise", action="store_true",
                        help="Treat predictions as pairwise interactions (only with --run_batch_eval); "
                             "subunits are derived from the results folder name, separated by \"_\".")
    parser.add_argument("--continue_from_existing_results", action="store_true",
                        default=False,
                        help="(Optional) Continue from previously extracted PAE values.")
    parser.add_argument("--custom_spacing", default=None,
                        help="(Optional) Custom spacing of major tick labels.")
    parser.add_argument(
        "--custom_start_residue_list", default=None,
        help="(Optional) Custom axis minimum numbers. For predictions done with "
             "internal protein segments the real residue numbering can be restored "
             "by giving the starting residue numbers of each segment separated by a "
             "comma, e.g. --custom_start_residue_list 15,50",
    )
    parser.add_argument(
        "--custom_axis_label_list", default=None,
        help="(Optional) Custom axis labels. Labels for different subunits must be "
             "separated by a comma and given in the same order as the sequences in "
             'the fasta file, e.g. --custom_axis_label_list "Protein A","Protein B"',
    )
    parser.add_argument("--num_cpu", type=int, default=1,
                        help="Number of CPUs for batch evaluation.")
    parser.add_argument("--chain_id_name_mapping_path", type=str, default=None,
                        help="Path to the chain_id -> protein name mapping JSON file.")
    return parser


def _run_batch(args: argparse.Namespace) -> None:
    """Run the evaluation over every sub-directory then build the summary."""
    manager = multiprocessing.Manager()
    task_queue = manager.Queue()

    for entry in os.listdir(args.results_dir):
        results_dir = os.path.join(args.results_dir, entry)
        if not os.path.isdir(results_dir):
            continue
        # In the pairwise case protein names are expected to be '_' separated.
        if args.pairwise and "_" not in entry:
            continue
        task_queue.put((results_dir, args))

    # Process the queue in waves of ``num_cpu`` worker processes.
    while not task_queue.empty():
        active_processes = []
        for _ in range(args.num_cpu):
            if task_queue.empty():
                break
            res_dir, task_args = task_queue.get()
            p = multiprocessing.Process(target=run_evaluation, args=(res_dir, task_args))
            active_processes.append(p)
            p.start()
        for p in active_processes:
            p.join()

    EvaluationPipelineBatch(results_dir=args.results_dir, pairwise=args.pairwise).run()


def main() -> None:
    """Command line entry point."""
    # Force CPU execution for jax (evaluation is light and avoids GPU conflicts).
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    jax.config.update("jax_platform_name", "cpu")

    parser = _build_arg_parser()
    args, _unknown = parser.parse_known_args()
    if not args.results_dir:
        raise SystemExit("--results_dir is required")

    logging.set_verbosity(logging.DEBUG if args.debug else logging.INFO)

    if not args.results_dir:
        raise SystemExit("--results_dir is required")

    if args.run_batch_eval:
        _run_batch(args)
    else:
        EvaluationPipeline(
            results_dir=args.results_dir,
            continue_from_existing_results=args.continue_from_existing_results,
            custom_spacing=args.custom_spacing,
            custom_start_residue_list=args.custom_start_residue_list,
            custom_axis_label_list=args.custom_axis_label_list,
            chain_id_name_mapping_file=args.chain_id_name_mapping_path,
            pairwise=args.pairwise,
        ).run_pipeline()


if __name__ == "__main__":
    main()
