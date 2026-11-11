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
af3_batch.py -- Batch driver for AlphaFold 3 (monomers and binary complexes).

Reads protein (pairs) from a CSV file or a list of AF3 jobs from a JSON file,
writes AlphaFold 3 input JSONs and runs ``run_alphafold.py`` on one or more
GPUs, with a configurable number of concurrent jobs per GPU.

Modes (can be combined; executed in this order):

1. ``--features``
       Run only the AF3 data pipeline (MSA + template search) once per unique
       sequence found in the CSV. Produces ``<name>/<name>_data.json``.
2. ``--use_precomputed_features``
       Build prediction jobs that re-use the MSAs/templates from step 1
       (optionally dropping them with ``--no_msa`` / ``--no_templates``).
   ``--no_msa --no_templates`` (without precomputed features)
       Build single-sequence prediction jobs with no MSA and no templates.
3. ``--batch_json``
       Run a ready-made list of AF3 job dictionaries.

Already finished jobs (output dir contains ``ranking_scores*``) are skipped
unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import string
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from shutil import rmtree
from typing import IO, Dict, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("af3_batch")

DEFAULT_RUN_ALPHAFOLD = (
    "/tungstenfs/scratch/gstructb_share/modules/software/alphafold/"
    "alphafold3/alphafold3/run_alphafold.py"
)
DEFAULT_JAX_CACHE = "/tachyon/temp/gstructb/"
DEFAULT_BUCKETS = (
    "64,128,256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120"
)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Show defaults and keep the raw formatting of description/epilog."""


EPILOG = """\
examples:
  # 1) Compute MSAs/templates for every unique sequence in the CSV
  af3_batch.py --csv pairs.csv --features \\
      --name_column1 gene_a --seq_column1 seq_a \\
      --name_column2 gene_b --seq_column2 seq_b

  # 2) Predict all pairs on 4 GPUs, re-using the precomputed features
  af3_batch.py --csv pairs.csv --use_precomputed_features --num_gpus 4 \\
      --name_column1 gene_a --seq_column1 seq_a \\
      --name_column2 gene_b --seq_column2 seq_b

  # 1+2 in one call
  af3_batch.py --csv pairs.csv --features --use_precomputed_features ...

  # Single-sequence mode (no MSA, no templates), 2 jobs per GPU
  af3_batch.py --csv mono.csv --no_msa --no_templates --jobs_per_gpu 2 \\
      --name_column1 name --seq_column1 sequence

  # Run a JSON list of ready-made AF3 jobs
  af3_batch.py --batch_json jobs.json --num_gpus 2
"""


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG, formatter_class=_HelpFormatter
    )

    modes = p.add_argument_group("run modes")
    modes.add_argument("--features", action="store_true",
                       help="Run only the AF3 data pipeline (MSA/template "
                            "search) for every unique sequence in --csv.")
    modes.add_argument("--use_precomputed_features", action="store_true",
                       help="Predict CSV rows re-using <name>/<name>_data.json "
                            "from --precomputed_features_dir.")
    modes.add_argument("--no_msa", action="store_true",
                       help="Do not use MSAs. Together with --no_templates "
                            "(and without --use_precomputed_features) runs "
                            "pure single-sequence predictions.")
    modes.add_argument("--no_templates", action="store_true",
                       help="Do not use templates.")
    modes.add_argument("--batch_json", default=None,
                       help="Path to a JSON file containing a list of AF3 job "
                            "dicts (each must have a 'name').")
    modes.add_argument("--prediction", action="store_true",
                       help="Deprecated, has no effect (kept for backwards "
                            "compatibility).")

    csv_g = p.add_argument_group("CSV input")
    csv_g.add_argument("--csv", default=None, help="Path to input CSV.")
    csv_g.add_argument("--name_column1", default=None,
                       help="Column with the name of protein 1.")
    csv_g.add_argument("--seq_column1", default=None,
                       help="Column with the sequence of protein 1.")
    csv_g.add_argument("--id_column1", default=None,
                       help="Optional column with an ID appended to name 1 "
                            "as '<name>-<id>'.")
    csv_g.add_argument("--name_column2", default=None,
                       help="Column with the name of protein 2 (pair mode).")
    csv_g.add_argument("--seq_column2", default=None,
                       help="Column with the sequence of protein 2 (pair mode).")
    csv_g.add_argument("--id_column2", default=None,
                       help="Optional column with an ID appended to name 2.")
    csv_g.add_argument("--precomputed_features_dir", default=None,
                       help="Directory with feature folders. "
                            "Defaults to --output_dir.")

    run_g = p.add_argument_group("execution")
    run_g.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use (IDs 0..N-1).")
    run_g.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                       help="Explicit GPU IDs; overrides --num_gpus.")
    run_g.add_argument("--jobs_per_gpu", type=int, default=1,
                       help="Concurrent prediction jobs per GPU.")
    run_g.add_argument("--feature_workers", type=int, default=1,
                       help="Concurrent data-pipeline jobs for --features "
                            "(CPU bound; each jackhmmer uses several threads).")
    run_g.add_argument("--wait_for_idle_gpu", action="store_true",
                       help="Do not start a job on a GPU that already runs "
                            "foreign compute processes (checked via nvidia-smi).")
    run_g.add_argument("--poll_interval", type=float, default=10.0,
                       help="Seconds between job status checks.")
    run_g.add_argument("--overwrite", action="store_true",
                       help="Re-run jobs that already finished.")
    run_g.add_argument("--remove_unfinished", action="store_true",
                       help="Delete output dirs of unfinished jobs before "
                            "re-running (dirs holding *_data.json features "
                            "are never deleted).")

    af3_g = p.add_argument_group("AlphaFold 3 options")
    af3_g.add_argument("--run_alphafold", default=DEFAULT_RUN_ALPHAFOLD,
                       help="Path to run_alphafold.py.")
    af3_g.add_argument("--python", default="python3",
                       help="Python interpreter of the AF3 environment.")
    af3_g.add_argument("--flash_attention", default="xla",
                       choices=["xla", "triton", "cudnn"],
                       help="--flash_attention_implementation passed to AF3.")
    af3_g.add_argument("--jax_cache_dir", default=DEFAULT_JAX_CACHE,
                       help="JAX compilation cache dir ('' to disable).")
    af3_g.add_argument("--buckets", default=DEFAULT_BUCKETS,
                       help="Token buckets passed to AF3.")
    af3_g.add_argument("--seeds", type=int, nargs="+", default=[42],
                       help="Model seeds for generated jobs.")
    af3_g.add_argument("--output_dir", default=".",
                       help="AF3 output directory.")
    af3_g.add_argument("--json_dir", default=".",
                       help="Where generated AF3 input JSONs are written.")

    log_g = p.add_argument_group("logging")
    log_g.add_argument("--log_file", default="debug.log",
                       help="Main log file.")
    log_g.add_argument("--log_dir", default="af3_logs",
                       help="Directory for per-job AF3 stdout/stderr logs.")
    log_g.add_argument("-v", "--verbose", action="store_true",
                       help="Enable debug logging.")
    return p


def validate_args(parser: argparse.ArgumentParser,
                  args: argparse.Namespace) -> None:
    """Check argument combinations and abort with a helpful message if invalid.

    Args:
        parser: The parser (used for ``parser.error``).
        args: Parsed arguments.
    """
    single_seq = args.no_msa and args.no_templates
    csv_mode = args.features or args.use_precomputed_features or single_seq

    if not (csv_mode or args.batch_json):
        parser.error("No run mode selected. Use --features, "
                     "--use_precomputed_features, --no_msa --no_templates "
                     "or --batch_json.")
    if csv_mode:
        if not args.csv:
            parser.error("--csv is required for the selected mode.")
        if not args.name_column1 or not args.seq_column1:
            parser.error("--name_column1 and --seq_column1 are required "
                         "with --csv.")
    if bool(args.name_column2) != bool(args.seq_column2):
        parser.error("--name_column2 and --seq_column2 must be given together.")
    if (args.no_msa != args.no_templates) and not args.use_precomputed_features:
        parser.error("--no_msa or --no_templates alone requires "
                     "--use_precomputed_features.")
    for opt in ("num_gpus", "jobs_per_gpu", "feature_workers"):
        if getattr(args, opt) < 1:
            parser.error(f"--{opt} must be >= 1.")
    if not os.path.isfile(args.run_alphafold):
        parser.error(f"run_alphafold.py not found: {args.run_alphafold}")
    if args.batch_json and not os.path.isfile(args.batch_json):
        parser.error(f"--batch_json not found: {args.batch_json}")
    if args.prediction:
        print("Note: --prediction is deprecated and ignored.", file=sys.stderr)
    if args.precomputed_features_dir is None:
        args.precomputed_features_dir = args.output_dir


def setup_logging(log_file: str, verbose: bool) -> None:
    """Configure logging to stderr and ``log_file``."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


# --------------------------------------------------------------------------- #
# Helpers: naming, CSV parsing, output state
# --------------------------------------------------------------------------- #
def sanitised_name(name: str) -> str:
    """Return the directory/file stem AF3 uses for a job name.

    Mirrors ``folding_input.Input.sanitised_name()`` of AlphaFold 3:
    lower-case, spaces -> '_', and only ``[a-z0-9_.-]`` kept.
    """
    lower = str(name).lower().replace(" ", "_")
    allowed = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(c for c in lower if c in allowed)


def _has_value(row: pd.Series, column: Optional[str]) -> bool:
    """True if ``column`` was given, exists in ``row`` and is not NaN/empty."""
    return (column is not None and column in row.index
            and pd.notna(row[column]) and str(row[column]).strip() != "")


def protein_name(row: pd.Series, name_col: str, id_col: Optional[str]) -> str:
    """Build a protein name ``<name>`` or ``<name>-<id>`` from a CSV row."""
    name = str(row[name_col]).strip()
    if _has_value(row, id_col):
        name = f"{name}-{str(row[id_col]).strip()}"
    return name


def get_sequence(row: pd.Series, seq_col: str) -> Optional[str]:
    """Return the whitespace-free sequence in ``seq_col`` or None if missing."""
    if not _has_value(row, seq_col):
        return None
    return "".join(str(row[seq_col]).split())


def chain_specs(args: argparse.Namespace) -> List[Tuple[str, Optional[str], str]]:
    """Return ``(name_col, id_col, seq_col)`` for each chain (1 or 2)."""
    specs = [(args.name_column1, args.id_column1, args.seq_column1)]
    if args.name_column2 and args.seq_column2:
        specs.append((args.name_column2, args.id_column2, args.seq_column2))
    return specs


def load_csv(args: argparse.Namespace,
             parser: argparse.ArgumentParser) -> pd.DataFrame:
    """Load the CSV as strings and verify that all requested columns exist."""
    df = pd.read_csv(args.csv, dtype=str)
    wanted = [c for c in (args.name_column1, args.seq_column1, args.id_column1,
                          args.name_column2, args.seq_column2, args.id_column2)
              if c]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        parser.error(f"Columns not found in {args.csv}: {missing}. "
                     f"Available: {list(df.columns)}")
    LOGGER.info("Loaded %d rows from %s", len(df), args.csv)
    return df


def is_finished(job_dir: str) -> bool:
    """True if an AF3 output dir contains a ``ranking_scores*`` file."""
    return os.path.isdir(job_dir) and any(
        "ranking_scores" in f for f in os.listdir(job_dir))


def should_run(name: str, args: argparse.Namespace) -> bool:
    """Decide whether a prediction job must be (re-)run.

    Finished jobs are skipped unless ``--overwrite``. Unfinished output dirs are
    deleted with ``--remove_unfinished`` unless they contain feature files.
    """
    sname = sanitised_name(name)
    job_dir = os.path.join(args.output_dir, sname)
    if not os.path.exists(job_dir):
        return True
    if is_finished(job_dir):
        if args.overwrite:
            LOGGER.info("%s: finished, overwriting (--overwrite).", name)
            return True
        LOGGER.info("%s: already finished, skipping.", name)
        return False
    if args.remove_unfinished:
        if os.path.exists(os.path.join(job_dir, f"{sname}_data.json")):
            LOGGER.warning("%s: unfinished dir contains features "
                           "(%s_data.json); not removing it.", name, sname)
        else:
            LOGGER.info("%s: removing unfinished output %s", name, job_dir)
            rmtree(job_dir)
    return True


def make_job(name: str, sequences: List[dict], seeds: Sequence[int]) -> dict:
    """Return an AF3 input dict (alphafold3 dialect, version 1)."""
    return {"name": name, "modelSeeds": list(seeds), "sequences": sequences,
            "dialect": "alphafold3", "version": 1}


def write_job_json(job: dict, json_dir: str) -> str:
    """Write ``job`` to ``<json_dir>/<sanitised name>.json`` and return the path."""
    path = os.path.join(json_dir, f"{sanitised_name(job['name'])}.json")
    with open(path, "w") as fh:
        json.dump(job, fh, indent="\t")
    return path


# --------------------------------------------------------------------------- #
# Features (data pipeline)
# --------------------------------------------------------------------------- #
def build_feature_name_map(df: pd.DataFrame,
                           args: argparse.Namespace) -> Dict[str, str]:
    """Map each unique sequence to the protein name its features are stored under.

    The first occurrence of a sequence in the CSV defines the name, so
    duplicate sequences under different names are computed only once.
    """
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        for name_col, id_col, seq_col in chain_specs(args):
            seq = get_sequence(row, seq_col)
            if seq is not None:
                mapping.setdefault(seq, protein_name(row, name_col, id_col))
    return mapping


def prepare_feature_jobs(feature_map: Dict[str, str],
                         args: argparse.Namespace) -> List[str]:
    """Write one single-chain AF3 JSON per unique sequence lacking features."""
    paths = []
    for seq, name in feature_map.items():
        sname = sanitised_name(name)
        data_json = os.path.join(args.output_dir, sname, f"{sname}_data.json")
        if os.path.exists(data_json) and not args.overwrite:
            LOGGER.debug("%s: features exist, skipping.", name)
            continue
        job = make_job(name, [{"protein": {"id": "A", "sequence": seq}}],
                       args.seeds)
        paths.append(write_job_json(job, args.json_dir))
    return paths


def load_precomputed_chain(seq: str, name: str, feature_map: Dict[str, str],
                           features_dir: str) -> dict:
    """Load the protein entry of a ``<name>_data.json`` feature file.

    Looks up the file under ``name`` first, then under the name the sequence
    was computed with (see :func:`build_feature_name_map`).

    Raises:
        FileNotFoundError: If no feature file exists.
        ValueError: If the stored sequence does not match ``seq``.
    """
    candidates = [name] + ([feature_map[seq]] if seq in feature_map else [])
    for cand in dict.fromkeys(candidates):
        sname = sanitised_name(cand)
        path = os.path.join(features_dir, sname, f"{sname}_data.json")
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(f"No input features found for {name} "
                                f"(looked for {candidates} in {features_dir})")
    with open(path) as fh:
        data = json.load(fh)
    protein = data["sequences"][0]["protein"]
    if protein["sequence"] != seq:
        raise ValueError(f"Sequence mismatch between CSV and {path} for {name}")
    LOGGER.debug("%s: %d templates in %s", name,
                 len(protein.get("templates") or []), path)
    return protein


def chain_entry(chain_id: str, seq: str, precomputed: Optional[dict],
                no_msa: bool, no_templates: bool) -> dict:
    """Build an AF3 protein entry.

    Args:
        chain_id: Chain ID ("A", "B", ...).
        seq: Amino-acid sequence.
        precomputed: Protein dict from a ``_data.json`` or None for
            single-sequence mode (empty MSA and templates).
        no_msa: Replace MSAs with empty strings.
        no_templates: Replace templates with an empty list.
    """
    entry = {"id": chain_id, "sequence": seq}
    if precomputed is None or no_msa:
        entry["unpairedMsa"], entry["pairedMsa"] = "", ""
    else:
        entry["unpairedMsa"] = precomputed.get("unpairedMsa") or ""
        entry["pairedMsa"] = precomputed.get("pairedMsa") or ""
    if precomputed is None or no_templates:
        entry["templates"] = []
    else:
        entry["templates"] = precomputed.get("templates") or []
    return {"protein": entry}


# --------------------------------------------------------------------------- #
# Prediction job preparation
# --------------------------------------------------------------------------- #
def prepare_prediction_jobs(df: pd.DataFrame, args: argparse.Namespace,
                            feature_map: Dict[str, str]) -> Tuple[List[str], int]:
    """Write one AF3 JSON per CSV row (monomer or pair).

    Returns:
        ``(json_paths, n_errors)`` -- rows that fail (missing features,
        sequence mismatch, missing sequence) are logged and counted.
    """
    paths: List[str] = []
    seen: set = set()
    n_errors = 0
    for idx, row in df.iterrows():
        try:
            chains, names = [], []
            for chain_id, (name_col, id_col, seq_col) in zip("AB", chain_specs(args)):
                seq = get_sequence(row, seq_col)
                if seq is None:
                    raise ValueError(f"missing sequence in column '{seq_col}'")
                pname = protein_name(row, name_col, id_col)
                pre = (load_precomputed_chain(seq, pname, feature_map,
                                              args.precomputed_features_dir)
                       if args.use_precomputed_features else None)
                chains.append(chain_entry(chain_id, seq, pre,
                                          args.no_msa, args.no_templates))
                names.append(pname)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            LOGGER.error("Row %s skipped: %s", idx, exc)
            n_errors += 1
            continue

        name = "_".join(names)
        sname = sanitised_name(name)
        if sname in seen:
            LOGGER.warning("Row %s: duplicate job name %s, skipping.", idx, name)
            continue
        seen.add(sname)
        if not should_run(name, args):
            continue
        paths.append(write_job_json(make_job(name, chains, args.seeds),
                                    args.json_dir))
    return paths, n_errors


def prepare_batch_jobs(args: argparse.Namespace) -> List[str]:
    """Write each job of ``--batch_json`` to its own AF3 input JSON."""
    with open(args.batch_json) as fh:
        jobs = json.load(fh)
    if isinstance(jobs, dict):
        jobs = [jobs]
    paths = []
    for i, job in enumerate(jobs):
        if "name" not in job:
            LOGGER.error("Job #%d in %s has no 'name', skipping.", i, args.batch_json)
            continue
        if should_run(job["name"], args):
            paths.append(write_job_json(job, args.json_dir))
    return paths


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def gpu_compute_pids(gpu_index: int) -> List[str]:
    """Return PIDs of compute processes on a GPU (via nvidia-smi).

    Returns an empty list if nvidia-smi is unavailable or fails.
    """
    cmd = ["nvidia-smi", "-i", str(gpu_index),
           "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        LOGGER.warning("nvidia-smi not found; cannot check GPU %s.", gpu_index)
        return []
    if res.returncode != 0:
        LOGGER.warning("nvidia-smi failed for GPU %s: %s", gpu_index,
                       res.stderr.strip())
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def build_af3_command(json_path: str, args: argparse.Namespace,
                      data_pipeline_only: bool) -> List[str]:
    """Build the ``run_alphafold.py`` command line for one job."""
    cmd = [args.python, args.run_alphafold,
           f"--json_path={json_path}",
           f"--output_dir={args.output_dir}",
           "--force_output_dir",
           f"--flash_attention_implementation={args.flash_attention}",
           f"--buckets={args.buckets}"]
    if args.jax_cache_dir:
        cmd.append(f"--jax_compilation_cache_dir={args.jax_cache_dir}")
    if data_pipeline_only:
        cmd += ["--run_data_pipeline", "--norun_inference"]
    return cmd


@dataclass
class _RunningJob:
    """Book-keeping for a launched AF3 subprocess."""
    json_path: str
    slot: Optional[int]
    process: subprocess.Popen
    log_handle: IO
    log_path: str
    start: float = field(default_factory=time.time)


def run_jobs(json_paths: List[str], slots: List[Optional[int]],
             args: argparse.Namespace, data_pipeline_only: bool) -> Dict[str, int]:
    """Run AF3 jobs concurrently, one per free slot.

    Args:
        json_paths: AF3 input JSON files.
        slots: One entry per concurrent job; an int is a GPU ID (sets
            ``CUDA_VISIBLE_DEVICES``), ``None`` means "do not touch GPU env".
        args: Parsed arguments.
        data_pipeline_only: Run only the data pipeline (features).

    Returns:
        Mapping ``json_path -> return code`` (-1 if the launch failed).
    """
    queue = deque(json_paths)
    free = deque(slots)
    running: List[_RunningJob] = []
    results: Dict[str, int] = {}
    total = len(json_paths)
    LOGGER.info("%d job(s) queued on %d slot(s).", total, len(slots))

    try:
        while queue or running:
            progressed = False

            # Launch jobs on free slots.
            for _ in range(len(free)):
                if not queue:
                    break
                slot = free.popleft()
                if (args.wait_for_idle_gpu and slot is not None
                        and not any(r.slot == slot for r in running)
                        and gpu_compute_pids(slot)):
                    LOGGER.info("GPU %s busy with foreign processes, waiting.", slot)
                    free.append(slot)
                    continue
                json_path = queue.popleft()
                env = os.environ.copy()
                if slot is not None:
                    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
                    env["CUDA_VISIBLE_DEVICES"] = str(slot)
                cmd = build_af3_command(json_path, args, data_pipeline_only)
                stem = os.path.splitext(os.path.basename(json_path))[0]
                suffix = "_features" if data_pipeline_only else ""
                log_path = os.path.join(args.log_dir, f"{stem}{suffix}.log")
                log_fh = open(log_path, "w")
                LOGGER.info("[%d/%d] Starting %s on slot %s",
                            total - len(queue), total, json_path, slot)
                LOGGER.debug("Command: %s", " ".join(cmd))
                try:
                    proc = subprocess.Popen(cmd, stdout=log_fh,
                                            stderr=subprocess.STDOUT, env=env)
                except OSError as exc:
                    LOGGER.error("Failed to launch %s: %s", json_path, exc)
                    log_fh.close()
                    results[json_path] = -1
                    free.append(slot)
                    continue
                running.append(_RunningJob(json_path, slot, proc, log_fh, log_path))
                progressed = True

            # Collect finished jobs (build a new list; never mutate while iterating).
            still_running = []
            for job in running:
                rc = job.process.poll()
                if rc is None:
                    still_running.append(job)
                    continue
                job.log_handle.close()
                elapsed = time.time() - job.start
                results[job.json_path] = rc
                if rc == 0:
                    LOGGER.info("Finished %s in %.1f s (slot %s).",
                                job.json_path, elapsed, job.slot)
                else:
                    LOGGER.error("FAILED %s (exit %d, %.1f s). See %s",
                                 job.json_path, rc, elapsed, job.log_path)
                free.append(job.slot)
                progressed = True
            running = still_running

            if running or queue:
                LOGGER.debug("%d running, %d queued, free slots: %s",
                             len(running), len(queue), list(free))
                if not progressed:
                    time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; terminating %d running job(s).", len(running))
        for job in running:
            job.process.terminate()
        for job in running:
            try:
                job.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                job.process.kill()
            job.log_handle.close()
        raise

    n_fail = sum(1 for rc in results.values() if rc != 0)
    LOGGER.info("Done: %d succeeded, %d failed.", len(results) - n_fail, n_fail)
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns 0 on success, 1 if any job failed."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    setup_logging(args.log_file, args.verbose)
    for d in (args.output_dir, args.json_dir, args.log_dir):
        os.makedirs(d, exist_ok=True)

    gpu_ids = args.gpu_ids if args.gpu_ids else list(range(args.num_gpus))
    # Round-robin so concurrent jobs are spread across GPUs.
    gpu_slots: List[Optional[int]] = [g for _ in range(args.jobs_per_gpu)
                                      for g in gpu_ids]
    failures = 0
    df = load_csv(args, parser) if args.csv else None

    if args.features:
        feature_map = build_feature_name_map(df, args)
        LOGGER.info("Number of unique sequences: %d", len(feature_map))
        paths = prepare_feature_jobs(feature_map, args)
        res = run_jobs(paths, [None] * args.feature_workers, args,
                       data_pipeline_only=True)
        failures += sum(rc != 0 for rc in res.values())

    if args.use_precomputed_features or (args.no_msa and args.no_templates):
        feature_map = (build_feature_name_map(df, args)
                       if args.use_precomputed_features else {})
        paths, n_err = prepare_prediction_jobs(df, args, feature_map)
        failures += n_err
        res = run_jobs(paths, gpu_slots, args, data_pipeline_only=False)
        failures += sum(rc != 0 for rc in res.values())

    if args.batch_json:
        paths = prepare_batch_jobs(args)
        res = run_jobs(paths, gpu_slots, args, data_pipeline_only=False)
        failures += sum(rc != 0 for rc in res.values())

    if failures:
        LOGGER.error("%d job(s)/row(s) failed.", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())