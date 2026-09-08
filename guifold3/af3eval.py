# Copyright 2022 Friedrich Miescher Institute for Biomedical Research
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

import copy
import gc
import json
import multiprocessing
import string
import traceback
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import pandas as pd
import pkg_resources
import os
import sys
import argparse
import numpy as np
from absl import logging
from Bio.PDB import MMCIFParser, Superimposer, PDBIO
from Bio import SeqIO
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import seaborn as sns
from jinja2 import Template, Environment, FileSystemLoader
import re
import typing
import logging as logging_python


import plotly.graph_objects as go

import jax
import jax.numpy as jnp
from contextlib import closing


matplotlib.use('Agg')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
logging_python.getLogger('matplotlib.font_manager').disabled = True

class EvaluationPipeline:
    def __init__(self, results_dir: str = None, continue_from_existing_results: bool = False, custom_spacing: bool = False,
                 custom_start_residue_list: str = None, custom_axis_label_list: str = None, name_id_pattern: bool = False,
                   server: bool = False, chain_id_name_mapping_file: str = None):

        self.results_dir = results_dir
        self.continue_from_existing_results = continue_from_existing_results
        self.custom_spacing = custom_spacing
        self.custom_start_residue_list = custom_start_residue_list
        self.custom_axis_label_list = custom_axis_label_list
        self.name_id_pattern = name_id_pattern
        self.confidence_metrics = [('atom_plddts', 'max'),
                                        ('pae', 'min'),
                                        ('chain_iptm', 'max'),
                                        ('chain_pair_iptm', 'max'),
                                        ('chain_pair_pae_min', 'min'),
                                        ('chain_ptm', 'max'),
                                        ('fraction_disordered', 'min'),
                                        ('has_clash', 'min'),
                                        ('iptm', 'max'),
                                        ('ptm', 'max'),
                                        ('ranking_score', 'max')]
        self.server = server
        
        #Check if input path/files exist
        if not os.path.exists(self.results_dir):
            error_msg = f"Results directory {self.results_dir} not found"
            logging.error(error_msg)
            raise SystemExit(error_msg)
        self.chain_id_name_mapping_file = chain_id_name_mapping_file
    
    def collect_files_local(self, dir_name):
        self.cif_dict = {}
        self.json_full_data_dict = {}
        self.json_summary_confidences_dict = {}
        for file in os.listdir(dir_name):
            file = os.path.join(dir_name, file)
            regex = r'seed-(\d+)_sample-(\d+)'
            if re.search(regex, file) and os.path.isdir(file):
                m = re.search(regex, file)
                seed = m.group(1)
                sample = m.group(2)
                model = f'seed-{seed}_sample-{sample}'
                for file2 in os.listdir(file):
                    file2 = os.path.join(file, file2)
                    if file2.endswith('confidences.json') and not file2.endswith('summary_confidences.json'):
                        self.json_full_data_dict[model] = file2
                    elif file2.endswith('summary_confidences.json'):
                        self.json_summary_confidences_dict[f'seed-{seed}_sample-{sample}'] = file2
                    if file2.endswith('_model.cif'):
                        self.cif_dict[model] = file2
            if file.endswith('chain_id_name_mapping.json') and not self.chain_id_name_mapping_file:
                self.chain_id_name_mapping_file = file
        if len(self.cif_dict.values()) == 0:
            raise Exception("No cif files found")
        if len(self.json_full_data_dict.values()) == 0:
            raise Exception("No confidence files found")
        if len(self.json_summary_confidences_dict.values()) == 0:
            raise Exception("No summary confidence files found")
        if not self.chain_id_name_mapping_file:
            raise Exception("No chain_id_name_mapping.json found")

                

    def find_sequence_in_dict(self, d, sequences):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == "sequence":
                    sequences.append(value)
                elif isinstance(value, dict) or isinstance(value, list):
                    self.find_sequence_in_dict(value, sequences)
        elif isinstance(d, list):
            for item in d:
                self.find_sequence_in_dict(item, sequences)

        return sequences

    def create_alphabet_list(self, n):
        if n > len(string.ascii_lowercase):
            raise ValueError("Number of sequences is higher than the length of the alphabet")
        return list(string.ascii_lowercase[:n])

    
    def get_chain_id_name_mapping(self):
        with open(self.chain_id_name_mapping_file) as fh:
            chain_id_name_mapping = json.load(fh)
        return chain_id_name_mapping
    
        
    def round_float(self, data):
        decimal_places = 2
        if isinstance(data, float):
            return round(data, decimal_places)
        elif isinstance(data, tuple):
            return tuple(self.round_float(item) for item in data)
        elif isinstance(data, list):
            return [self.round_float(item) for item in data]
        else:
            return data
    
    def convert_to_list(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, (jnp.ndarray, jax.interpreters.xla.DeviceArray)):
            return jax.device_get(obj).tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
        
    def convert_to_np(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_np(v) for k, v in obj.items()}
        elif isinstance(obj, (jnp.ndarray, jax.interpreters.xla.DeviceArray)):
            return np.array(jax.device_get(obj).tolist())
        #elif isinstance(obj, np.ndarray):
        #    return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return np.array(obj)
        else:
            return obj
        
    def convert_cif_to_pdb(self, cif_list):
        from Bio.PDB import MMCIFParser, PDBIO
        pdb_list = []
        for cif_file in cif_list:
            parser = MMCIFParser()
            structure = parser.get_structure("structure", cif_file)

            pdb_file = cif_file.replace(".cif", ".pdb")
            io = PDBIO()
            io.set_structure(structure)
            io.save(pdb_file)
            with open(pdb_file, 'r') as fin:
                content = fin.readlines()
                content = ''.join([x for x in content if x.startswith("ATOM")])
            pdb_list.append((pdb_file, content))
        return pdb_list
        
    def save_scoring_results(self):
        print(self.iptm_by_model_dict)
        ptm_model = list(self.ptm_by_model_dict.keys())[0]
        ptm = self.ptm_by_model_dict[ptm_model]
        iptm_model = list(self.iptm_by_model_dict.keys())[0]
        iptm = self.iptm_by_model_dict[iptm_model]
        multimer_score_model = list(self.ranking_score_by_model_dict.keys())[0]
        multimer_score = self.ranking_score_by_model_dict[multimer_score_model]
        inter_pae_model = list(self.inter_pae_by_model_dict.keys())[0]
        inter_pae = self.inter_pae_by_model_dict[inter_pae_model]
        best_scores = {'protein_names': '_'.join(self.seq_titles),
                                        'subunits': self.subunit_list,
                                        'max_ptm_model_name': ptm_model,
                                        'max_ptm_value': ptm,
                                        'max_iptm_model_name': iptm_model,
                                        'max_iptm_value': iptm,
                                        'min_pae_model_name': inter_pae_model,
                                        'min_pae_value': inter_pae,
                                        'max_multimer_score_model_name': multimer_score_model,
                                        'max_multimer_score_value': multimer_score}
        with open(os.path.join(self.results_dir, 'best_scores.json'), 'w') as f:
            json.dump(best_scores, f)

    def run_pipeline(self):
        logging.info(f"Running evaluation pipeline in {self.results_dir}.")

        self.collect_files_local(self.results_dir)
        self.chain_id_name_mapping =  self.get_chain_id_name_mapping()
            

        logging.info(f"Extracting results.")
        for name, _ in self.confidence_metrics:
            setattr(self, f"{name}_by_model_dict", {})
        data_by_model_dict = {}
        for model, json_full in self.json_full_data_dict.items():
            json_confidence = self.json_summary_confidences_dict[model]
            with open(json_full, 'r') as json_full_handle, open(json_confidence, 'r') as json_confidence_handle:
                data_full = json.load(json_full_handle)
                data_confidence = json.load(json_confidence_handle)
                data_full.update(data_confidence)
                data = data_full
                data_by_model_dict[model] = data
                print(data.keys())
        
        indices = []
        chain_ids = []
        for model, data in data_by_model_dict.items():
            self.pae_by_model_dict[model] = data['pae']
            chain_ids = list(set(data['token_chain_ids']))
        print(chain_ids)
        seq_titles = [self.chain_id_name_mapping[chain_id] for chain_id in chain_ids]
        print(seq_titles)
        self.subunit_list = self.seq_titles = seq_titles
        prev = 0
        for chain_id in chain_ids:
            seq_len = data['token_chain_ids'].count(chain_id) + prev
            indices.append((prev, seq_len))
            prev = seq_len
        

        #Generate PAE plots
        if not self.continue_from_existing_results:
            self.plot_pae(self.pae_by_model_dict, indices, seq_titles, self.custom_spacing, self.custom_start_residue_list,
                        self.custom_axis_label_list)

        #Get average paes
        self.average_pae_by_model_dict = self.analyse_pae(self.pae_by_model_dict,
                        indices,
                        seq_titles)
        
        self.average_pae_by_model_dict = self.sort_pae_results(self.average_pae_by_model_dict)
        
        #Fill dicts for each model with confidence values
        
        for name, _ in self.confidence_metrics:
            for model, data in data_by_model_dict.items():
                dict_ = getattr(self, f"{name}_by_model_dict")
                dict_[model] = data[name]

        #Sort dicts
        for name, direction in self.confidence_metrics:
            dict_ = getattr(self, f"{name}_by_model_dict")
            if direction == 'min':
                reverse = True
            else:
                reverse = False
            dict_ = self.sort_dict(dict_, reverse)
            dict_ = self.round_float(dict_)

        #Make dict with chain average plddts
        self.plddt_by_model_dict = {}
        for model, plddt_by_atom in self.atom_plddts_by_model_dict.items():
            chain_average_plddt = np.mean(plddt_by_atom, axis=-1)
            print(chain_average_plddt)
            self.plddt_by_model_dict[model] = chain_average_plddt


        #Make dict with chain_pair_pae
        self.inter_pae_by_model_dict = {}
        for model, combination_pae_dict in self.average_pae_by_model_dict:
            inter_pae = []
            for combination, pae in combination_pae_dict.items():
                combination_pair = combination.split(' vs ')
                print(combination_pair)
                if len(combination_pair) > 1:
                    if combination_pair[0] != combination_pair[1]:
                        print(f"Adding combination {combination_pair}")
                        inter_pae.append(pae)
            
            self.inter_pae_by_model_dict[model] = np.mean(inter_pae)

        self.confidence_metrics_names = [x[0] for x in self.confidence_metrics]
        print(self.cif_dict.values())
        chain_color = self.get_chain_color()
        
        pdb_files = self.convert_cif_to_pdb(list(self.cif_dict.values()))

        templates_path = pkg_resources.resource_filename("guifold3", "templates")
        pae_examples_path = pkg_resources.resource_filename("guifold3", "images/pae_examples.png")
        logging.info(templates_path)

        
        ### Render template
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('results.html')

        render_attributes = {f'{name}_by_model_dict': getattr(self, f"{name}_by_model_dict") for name in self.confidence_metrics_names}
        print(render_attributes.keys())

        render_attributes['pae_results'] = self.average_pae_by_model_dict
        render_attributes['pdb_path_list'] = [(os.path.basename(x[0]), x[1]) for x in pdb_files]
        render_attributes['chain_color'] = chain_color
        render_attributes['images'] = self.get_image_path_list(self.average_pae_by_model_dict)
        render_attributes['plddt_by_model_dict'] = self.plddt_by_model_dict
        render_attributes['use_model_viewer'] = True
        for k, v in render_attributes.items():
            if isinstance(v, dict):
                print(f"Len attribute {k}: {len(v)}")

        rendered = template.render(render_attributes)

        html_path = os.path.join(self.results_dir, "results_model_viewer.html")
        with open(html_path, "w") as f_out:
            f_out.write(rendered)

        plt.close('all')
        logging.info(f"Finished. Results written to {self.results_dir}.")

        self.save_scoring_results()


    def check_none(self, nested_list):
        if isinstance(nested_list, (list, tuple)):
            for x in nested_list:
                if self.check_none(x):
                    return True
            return False
        else:
            if nested_list is None:
                return True
            else:
                return False

    def get_major_tick_spacing(self, num_elements, len_seq, custom_spacing):
        if custom_spacing:
            return int(custom_spacing)
        elif len_seq <= 50:
            return 10
        elif num_elements < 4 and not len_seq >= 2000:
            return 100
        elif num_elements >= 4 or len_seq >= 2000:
            return 200
        
    def analyse_pae(self, pae_dict, indices, seq_titles):
        num_seqs = len(indices)
        matrix_len = num_seqs * num_seqs
        u = 0
        y_labels = []
        for i, _ in enumerate(range(matrix_len)):
            y_labels.append(seq_titles[u])
            if (i + 1) % num_seqs == 0:
                u += 1
        x_labels = []
        for i, _ in enumerate(range(num_seqs)):
            x_labels.append(seq_titles[i])
        x_labels = x_labels*num_seqs
        average_pae_dict = {}
        for i,(model_name, pae) in enumerate(pae_dict.items()):
            if not isinstance(pae, np.ndarray):
                pae = np.array(pae)
            average_pae_dict[model_name] = {"Overall": np.mean(pae)}
            count = 0
            for u in range(len(indices)):
                for v in range(len(indices)):
                    logging.debug(seq_titles)
                    logging.debug(type(pae))
                    logging.debug(np.array(pae).shape)
                    logging.debug(f"{indices[u][0]}:{indices[u][1]},{indices[v][0]}:{indices[v][1]}")
                    data = pae[indices[u][0]:indices[u][1],indices[v][0]:indices[v][1]]
                    average_pae = np.mean(data)
                    average_pae_dict[model_name][f"{x_labels[count]} vs {y_labels[count]}"] = average_pae
                    count += 1
        return average_pae_dict

    def plot_pae(self, pae_dict, indices, seq_titles, custom_spacing=None, custom_aa_start=None, custom_labels=None):
        """Plots PAE. Adapted from https://github.com/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"""
        num_seqs = len(indices)
        print(f'len indices {num_seqs}')
        matrix_len = num_seqs * num_seqs
        custom_aa_start_list = []
        if custom_aa_start:
            custom_aa_start_list = custom_aa_start.split(',')
            if not isinstance(custom_aa_start_list, list):
                custom_aa_start_list = [custom_aa_start_list]
            custom_aa_start_list = [int(x) for x in custom_aa_start_list] * num_seqs

            if len(custom_aa_start_list) != num_seqs*num_seqs:
                raise ValueError(f"Number of given starting residue values ({len(custom_aa_start_list)})"
                                 f" does not match the number of protein subunits ({num_seqs*num_seqs}).")

        if custom_labels:
            custom_labels_list = custom_labels.split(',')
            if not isinstance(custom_labels_list, list):
                custom_labels_list = [custom_labels_list]
            seq_titles = custom_labels_list

        u = 0
        y_labels = []
        for i, _ in enumerate(range(matrix_len)):
            y_labels.append(seq_titles[u])
            if (i + 1) % num_seqs == 0:
                u += 1
        x_labels = []
        for i, _ in enumerate(range(num_seqs)):
            x_labels.append(seq_titles[i])
        x_labels = x_labels*num_seqs
        titles_raw = zip(y_labels, x_labels)
        titles = []
        for title in titles_raw:
            title_1 = title[0]
            title_2 = title[1]
            title = f"{title_1} against {title_2}"
            titles.append(title)
        plt.rc('xtick',labelsize=6)
        plt.rc('ytick',labelsize=6)
        plt.rc('font',**{'family':'sans-serif', 'size': 8})

        #average_paes = {}
        for i, (model_name, pae) in enumerate(pae_dict.items()):
            if not isinstance(pae, np.ndarray):
                pae = np.array(pae)
            heights, widths = [], []
            #average_paes[model_name] = {"Overall": np.mean(pae)}
            #logging.debug(average_paes)
            logging.debug(f"Number of inidices {len(indices)}")
            for u in range(len(indices)):
                for v in range(len(indices)):
                    data = pae[indices[u][0]:indices[u][1],indices[v][0]:indices[v][1]]
                    num_rows = data.shape[0]
                    num_cols = data.shape[1]
                    heights.append(num_rows)
                    widths.append(num_cols)

            logging.debug(f"Len heights {len(heights)}")
            #logging.debug(heights)
            max_height = max(heights)
            #heights = list(dict.fromkeys(heights))
            #logging.debug(heights)
            heights = [h / max_height for i, h in enumerate(heights) if i % len(indices) == 0]
            #logging.debug(heights)
            fig, axs = plt.subplots(num_seqs, num_seqs, squeeze=False,
                                    gridspec_kw={'height_ratios': heights, 'width_ratios': heights},
                                    constrained_layout=True)#, sharex=True, sharey=True)
            count = 0
            num_rows_cols_list = []
            maps = []
            for u in range(len(indices)):
                for v in range(len(indices)):
                    data = pae[indices[u][0]:indices[u][1],indices[v][0]:indices[v][1]]
                    #average_pae = np.mean(data)
                    #average_paes[model_name][f"{x_labels[count]} vs {y_labels[count]}"] = average_pae
                    num_rows = data.shape[0]
                    num_cols = data.shape[1]
                    num_rows_cols_list.append((num_rows, num_cols))
                    im = axs[u, v].pcolor(data,cmap="bwr", vmin=0, vmax=30, rasterized=True)#,
                              #extent=(0, num_cols, num_rows, 0),
                              #interpolation='nearest'
                              #)
                    axs[u, v].invert_yaxis()
                    #Show labels only for bottom row
                    if u == len(indices) - 1:
                        logging.debug(f"logging.debug x label {u} {v}")
                        axs[u, v].set_xlabel(f"{x_labels[count]} [aa]")

                    if v % len(indices) == 0:
                        logging.debug(f"logging.debug y label {u} {v}")
                        axs[u, v].set_ylabel(f"{y_labels[count]} [aa]")

                    if len(custom_aa_start_list) > 0:
                        start_tick = custom_aa_start_list[count]
                        major_start_tick_rounded = round(start_tick/100)*100
                        minor_start_tick_rounded = round(start_tick/10)*10
                    else:
                        minor_start_tick_rounded = major_start_tick_rounded = start_tick = 0
                    logging.debug(f"u: {u}\nv: {v}\nx label {x_labels[count]}\ny label {y_labels[count]}\nlen_data: {len(data)}\nmajor tick spacing: {self.get_major_tick_spacing(len(indices), len(data), custom_spacing)}")
                    logging.debug(f"len_data_x: {np.size(data, 1)} len_data_y: {np.size(data, 0)}")
                    len_y_axis = np.size(data, 0)
                    len_x_axis = np.size(data, 1)
                    major_tick_spacing_x = self.get_major_tick_spacing(len(indices), len_x_axis, custom_spacing)
                    major_tick_spacing_y = self.get_major_tick_spacing(len(indices), len_y_axis, custom_spacing)
                    major_tick_positions_x = np.arange(major_start_tick_rounded - start_tick,
                                                       len_x_axis,
                                                       major_tick_spacing_x)
                    minor_tick_positions_x = np.arange(minor_start_tick_rounded - start_tick,
                                                       len_x_axis,
                                                       major_tick_spacing_x / 10)
                    major_tick_positions_y = np.arange(major_start_tick_rounded - start_tick,
                                                       len_y_axis,
                                                       major_tick_spacing_y)
                    minor_tick_positions_y = np.arange(minor_start_tick_rounded - start_tick,
                                                        len_y_axis,
                                                        major_tick_spacing_y / 10)
                    tick_labels_x = np.arange(major_start_tick_rounded,
                                              len_x_axis + start_tick,
                                              major_tick_spacing_x)
                    tick_labels_y = np.arange(major_start_tick_rounded,
                                              len_y_axis + start_tick,
                                              major_tick_spacing_y)
                    axs[u,v].set_xticks(major_tick_positions_x)
                    axs[u,v].set_xticks(minor_tick_positions_x, minor=True)
                    axs[u,v].set_xticklabels(tick_labels_x)
                    axs[u,v].set_yticks(major_tick_positions_y)
                    axs[u,v].set_yticks(minor_tick_positions_y, minor=True)
                    axs[u,v].set_yticklabels(tick_labels_y)

                        #plt.sca(axs[u, v])
                        #plt.xticks(np.arange(len(data)), np.arange(custom_aa_start_list[count],
                        #                                         len(data)+custom_aa_start_list[count]))

                    #axs[u, v].xaxis.set_major_locator(MultipleLocator(
                    #    self.get_major_tick_spacing(len(indices), len(data), custom_spacing)))
                    #axs[u, v].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                    #axs[u, v].xaxis.set_minor_locator(MultipleLocator(50))
                    #axs[u, v].yaxis.set_major_locator(MultipleLocator(100))
                    #axs[u, v].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                    #axs[u, v].yaxis.set_minor_locator(MultipleLocator(50))
                    count += 1
            #fig.subplots_adjust(hspace=0.6, wspace=0.3)
            #adjustw(ims, wspace=1)
            #divider = make_axes_locatable(plt.gca())
            #cax = divider.append_axes("right", "5%", pad="3%")
            cbar = plt.colorbar(im, ax=axs, cmap="bwr", shrink=0.8)
            cbar.set_label('Predicted aligned error (PAE) [$\AA$]', rotation=90, labelpad=10)
            #plt.tight_layout()
            fn_png = f'pae_{model_name}.png'
            fn_svg = f'pae_{model_name}.svg'
            png_path = os.path.join(self.results_dir, fn_png)
            svg_path = os.path.join(self.results_dir, fn_svg)
            plt.savefig(png_path, dpi=300)
            plt.savefig(svg_path, dpi=300)
            plt.close(fig)
        #logging.debug(average_paes)
        #return average_paes
        #plt.show()
        #plt.savefig(os.path.join(output_dir, f'pae_{model_name}.png'))#, bbox_inches = 'tight')
        #plt.clf()



    def get_image_path_list(self, results):
        images = []
        for item in results:
            images.append((item[0], f'pae_{item[0]}.png'))
        return images


    def sort_dict(self, results: dict, reverse=False):
        return sorted(results.items(),
                      key = lambda x: x[1],
                      reverse=reverse)
    
    def sort_pae_results(self, results: dict):
        return sorted(results.items(),
                      key = lambda x: x[1]['Overall'],
                      reverse=False)


    def find_index_by_protein_name(self, data_list, search_value):
        for index, item in enumerate(data_list):
            if 'protein_names' in item and item['protein_names'] == search_value:
                return index
        return -1

    def get_pae_messages(self, results):
        best_result = float(results[0][1]['Overall'])
        if best_result >= 20.0:
            messages = f"The best overall PAE is higher than 20 A." \
                       " This does not indicate a high confidence." \
                       " The prediction is likely incorrect."
        elif best_result > 10.0 and best_result < 20.0:
            messages = f"The best overall PAE is between 10 and 20 A." \
                       " This indicates an intermediate confidence" \
                       " and prediction might be close to reality."
        elif best_result < 10.0:
            messages = f"The best overall PAE is lower than 10 A." \
                       " This indicates a high confidence" \
                       " and prediction is likely correct."

        return messages


    def get_chain_color(self):
        structure = list(self.cif_dict.values())[0]
        pdb = MMCIFParser().get_structure("structure", structure)[0]
        chains = [str(c.get_id()) for c in pdb.get_chains()]
        return list(zip(chains,
            ["lime","cyan","magenta","yellow","salmon","white","blue","orange"]))
        

class EvaluationPipelineBatch:
    def __init__(self, results_dir: str, pairwise: bool = False):
        self.results_dir = results_dir
        self.pairwise = pairwise
        
    def run(self):
        self.process_scores()
        self.write_html()
        plt.close('all')

    def remove_split(self, x):
        matches = re.findall('(_split\d+)', x)
        for m in matches:
            x = x.replace(m, '')
        return x
    
    def split_by_protein_name(self, item, protein_num, remove_split):
        found_names = []
        protein_name = None
        #logging.debug(f"item: {item}")
        for protein in self.proteins:
            regex_grouped = r'(_(?:bait|prey)_(?:(?!_split\d+).)*)'
            m = re.search(regex_grouped, protein)
            if m:
                id = m.group(1)
                protein = protein.replace(id, '')
            else:
                logging.debug(f"Looking for {protein} in {item}")
            if remove_split:
                pattern = re.compile(r'_split\d+')
                protein_name = re.sub(pattern, '', protein)
            else:
                protein_name = protein
            if re.search(protein_name, item):
                found_names.append(protein_name)
        #print("found names")
        found_names = list(set(found_names))
        #logging.debug(found_names)
        if len(found_names) < 2:
            return None
        else:
            protein_name = None
            if item.startswith(found_names[0]) and protein_num == 0:
                protein_name = found_names[0]
            elif item.startswith(found_names[0]) and protein_num == 1:
                protein_name = found_names[1]
            elif item.startswith(found_names[1]) and protein_num == 0:
                protein_name = found_names[1]
            elif item.startswith(found_names[1]) and protein_num == 1:
                protein_name = found_names[0]
            else:
                logging.error(f"Error in splitting {item}")
                logging.debug(found_names)
            #print(f"Returning protein name: {protein_name}")
            return protein_name

    def generate_summary_plots(self, df, grouped=False):
        logging.debug("Generating summary plots")
        height = 0.2 * len(df['protein_names'])
        if height < 3:
            height = 3

        x_axis_mapping = {'max_multimer_score_value': 'Multimer Score', 'max_iptm_value': 'ipTM', 'max_ptm_value': 'pTM', 'min_pae_value': 'Inter-subunit PAE'}
        if grouped:
            grouped_str = '_merged_'
        else:
            grouped_str = '_'
        for category in ['max_multimer_score_value', 'max_iptm_value', 'max_ptm_value', 'min_pae_value']:
            #Ranking Barplot
            if category == 'min_pae_value':
                df = df.sort_values(category, ascending=False)
                range_ = [0, 35]
            else:
                df = df.sort_values(category)
                range_ = [0,1]

        

            #plt.figure(figsize=(height, height))

            #plt.barh(df['protein_names'], df[category], color='skyblue', edgecolor='black', height=0.5)
            

            #plt.xlabel(x_axis_mapping[category])
            #plt.ylabel('Protein combination')
            #plt.tight_layout()
                
            fig = go.Figure(go.Bar(
                y=df['protein_names'],
                x=df[category],
                orientation='h',
                marker=dict(color='skyblue', line=dict(color='black', width=0.5)),
            ))

            base_height = 300
            label_height = 20
            plot_height = base_height + df['protein_names'].size * label_height

            fig.update_layout(
                xaxis=dict(title=category, range=range_, showticklabels=True, nticks=10, showgrid=True),
                yaxis=dict(title='Protein Names', tickmode='linear', automargin=True),
                height=plot_height,
                margin=dict(
                    b=50,
                    t=50,
                    l=100,
                    r=50,
                )
                #width=800,
                #height=500
            )
            fig.update_yaxes(automargin=True)
            fig.update_xaxes(automargin=True)

            #if category == 'max_multimer_score_value':
            #    plt.axvline(x=0.2, color='red', linestyle='--', label='Value 0.2')
            #    plt.axvline(x=0.3, color='yellow', linestyle='--', label='Value 0.3')
            #    plt.axvline(x=0.5, color='green', linestyle='--', label='Value 0.5')



            key = f"{category.replace('_value', '')}{grouped_str}barplot"
            self.plotly_dict[key] = fig.to_html(full_html=False)
            
            #plt.close()

            #Heatmap
            logging.debug(df)
            if not 'protein_1' in df.columns and not 'protein_2' in df.columns:
                df['protein_1'] = df['protein_names'].apply(lambda x, idx=0, remove_split=grouped: self.split_by_protein_name(x, idx, remove_split))
                df['protein_2'] = df['protein_names'].apply(lambda x, idx=1, remove_split=grouped: self.split_by_protein_name(x, idx, remove_split))
                #pattern = r'(?P<protein_1>\w+(?:_split\d+_)?)(?P<protein_2>\w+(?:_split\d+)?)'
                #df_extracted = df['protein_names'].str.extract(pattern)
                #df = pd.concat([df, df_extracted], axis=1)
            logging.debug(df)

            if self.pairwise:
                df_swapped = df.copy()
                df_swapped['protein_1'], df_swapped['protein_2'] = df['protein_2'], df['protein_1']

                df_combined = pd.concat([df, df_swapped], ignore_index=True)
                logging.debug(category)
                logging.debug(df_combined)
                cmap = sns.diverging_palette(240, 10, as_cmap=True)
                if category == 'min_pae_value':
                    heatmap_data = df_combined.pivot_table(index='protein_1', columns='protein_2', values=category, aggfunc='min')
                    vmin = 0
                    vmax = 35
                else:
                    #Replace negative values with 0
                    df_combined[df_combined[category] < 0] = 0
                    heatmap_data = df_combined.pivot_table(index='protein_1', columns='protein_2', values=category, aggfunc='max')
                    vmin = 0
                    vmax = 1
                logging.debug("heatmap_data")
                logging.debug(heatmap_data)
                plt.figure(figsize=(height, height))
                #ax = sns.heatmap(heatmap_data, mask=np.isnan(heatmap_data), cmap=cmap,  annot=False, fmt=".2f", square=True, vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='black')
                x_labels = heatmap_data.columns.tolist()
                y_labels = heatmap_data.index.tolist()
                fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=x_labels, y=y_labels, colorscale='Viridis'))

                fig.update_layout(
                    xaxis=dict(title='Protein 2', showgrid=True, gridwidth=1, gridcolor='lightgray', tickmode='linear'),
                    yaxis=dict(title='Protein 1', showgrid=True, gridwidth=1, gridcolor='lightgray', tickmode='linear'),
                    height=base_height + df['protein_names'].size * 10,
                    margin=dict(
                        b=50,
                        t=50,
                        l=100,
                        r=50,)
                    )

                key = f"{category.replace('_value', '')}{grouped_str}heatmap"
                self.plotly_dict[key] = fig.to_html(full_html=False)
            #fig.tight_layout()
            #fig.savefig(os.path.join(self.results_dir, fn))
            #fig.close()

    def collect_scores(self):
        score_list = []
        score_files = []
        print(self.results_dir)
        for root, folders, files in os.walk(self.results_dir, followlinks=True):
            print(folders)
            for f in files:
                if f == 'best_scores.json':
                    score_file = os.path.join(root, f)
                    score_files.append(score_file)
            
        for f in score_files:
            with open(f) as h:
                data = json.load(h)
                score_list.append(data)
        return score_list

    def process_scores(self):
        self.scores = scores = self.collect_scores()

        self.proteins = []
        for score_dict in self.scores:
            self.proteins.extend(score_dict['subunits'])

        df = pd.DataFrame(scores)
        value_columns = df.filter(like='value')
        df[value_columns.columns] = df[value_columns.columns].astype(float)
        found_split = False
        self.plotly_dict = {}
        logging.debug(df)
        self.generate_summary_plots(df)
        for score in self.scores:
            if re.search(r'_split\d+', score['protein_names']):
                found_split = True
                break
        if found_split:          
            df['protein_names'] = df['protein_names'].apply(lambda x: self.remove_split(x))
            df_pae = df.copy()
            df_pae = df_pae.groupby(['protein_names'])['min_pae_value'].min().reset_index().sort_values('min_pae_value', ascending=True)
            df_iptm = df.copy()
            df_iptm = df_iptm.groupby(['protein_names'])['max_iptm_value'].max().reset_index().sort_values('max_iptm_value', ascending=False)
            df_ptm = df.copy()
            df_ptm= df_ptm.groupby(['protein_names'])['max_ptm_value'].max().reset_index().sort_values('max_ptm_value', ascending=False)
            df_multimer_score = df.copy()
            df_multimer_score = df_multimer_score.groupby(['protein_names'])['max_multimer_score_value'].max().reset_index().sort_values('max_multimer_score_value', ascending=False)
            self.pae_list_grouped = df_pae[['protein_names', 'min_pae_value']].values.tolist()
            self.iptm_list_grouped = df_iptm[['protein_names', 'max_iptm_value']].values.tolist()
            self.ptm_list_grouped = df_ptm[['protein_names', 'max_ptm_value']].values.tolist()
            self.multimer_score_list_grouped = df_multimer_score[['protein_names', 'max_multimer_score_value']].values.tolist()
            df_merged = pd.merge(df_pae, df_iptm, on='protein_names', how='inner')
            df_merged = pd.merge(df_merged, df_ptm, on='protein_names', how='inner')
            df_merged = pd.merge(df_merged, df_multimer_score, on='protein_names', how='inner')
            df_merged.to_csv(os.path.join(self.results_dir, 'regrouped_pairwise_prediction_results.csv'))
            self.generate_summary_plots(df_merged, grouped=True)
        else:
            for score in self.scores:
                logging.debug(len(score['protein_names'].split('_')))

            self.pae_list_grouped, self.iptm_list_grouped, self.ptm_list_grouped, self.multimer_score_list_grouped = None, None, None, None
        protein_pae_mapping, protein_iptm_mapping, protein_ptm_mapping, protein_multimer_score_mapping = {}, {}, {}, {}
        logging.debug(self.scores)
        for score in self.scores:
            protein_pae_mapping[score['protein_names']] = (score['min_pae_value'], [score['min_pae_model_name']])
            self.protein_pae_mapping = dict(sorted(protein_pae_mapping.items(), key=lambda item: float(item[1][0]), reverse=False))
            protein_iptm_mapping[score['protein_names']] = (score['max_iptm_value'], [score['max_iptm_model_name']])
            self.protein_iptm_mapping = dict(sorted(protein_iptm_mapping.items(), key=lambda item: float(item[1][0]), reverse=True))
            protein_ptm_mapping[score['protein_names']] = (score['max_ptm_value'], [score['max_ptm_model_name']])
            self.protein_ptm_mapping = dict(sorted(protein_ptm_mapping.items(), key=lambda item: float(item[1][0]), reverse=True))
            protein_multimer_score_mapping[score['protein_names']] = (score['max_multimer_score_value'], [score['max_multimer_score_model_name']])
            self.protein_multimer_score_mapping = dict(sorted(protein_multimer_score_mapping.items(), key=lambda item: float(item[1][0]), reverse=True))

    def write_html(self):
        templates_path = pkg_resources.resource_filename("guifold", "templates")
        logging.debug(templates_path)
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('batch_summary.html')
        logging.debug("Writing to summary.html")
        logging.debug(self.scores)
        logging.debug(f"pairwise: {self.pairwise}")

        self.protein_pae_mapping = {k.replace('-', '_'): v for k,v in self.protein_pae_mapping.items()}
        self.protein_iptm_mapping = {k.replace('-', '_'): v for k,v in self.protein_iptm_mapping.items()}
        self.protein_ptm_mapping = {k.replace('-', '_'): v for k,v in self.protein_ptm_mapping.items()}
        self.protein_multimer_score_mapping = {k.replace('-', '_'): v for k,v in self.protein_multimer_score_mapping.items()}

        rendered = template.render(protein_pae_mapping=self.protein_pae_mapping,
                                   protein_iptm_mapping=self.protein_iptm_mapping,
                                   protein_ptm_mapping=self.protein_ptm_mapping,
                                   protein_multimer_score_mapping=self.protein_multimer_score_mapping,
                                   pae_list_grouped=self.pae_list_grouped,
                                   iptm_list_grouped=self.iptm_list_grouped,
                                   ptm_list_grouped=self.ptm_list_grouped,
                                   multimer_score_list_grouped=self.multimer_score_list_grouped,
                                   plotly=self.plotly_dict,
                                   pairwise=self.pairwise)
        
        html_path = os.path.join(self.results_dir, "batch_summary.html")
        with open(html_path, "w") as fout:
            fout.write(rendered)
        logging.info(f"Summary HTML written to {html_path}")

def get_score_dict(csv_file):
    scores = []
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip('\n') for l in lines]
    if len(lines) > 1:
        keys = lines[0].split(',')
    for line in lines[1:]:
        values = line.split(',')
        values = [None if value == 'None' else value for value in values]
        dict_ = {key: value for (key, value) in zip(keys, values)}
        if not dict_['protein_names'] is None:
            scores.append(dict_)
    return scores

def run_evaluation(results_dir, args):
    try:
        if not os.path.exists(os.path.join(results_dir, "best_scores.json")) or args.overwrite:
            evaluation = EvaluationPipeline(
                    results_dir=results_dir,
                    continue_from_existing_results=args.continue_from_existing_results,
                    custom_spacing=args.custom_spacing,
                    custom_start_residue_list=args.custom_start_residue_list,
                    custom_axis_label_list=args.custom_axis_label_list,
                    name_id_pattern=args.name_id_pattern)
            evaluation.run_pipeline()
    except Exception as e:
        logging.error(f"Exception in process: {e}")
        traceback.print_exc()

def main():
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    jax.config.update('jax_platform_name', 'cpu')
    

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--results_dir', default=None, help='Path to results (predictions).')
    parser.add_argument('--run_batch_eval', action='store_true', default=None, help='Evaluation of multiple predictions')
    parser.add_argument('--debug', default=False, action='store_true', help='(Optional) Debug mode.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--continue_from_existing_results', default=False, action='store_true',
                        help='(Optional) Continue from previously extracted PAE values.')
    parser.add_argument('--custom_spacing', default=None, help='(Optional) Custom value for spacing of major tick labels.')
    parser.add_argument('--custom_start_residue_list', default=None, help='(Optional) Define custom axes minimum number. If the prediction was done with internal segments of the proteins,'
                                                            'the actual residue numbering can be restored in the plot by'
                                                            'giving the starting residue numbers of the segments separated by a comma.'
                                                            'For example, if there are two subunits with predicted segments'
                                                            '15-100 and 50-300, the list would be --custom_start_residue_list 15,50')
    parser.add_argument('--custom_axis_label_list', default=None, help='(Optional) Define custom axes labels.'
                                                          ' Labels for different subunits need to be separated by a comma'
                                                          ' and given in the same order as the sequences in the fasta file. Example: --custom_axis_label_list \"Protein A\", \"Protein B\"')
    parser.add_argument('--num_cpu', type=int, default=1, help='Number of cpus for batch evaluation')
    parser.add_argument('--name_id_pattern', action='store_true', help='Split pairwise protein name based on name_id pattern.')
    parser.add_argument('--chain_id_name_mapping_path', type=str, default=None, help='Path to chain_id to protein name mapping json file')
    args, unknown = parser.parse_known_args()
    if not args.debug:
        logging.set_verbosity(logging.INFO)
    else:
        logging.set_verbosity(logging.DEBUG)

    if args.run_batch_eval:
        score_list = []
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        results_queue = manager.Queue()
        lock = multiprocessing.Lock()
        for results_dir in [os.path.join(args.results_dir, f) for f in os.listdir(args.results_dir) if os.path.isdir(os.path.join(args.results_dir, f))]:
            if args.pairwise:
                # In pairwise case script expects that protein names are separated by '_'
                if not '_' in os.path.basename(results_dir):
                    continue
            task_queue.put((results_dir, args))
        while task_queue.qsize() > 0:
            active_processes = []
            for _ in range(args.num_cpu):
                if task_queue.qsize() > 0:
                    task_args = task_queue.get()
                    p = multiprocessing.Process(target=run_evaluation, args=(task_args[0],task_args[1]))
                    active_processes.append(p)
                    p.start()
            
            for p in active_processes:
                    p.join()
        EvaluationPipelineBatch(results_dir=args.results_dir, pairwise=args.pairwise).run()
    else:
        EvaluationPipeline(results_dir=args.results_dir,
                            continue_from_existing_results=args.continue_from_existing_results,
                            custom_spacing=args.custom_spacing,
                            custom_start_residue_list=args.custom_start_residue_list,
                            custom_axis_label_list=args.custom_axis_label_list,
                            name_id_pattern=args.name_id_pattern,
                            chain_id_name_mapping_file=args.chain_id_name_mapping_path).run_pipeline()

if __name__ == '__main__':
    main()
    

