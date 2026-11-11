# af3_batch.py — Batch Driver for AlphaFold 3

`af3_batch.py` runs AlphaFold 3 (AF3) on many monomers or binary protein complexes at once. It reads proteins (or protein pairs) from a CSV file, or a list of ready-made AF3 jobs from a JSON file. It writes the AF3 input JSONs and runs `run_alphafold.py` on one or more GPUs, with a configurable number of jobs per GPU.

> **Note:** By default the script writes the `precomputedMsaPath` field, which is only supported by the **local AF3 fork**. To use stock AlphaFold 3, pass `--msa_input inline`.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Run Modes](#run-modes)
4. [Input Formats](#input-formats)
5. [Job and Folder Naming](#job-and-folder-naming)
6. [MSA and Template Handling](#msa-and-template-handling)
7. [Skipping, Overwriting and Cleaning Up](#skipping-overwriting-and-cleaning-up)
8. [GPU Scheduling](#gpu-scheduling)
9. [Command-Line Reference](#command-line-reference)
10. [Output Layout and Logging](#output-layout-and-logging)
11. [Exit Codes](#exit-codes)
12. [Examples](#examples)
13. [Troubleshooting](#troubleshooting)
14. [Changes from the Original Script](#changes-from-the-original-script)

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.8 | The driver itself; standard library plus `pandas`. |
| `pandas` | Needed to read the CSV. |
| AlphaFold 3 | `run_alphafold.py` must exist at `--run_alphafold`. The default `--msa_input path` needs the local fork that supports `precomputedMsaPath`. |
| AF3 Python environment | Set with `--python` if AF3 runs in a different interpreter than the driver. |
| `nvidia-smi` | Optional; only used by `--wait_for_idle_gpu`. |

The default paths point to the shared installation:

```text
run_alphafold.py : /tungstenfs/scratch/gstructb_share/modules/software/alphafold/alphafold3/alphafold3/run_alphafold.py
JAX cache        : /tachyon/temp/gstructb/
```

---

## Quick Start

```bash
# 1) Compute MSAs/templates once per unique sequence
python af3_batch.py --csv pairs.csv --features \
    --name_column1 gene_a --seq_column1 seq_a \
    --name_column2 gene_b --seq_column2 seq_b

# 2) Predict all pairs on 4 GPUs re-using those features
python af3_batch.py --csv pairs.csv --use_precomputed_features --num_gpus 4 \
    --name_column1 gene_a --seq_column1 seq_a \
    --name_column2 gene_b --seq_column2 seq_b
```

Steps 1 and 2 can be run in a single call by passing both `--features` and `--use_precomputed_features`.

Show all options:

```bash
python af3_batch.py --help
```

---

## Run Modes

Modes can be combined. They always run in this order:

| Order | Mode | Flags | Input | What it does |
|---|---|---|---|---|
| 1 | Features | `--features` | CSV | Data pipeline only (MSA + template search), once per unique sequence. |
| 2a | Precomputed prediction | `--use_precomputed_features` | CSV + feature folders | One prediction per CSV row, re-using features from step 1. |
| 2b | Single-sequence prediction | `--no_msa --no_templates` | CSV | One prediction per CSV row with no MSA and no templates. |
| 3 | Batch JSON | `--batch_json FILE` | JSON | Runs a list of ready-made AF3 job dictionaries. |

At least one mode must be selected, otherwise the script stops with an error.

### 1. Features (`--features`)

- Collects every sequence in `--seq_column1` (and `--seq_column2` if given).
- Each **unique** sequence is computed **once**, under the name of the first row it appears in.
- Writes a single-chain job JSON and runs AF3 with `--run_data_pipeline --norun_inference`.
- Result: `<output_dir>/<name>/<name>_data.json`.
- Sequences that already have a `_data.json` are skipped, unless `--overwrite` is given.
- Runs `--feature_workers` jobs at once (default 1). Feature jobs are CPU-bound and do **not** set `CUDA_VISIBLE_DEVICES`.

### 2a. Precomputed Features (`--use_precomputed_features`)

- For each CSV row, loads `<precomputed_features_dir>/<name>/<name>_data.json` for every chain.
- Checks that the sequence in the feature file matches the CSV. Rows that don't match are logged as errors and skipped.
- If a sequence's features were computed under a different name (a duplicate sequence), that folder is used instead.
- How MSAs and templates are passed is set by `--msa_input`, `--msa_path_format`, `--no_msa` and `--no_templates` (see [MSA and Template Handling](#msa-and-template-handling)).

### 2b. Single-Sequence (`--no_msa --no_templates`)

- Used when **both** flags are set **without** `--use_precomputed_features`.
- Every chain gets `"unpairedMsa": ""`, `"pairedMsa": ""`, `"templates": []`.
- No feature files are needed.

### 3. Batch JSON (`--batch_json`)

- The file must hold a list of AF3 job dicts (a single dict also works).
- Each job must have a `"name"`; jobs without one are skipped with an error.
- Each job is written to `<json_dir>/<sanitised name>.json` and run on the GPU slots.

---

## Input Formats

### CSV

Any CSV that `pandas.read_csv` can read. Every value is read as a string. Whitespace inside sequences is removed.

**Monomers**

```csv
name,sequence
P53,MEEPQSDPSVEPPLSQETFSDLWKLL...
MDM2,MCNTNMSVPTDGAVTTSQIPASEQETL...
```

```bash
--name_column1 name --seq_column1 sequence
```

**Pairs, with optional IDs**

```csv
gene_a,uniprot_a,seq_a,gene_b,uniprot_b,seq_b
TP53,P04637,MEEPQ...,MDM2,Q00987,MCNTN...
```

```bash
--name_column1 gene_a --id_column1 uniprot_a --seq_column1 seq_a \
--name_column2 gene_b --id_column2 uniprot_b --seq_column2 seq_b
```

Rules:

- `--name_column1` and `--seq_column1` are required in every CSV mode.
- `--name_column2` and `--seq_column2` must be given together.
- ID columns are optional. An empty or NaN ID is ignored for that row.
- If a requested column is missing, the script stops and lists the available columns.
- Rows with a missing sequence are logged and skipped.

### Batch JSON

```json
[
  {
    "name": "complex_1",
    "modelSeeds": [42],
    "sequences": [
      {"protein": {"id": "A", "sequence": "MKT..."}},
      {"protein": {"id": "B", "sequence": "MSE..."}}
    ],
    "dialect": "alphafold3",
    "version": 1
  }
]
```

Jobs are passed to AF3 as they are. Any field your AF3 fork accepts (ligands, `precomputedMsaPath`, etc.) can be used.

---

## Job and Folder Naming

| Object | Rule | Example |
|---|---|---|
| Protein name | `<name>`, or `<name>-<id>` if an ID column is given and not empty | `TP53-P04637` |
| Pair job name | `<protein1>_<protein2>` | `TP53-P04637_MDM2-Q00987` |
| Folder / file name | AF3 `sanitised_name()`: lowercase, spaces → `_`, keep only `[a-z0-9_.-]` | `tp53-p04637_mdm2-q00987` |

- Output folder: `<output_dir>/<sanitised name>/`
- Input JSON: `<json_dir>/<sanitised name>.json`
- Feature file: `<output_dir>/<sanitised name>/<sanitised name>_data.json`
- If two rows produce the same sanitised job name, only the first is run; the others are logged as warnings.

---

## MSA and Template Handling

This section applies to `--use_precomputed_features`.

### `--msa_input`

| Value | Fields written per chain | Use with |
|---|---|---|
| `path` (default) | `precomputedMsaPath` + `templates` | Local AF3 fork |
| `inline` | `unpairedMsa` + `pairedMsa` copied from `_data.json`, + `templates` | Stock AlphaFold 3 |

### `--msa_path_format` (only with `--msa_input path`)

| Value | Value of `precomputedMsaPath` | Example |
|---|---|---|
| `name` (default) | Sanitised feature folder name, as in the original script | `tp53-p04637` |
| `relative` | `<precomputed_features_dir>/<name>` | `features/tp53-p04637` |
| `absolute` | Absolute path of that folder | `/scratch/me/features/tp53-p04637` |

If you use `name` while `--precomputed_features_dir` is not the working directory, the script prints a warning. Check how your fork resolves the path, or switch to `relative` or `absolute`.

### Turning off MSAs or templates

| Flags | MSA fields | Templates |
|---|---|---|
| (none) | per `--msa_input` | from `_data.json` |
| `--no_msa` | `unpairedMsa: ""`, `pairedMsa: ""` (no `precomputedMsaPath`) | from `_data.json` |
| `--no_templates` | per `--msa_input` | `[]` |
| `--no_msa --no_templates` | empty | `[]` |

`--no_msa` or `--no_templates` on its own requires `--use_precomputed_features`.

### Example generated chain (default settings)

```json
{"protein": {"id": "A", "sequence": "MKT...", "precomputedMsaPath": "genea-1", "templates": [...]}}
```

---

## Skipping, Overwriting and Cleaning Up

A job counts as **finished** if its output folder contains a file whose name includes `ranking_scores`.

| Folder state | Default | `--overwrite` | `--remove_unfinished` |
|---|---|---|---|
| Does not exist | run | run | run |
| Finished | **skip** | re-run | skip |
| Unfinished | run (folder kept) | run | delete folder, then run |
| Unfinished but contains `<name>_data.json` | run | run | **not deleted** (warning), then run |

- Folders holding feature files are never deleted, so features needed by later rows are safe.
- In features mode, `--overwrite` recomputes features that already exist.
- AF3 is always called with `--force_output_dir`.

---

## GPU Scheduling

- GPUs: `--gpu_ids 0 2 3`, or `--num_gpus N` (meaning IDs `0..N-1`). `--gpu_ids` takes priority.
- Slots: every GPU gets `--jobs_per_gpu` slots, handed out round-robin so work spreads across GPUs.
- Each job runs as its own `subprocess.Popen` with `CUDA_VISIBLE_DEVICES=<gpu>`. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is also set, unless it's already set in your environment, so GPU numbers match `nvidia-smi`.
- The scheduler checks job status every `--poll_interval` seconds (default 10) and starts the next queued job as soon as a slot frees up.
- `--wait_for_idle_gpu`: before starting a job on a GPU where none of this script's jobs are running, the script asks `nvidia-smi` for other compute processes and waits while there are any.
- **Ctrl-C** stops all running AF3 jobs (and kills them after 30 s if needed), closes their logs and exits.

---

## Command-Line Reference

### Run modes

| Option | Default | Description |
|---|---|---|
| `--features` | off | Data pipeline only, for every unique sequence in `--csv`. |
| `--use_precomputed_features` | off | Predict CSV rows re-using `<name>/<name>_data.json`. |
| `--no_msa` | off | Do not use MSAs. |
| `--no_templates` | off | Do not use templates. |
| `--batch_json FILE` | – | JSON list of AF3 job dicts. |
| `--prediction` | off | Deprecated, has no effect. |

### CSV input

| Option | Default | Description |
|---|---|---|
| `--csv FILE` | – | Input CSV (required for CSV modes). |
| `--name_column1` | – | Name column of protein 1 (required with `--csv`). |
| `--seq_column1` | – | Sequence column of protein 1 (required with `--csv`). |
| `--id_column1` | – | Optional ID column appended as `<name>-<id>`. |
| `--name_column2` | – | Name column of protein 2 (pair mode). |
| `--seq_column2` | – | Sequence column of protein 2 (pair mode). |
| `--id_column2` | – | Optional ID column for protein 2. |
| `--precomputed_features_dir` | `--output_dir` | Folder containing the feature folders. |

### Execution

| Option | Default | Description |
|---|---|---|
| `--num_gpus N` | 1 | Use GPUs `0..N-1`. |
| `--gpu_ids ID [ID ...]` | – | Explicit GPU IDs; overrides `--num_gpus`. |
| `--jobs_per_gpu N` | 1 | Concurrent prediction jobs per GPU. |
| `--feature_workers N` | 1 | Concurrent data-pipeline jobs (keep this low; each search is already multi-threaded). |
| `--wait_for_idle_gpu` | off | Wait for other users' processes to leave a GPU. |
| `--poll_interval SEC` | 10 | Seconds between status checks. |
| `--overwrite` | off | Re-run finished jobs / recompute existing features. |
| `--remove_unfinished` | off | Delete unfinished output folders (never ones holding features). |

### AlphaFold 3 options

| Option | Default | Description |
|---|---|---|
| `--run_alphafold PATH` | shared install path | Path to `run_alphafold.py`. |
| `--python EXE` | `python3` | Interpreter of the AF3 environment. |
| `--flash_attention` | `xla` | `xla`, `triton` or `cudnn`. |
| `--jax_cache_dir DIR` | `/tachyon/temp/gstructb/` | JAX compilation cache (`''` disables it). |
| `--buckets LIST` | `64,128,...,5120` | Token buckets passed to AF3. |
| `--seeds N [N ...]` | `42` | Model seeds for generated jobs. |
| `--output_dir DIR` | `.` | AF3 output folder. |
| `--json_dir DIR` | `.` | Where generated input JSONs are written. |
| `--msa_input` | `path` | `path` (`precomputedMsaPath`, local fork) or `inline` (stock AF3). |
| `--msa_path_format` | `name` | `name`, `relative` or `absolute`. |

### Logging

| Option | Default | Description |
|---|---|---|
| `--log_file FILE` | `debug.log` | Main log file. |
| `--log_dir DIR` | `af3_logs` | Per-job AF3 stdout/stderr logs. |
| `-v`, `--verbose` | off | Debug logging (includes full AF3 command lines). |

### Validation rules

The script stops with an error if:

- no run mode is selected;
- a CSV mode is used without `--csv`, `--name_column1` or `--seq_column1`;
- only one of `--name_column2` / `--seq_column2` is given;
- `--no_msa` or `--no_templates` is used alone without `--use_precomputed_features`;
- `--num_gpus`, `--jobs_per_gpu` or `--feature_workers` is below 1;
- `--run_alphafold` or `--batch_json` does not exist.

---

## Output Layout and Logging

```text
<json_dir>/
  tp53-p04637.json
  tp53-p04637_mdm2-q00987.json
<output_dir>/
  tp53-p04637/
    tp53-p04637_data.json          # features (--features)
  tp53-p04637_mdm2-q00987/
    ranking_scores.csv             # marks the job as finished
    ...                            # AF3 models, confidences
af3_logs/
  tp53-p04637_features.log         # data-pipeline log
  tp53-p04637_mdm2-q00987.log      # prediction log
debug.log                          # main driver log
```

- AF3 stdout and stderr are written straight to the job's log file while it runs.
- The main log records job starts and finishes, run times, failures (with the log path), skipped jobs and a final summary.

The AF3 command built for each job:

```bash
<python> <run_alphafold> --json_path=<json> --output_dir=<output_dir> --force_output_dir \
    --flash_attention_implementation=<impl> --buckets=<buckets> \
    [--jax_compilation_cache_dir=<dir>] [--run_data_pipeline --norun_inference]
```

---

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | All jobs succeeded (or there was nothing to do). |
| 1 | At least one AF3 job failed, failed to start, or a CSV row was skipped because of an error. |
| 2 | Invalid command-line arguments (argparse). |

---

## Examples

**Features and predictions in one call, 2 GPUs, 2 jobs per GPU**

```bash
python af3_batch.py --csv pairs.csv --features --use_precomputed_features \
    --num_gpus 2 --jobs_per_gpu 2 \
    --name_column1 gene_a --seq_column1 seq_a \
    --name_column2 gene_b --seq_column2 seq_b
```

**Features in a separate folder, absolute MSA paths**

```bash
python af3_batch.py --csv pairs.csv --use_precomputed_features \
    --precomputed_features_dir /scratch/me/features --msa_path_format absolute \
    --output_dir predictions --json_dir inputs \
    --name_column1 gene_a --seq_column1 seq_a \
    --name_column2 gene_b --seq_column2 seq_b
```

**Stock AlphaFold 3 (inline MSAs)**

```bash
python af3_batch.py --csv pairs.csv --use_precomputed_features --msa_input inline ...
```

**Templates only (no MSA)**

```bash
python af3_batch.py --csv pairs.csv --use_precomputed_features --no_msa ...
```

**Single-sequence monomers on specific GPUs**

```bash
python af3_batch.py --csv mono.csv --no_msa --no_templates --gpu_ids 1 3 \
    --name_column1 name --seq_column1 sequence
```

**Batch JSON, cleaning up unfinished runs, multiple seeds**

```bash
python af3_batch.py --batch_json jobs.json --num_gpus 4 --remove_unfinished
```

(`--seeds` only affects jobs the script generates from a CSV; batch jobs keep their own `modelSeeds`.)

**Shared GPU node**

```bash
python af3_batch.py --batch_json jobs.json --num_gpus 4 --wait_for_idle_gpu --poll_interval 30
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No input features found for X` | Run `--features` first, or point `--precomputed_features_dir` at the right folder. Folder names are sanitised (lowercase etc.). |
| `Sequence mismatch between CSV and ...` | The CSV sequence changed since the features were computed. Recompute with `--features --overwrite`. |
| AF3 cannot find the precomputed MSA | Your fork resolves `precomputedMsaPath` differently. Try `--msa_path_format relative` or `absolute`. |
| `Unknown field precomputedMsaPath` | You are running stock AF3. Use `--msa_input inline` or point to the fork. |
| Job marked `FAILED` | Check `af3_logs/<job>.log`. |
| Job always skipped | It's already finished (`ranking_scores*` exists). Use `--overwrite`. |
| Wrong GPU used | Check `--gpu_ids`. Numbering follows PCI bus order (as in `nvidia-smi`). |
| Out of GPU memory | Lower `--jobs_per_gpu`. |
| `--prediction` does nothing | Deprecated; choose a mode flag instead. |

---

## Changes from the Original Script

- Scheduler no longer skips finished jobs (no list changes while looping over it); `multiprocessing` replaced with `subprocess.Popen`.
- `--remove_unfinished` with `--batch_json` no longer deletes **finished** runs.
- Unfinished folders are only deleted with `--remove_unfinished`, never when they contain feature files.
- AF3 exit codes are checked; failures are logged and the script exits with code 1.
- No `shell=True`; per-job logs written while jobs run.
- AF3 `sanitised_name()` is used for all folder and file names.
- Duplicate sequences under different names share one feature folder.
- Each mode runs exactly once, in a fixed order.
- `precomputedMsaPath` (local fork) is the default, with `--msa_input inline` for stock AF3.
- Hardcoded paths are now options; new options `--gpu_ids`, `--feature_workers`, `--wait_for_idle_gpu`, `--poll_interval`, `--seeds`, `--json_dir`, `--log_dir`, `--msa_input`, `--msa_path_format`, `-v`.
- Pair names include each protein's ID whenever that ID column is given.
 