# GUIFold
GUI for running jobs with a local installation of AlphaFold3. Supports submission to queuing systems.

Important note: This is an ALPHA version and development is ongoing. Feedback and bug reports are very welcome.

Main Features:
* Organization of jobs into projects
* Tracking and restoring jobs
* Queue submission
* Splitting into feature generation (CPU) and prediction (GPU) steps
* Evaluation pipeline (PAE/pLDDT tables, PAE plots)

## Installation

### Installation of the repository and dependencies

The installation requires conda.

```
mkdir guifold
cd guifold
git clone https://github.com/fmi-basel/GUIFold3
```

Before proceeding with the installation it is recommended to setup the [global configuration file](#setup-of-global-configuration-file) and [cluster submission script](#setup-of-cluster-submission-template) (if needed).

If you have a separate initialization script for conda (if the initialization is not in your .bashrc) add
```
source path/to/conda_init.sh
```
to the beginning of `GUIFold/install.sh`.

Running the install.sh file will<br/>
* create a conda environment in the same folder<br/>
* install GUIFold and dependencies<br/>
* install alphafold3 and dependencies<br/><br/>

To run the setup:
```
bash GUIFold3/install.sh
```
`conda_env` will be the default name of the conda repository. The conda env will be installed with an absolute path (which is also needed for activation).

Test if the GUI opens (see Troubleshooting):
```
conda activate /path/to/conda_env
afgui
```



### (OPTIONAL) Setup of global configuration file

When GUIFold is installed in a shared location it is recommended 
to create a global configuration file so that the users don't have to configure the paths on their own.
When a user starts the app for the first time and a configuration file exists in the expected location, the parameters will be automatically
transferred to the database of the user (stored in the home directory of the user). The user can change settings in the GUI later on.
Open the file `GUIFold/guifold/config/template.conf` and adapt it to your local environment.
Further explanations of the different parameters are given as comments in the file.
After editing, save the file to `GUIFold/guifold/config/guifold.conf`. It is important to use this specific name and location otherwise it will not be loaded.
When the global configuration needs to be changed later on, the users can re-load it in the Settings dialog of the GUI.


### (OPTIONAL) Setup of a local configuration file

As an alternative to a global configuration file stored inside the GUIFold installation directory, a "local" configuration file can be created from the template `GUIFold/guifold/config/template.conf` and stored in any location. To load the settings from this file go to the `Settings` dialog in the GUI and click the `Load local config` button (bottom right). 

### Manual configuration of settings inside the GUI

If no configuration file is supplied, settings need to be defined in the `Settings` dialog and will be stored in the user's database. This has to be done by each user separately.

### (OPTIONAL) Configuration of cluster submission

**This step is only required if GUIFold is inteded to be used in combination with queuing systems (on a cluster).**

#### Settings

Queuing system specific settings need to be defined in the configuration file or in the `Settings`dialog of the GUI.

The following table shows configurations for common queueing systems:

Setting      | SLURM        | UGE/SGE
-----------: | -----------: | ----------:
Submit cmd   | sbatch       | qsub
Cancel cmd   | scancel      | qdel
JobID regex  | \D*(\d+)\D   | ?

Other settings:

`Min num CPUs` The minimum number of CPUs to request. This number of CPUs will be used for feature pipelines based on jackhmmer and hhblits. In our case a number of 20 CPUs works well.

`Max num CPUs` The maximum number of CPUs to request. This number is only used by feature pipelines based on mmseqs (currently only colabfold_local). In our case a number of 50 CPUs works well.

`Max RAM (GB)` The maximum RAM to request. This is only needed for certain tasks such as unified memory prediction or local mmseqs depending on the database (split) size (see setup of colabfold databases). We use 600 GB.

`Min RAM (GB)` The minimum RAM to request. This will be requested by default and should be high enough to avoid out of memory errors during the MSA step. In a few cases MSAs can get very large and require up to 120 GB.

`Max GPU memory (GB)` The maximum available GPU memory (on a single GPU model). 

`Split job` Whether to split the job into feature (CPU-only) and prediction steps (requires specific configuration of the submission script template, see below)

`Queue account` (Optional) Only needs to be set when an account is needed to submit jobs on a cluster. Each user has to define it individually.

`Use queue by default` Activates the `Submit to Queue` checkbox by default.

#### Preparation of a submission script template

The Jinja2 package is used to render the submission script template. See [Jinja2 documentation](https://jinja.palletsprojects.com/en/3.0.x/) for further information. The variables listed below can be used to create a template. An example file for a slurm cluster can be found in `guifold/templates/submission_script_slurm.j2`.

The template file can be saved to any location. The path to the template needs to be set in the configuration file (see above) or in the `Settings` dialog of the GUI


#### Automatically splitting a job into CPU and GPU parts

If the queueing system supports dependencies (i.e. a job waits in the queue until another job has finished), the "split job feature" can be activated in the GUI settings if needed. Since the feature generation step does not require GPU, this step can be run on CPU-only resources. Two jobs will be submitted, the first job will request CPU (`use_gpu=False`) and the second job (`use_gpu=True`) will wait for the first job to finish (if the dependency is configured). An example how to add a dependency for SLURM and how to create a conditional to request CPU or GPU resources is provided below. Alternatively, the job can be manually devided into CPU and GPU steps by choosing `Only Features` in the GUI and, after this job has finished, re-starting the job with `Only Features` deactivated. 

#### Variables that can be used in the submission script template

GUIFold supports the following variables that can be used in the submission template. The parameters are determined based on the input and settings (configuration file or Settings dialog in the app):<br/><br/>
`{{logfile}}` (required) Path to the log file. Generated by the App.<br/>
`{{account}}` (optional) When a specific account is needed to run jobs on the cluster. As defined in settings.<br/>
`{{use_gpu}}` (optional) This can be used to build a conditional (example below) to select CPU or GPU nodes/queues<br/>
`{{mem}}` (optional) How much RAM (in GB) should be reserved. The RAM will be automatically increased with the GPU memory for unified memory. Depends on min/max RAM defined in settings and on the task.<br/>
`{{num_cpu}}` (optional) Number of CPUs to request (depends on the task and min/max CPUs defined in Settings)
`{{num_gpu}}` (optional) Number of GPUs to request (depends on the task and number of GPUs set in Advanced job settings)
`{{gpu_mem}}` (optional) Useful when the queuing system supports selection of GPU by memory requirement. Calculated by the App (in GB unit) based on sequence length.<br/>
`{{split_mem}}` (optional) If the required memory exceeds the available GPU memory, the job can be run with unified memory. The split_mem variable holds None or the memory split fraction and can be used for a conditional to set the FLAGS required to enable unified memory use (see SLURM example below).<br/>
`{{add_dependency}}` (required) When the job is started with "split job setting", this variable will be True for the second job (prediction step) and allows adding a dependency on the first job (feature step). <br/>
`{{commnad}}` (required) The command to run the AlphaFold job. Generated by the App.

To cancel jobs from the GUI, the script also needs to write the Job ID to the logfile.
The pattern needs to be as follows:<br/>
```echo "QUEUE_JOB_ID=$JOB_ID_VARIABLE_FROM_QUEUING_SYSTEM"```<br/>
In case of SLURM it would be:<br/>
```echo "QUEUE_JOB_ID=$SLURM_JOB_ID"```



## Usage

When the conda env is activated (conda activate /path/to/af-conda) or added to PATH, LD_LIBRARY_PATH and PYTHONPATH (see [Setup of a module file](#setup-of-a-module-file) you can start GUIFold by typing:<br/>
`afgui`

To re-run an evaluation go to the job folder (where the FASTA sequence is stored) and type<br/>
`af3eval --results_path /path/to/predictions`


## Licenses

GUIFold is licensed under the Apache License, Version 2.0.

Icons are from the GTK framework, licensed under [GPL](https://gitlab.gnome.org/GNOME/gtk/-/blob/main/COPYING).

Third-party software and libraries may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.