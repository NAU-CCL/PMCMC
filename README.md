Estimating Dynamic Transmission Rates with Black-Karasinski Process in Stochastic SIHR models using particle MCMC

This codebase can be used to replicate our results from the paper. 

### <center> Setup </center> 

It is recomennded to use a python virtual environment to setup the python environment. This can be done easily in vscode or from the command line.

#### Command line

pip install virtualenv (if you don't already have virtualenv installed) <br>
virtualenv venv to create your new environment (called 'venv' here) <br>
source venv/bin/activate to enter the virtual environment <br>
pip install -r requirements.txt to install the requirements in the current environment

#### VSCode
Press ctrl-shift-p, cmd-shift-p on mac. Choose Python:Create Environment, then follow the steps. 

#### <center> Folder Structure </center>

The examples subdirectory features two files designed to serve as a simple example of the code. The first is to generate a synthetic dataset, the second performs PMCMC on the data. 

##### Note 
The .sh files can be run using sbatch job_file.sh (job_file is a placeholder for the filename) The bash scripts are designed to be run in a SLURM HPC environment. The python scripts can be run on their own,
but PMCMC is quite computationally intensive. 

Experiment_1 contains the files required to replicate the results and figures from Experiment 1 in the paper. To replicate the results run exp_1_job.sh and once finished, viz.ipynb to create the plots. 

Experiment_real_data_2022 and Experiment_real_data_2023 within Experiment_4 are folders containing the files to replicate the results on the real datasets from Arizona. Run real_data_job.sh to replicate the results.
Once the jobs are completed, the trace plots can be generated using the .sh file job_plot.sh



