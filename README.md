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

Experiment_1 contains the files required to replicate the results and figures from Experiment 1 in the paper. The bash script is designed to be run in a SLURM HPC environment. The python script can be run on its own,
but PMCMC is quite computationally intensive. 

Experiment_real_data_2022 and Experiment_real_data_2023 are folders containing the files to replicate the results on the real datasets from Arizona. The only difference is the data file in the directory. 



