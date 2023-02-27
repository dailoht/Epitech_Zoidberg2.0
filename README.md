# Zoidberg2.0 : Computer aided diagnosis

The aim of this project is to use machine learning to help doctors detecting pneumonia. We, thus, have to classify x-ray images into 3 classes : *normal*, *viral pneumonia*, *bacterial pneumonia*.  
The project has 2 parts :
- Analyze and model building in python. All results can be shown in notebooks
- Model deployment in rust (thanks to Mathys)

# Visualization with Notebook

To be able to run notebooks, you are going to need python 3.9 and [Anaconda](https://www.anaconda.com/) or Miniconda which is a minimal version of Anaconda. Below, we provide instruction for Miniconda.

## Setup the git repository


### Install Miniconda

Please go to the [Anaconda website](https://docs.conda.io/en/latest/miniconda.html). Download and install the latest Miniconda version for Python 3.9 for your operating system.  

Once it is done, you can check conda with :
```bash
conda -V
```


### Check-out the git repository

Once Miniconda is ready, clone the repository :

```bash
git clone git@github.com:EpitechMscProPromo2024/T-DEV-810-PAR_10.git
```


## Makefile commands

The next steps can be done with the Makefile. We provide here all commands that can be used : 
```
help                Return the list of all make commands
clean               Delete all compiled files, models and data 
clean-data          Delete all data 
clean-model         Delete all models 
clean-pyc           Delete all compiled Python files 
create_environment  Set up python interpreter environment 
jupyter             Start Jupyter Notebook local server 
lint                Lint using flake8 
requirements        Install Python Dependencies 
requirements_file    Update requirements file 
test_environment    Test python environment is setup correctly 
```

## Create isolated virtual environment

Change directory (`cd`) into the project folder, then type:

```bash
make create_environment
conda activate zoidberg_env
```

If you want to see this virtual environment in the kernels registered by jupyter then use the command (be sure that the environment is actived):
```bash
ipython kernel install --user --name=zoidberg_env
```

## Start Jupyter Notebook or JupyterLab

Finally, start from terminal :
```bash
make jupyter
```

Go on your browser to work on those jupyters notebooks

## Notebook default settings

# Report 

The document in the report folder is written in Latex, we use [Overleaf](https://www.overleaf.com/). Please ask us if you want access to is tex version.  
Access : [https://www.overleaf.com/project/63fcbca4e88c8a6d5976e9c6](https://www.overleaf.com/project/63fcbca4e88c8a6d5976e9c6)

# Deployment in rust