# Zoidberg2.0 : Computer aided diagnosis

The aim of this project is to use machine learning to help doctors detecting pneumonia. We, thus, have to classify x-ray images into 3 classes : *normal*, *viral pneumonia*, *bacterial pneumonia*.  
The project has 2 parts :
- Analyze and model building in python. All results can be shown in notebooks
- Model deployment

# 

# Visualization with Notebook

Notebooks can be run either with [Google Colab](https://colab.research.google.com/) or in a local environment. Steps below are provided to run them locally. If you want to use Colab, you only need to clone this repo on Colab and upload needed data [Section 3.1](#31-downloading-data).


You are going to need python 3.9 and [Anaconda](https://www.anaconda.com/) or Miniconda which is a minimal version of Anaconda. Below, we provide instruction for Miniconda.

## 1. Setup the git repository


### 1.1 Install Miniconda

Please go to the [Anaconda website](https://docs.conda.io/en/latest/miniconda.html). Download and install the latest Miniconda version for Python 3.9 for your operating system.  

Once it is done, you can check conda with :
```bash
conda -V
```


### 1.2 Check-out the git repository

Once Miniconda is ready, clone the repository :

```bash
git clone <repository_name>
```


## 2. Makefile commands

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

## 3. Create isolated virtual environment

Change directory (`cd`) into the project folder, then type:

```bash
make create_environment
conda activate zoidberg_env
```

If you want to see this virtual environment in the kernels registered by jupyter then use the command (be sure that the environment is actived):
```bash
ipython kernel install --user --name=zoidberg_env
```
## 3.1 Downloading data

Next step is to download data. Images are avalaible on Kaggle [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Then you have to move the folder `chest_Xray` in `data/raw`:
```bash
mv <path of the directory>/chest_Xray data/raw
```

## 3.2 Start Jupyter Notebook or JupyterLab

Finally, start from terminal :
```bash
make jupyter
```

Go on your browser to work on those jupyters notebooks

# Report 

The document in the report folder is written in Latex, we used [Overleaf](https://www.overleaf.com/). Please ask us if you want access to its tex version.  
Access : [https://www.overleaf.com/project/63fcbca4e88c8a6d5976e9c6](https://www.overleaf.com/project/63fcbca4e88c8a6d5976e9c6)

# Deployment in rust