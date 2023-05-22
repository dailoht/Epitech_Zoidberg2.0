# Zoidberg2.0 : Computer aided diagnosis

The aim of this project is to use machine learning to help doctors detecting pneumonia. We, thus, have to classify x-ray images into 3 classes : *normal*, *viral pneumonia*, *bacterial pneumonia*.  
The project has 2 parts :
- Analyze and model building in python. All results can be shown in notebooks
- Model deployment

# Architecture

Th directory structure below is widely inspired by the [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science) project.

```
.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── interim         <- Intermediate data that has been transformed
│   ├── processed       <- Final, canonical data for modeling
│   └── raw             <- Original data dump
│
├── deployment          <- API & Front for model deployment
│
├── models              <- Trained models (history, checkpoint, etc.)
│
├── notebooks
│   ├── data_processing <- Jupyter notebooks for data analysis and
│   │                      preprocessing
│   ├── models          <- Jupyter notebooks to train models
│   └── visualization   <- Jupyter notebooks to make some useful
│                          visualizations
│
├── report              <- Generated report
│
├── requirements.yml    <- Requirements file for reproducing the analysis
│                          environment
│
├── src                 <- Source code for use in this project
    ├── data            <- Scripts to handle data (download, metrics,
    │                      tensorflow)
    ├── tests           <- Unit testing
    └── visualization   <- Scripts to help visualizations
```

# Visualization with Notebook

Notebooks can be run either with [Google Colab](https://colab.research.google.com/) or in a local environment. Steps below are provided to run them locally. If you want to use Colab, you need to load this repo on your google drive, change drive folder in notebooks and upload needed data [Section 4](#31-downloading-data).


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
flask                Start Flask server
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
## 4. Downloading data

Next step is to download data. Images are avalaible on Kaggle [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Then you have to move either zip file or folder `chest_Xray` in `data/raw`:
```bash
mv <path of the directory>/chest_Xray data/raw
```

## 5. Start Jupyter Notebook or JupyterLab

Finally, start from terminal :
```bash
make jupyter
```

Go on your browser to work on those jupyters notebooks

# Report 

The document in the report folder is written in Latex, we used [Overleaf](https://www.overleaf.com/). Please ask us if you want access to its tex version.  
Access : [https://www.overleaf.com/project/63fcbca4e88c8a6d5976e9c6](https://www.overleaf.com/project/63fcbca4e88c8a6d5976e9c6)

# Deployment with Flask

We offer a deployment of the model in a Flask application where you can upload an image and it will output the class of the image.

To start flask server, just run:
```bash
make flask
```

Then go to the address 127.0.0.1:5000 in your web browser.