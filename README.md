# EC2PPI
## A machine learning tool that integrates evolutionary coupling results from EVcomplex2 and AlphaFold 3 confidences metrics to predict protein-protein interactions.
### Abstract
Protein-protein interactions (PPIs) are fundamental to most biological processes and understanding them is crucial for drug discovery and disease research. Despite many existing experimental and computational approaches, detection of PPIs remains challenging. With the release of AlphaFold 3, new possibilities for PPI prediction have emerged. In this thesis I integrate AlphaFold 3 and evolutionary coupling information to develop EC2PPI, a machine learning pipeline based on decision tree ensembles. EC2PPI leverages features derived from AlphaFold 3 predictions and evolutionary coupling analysis to capture relevant characteristics that indicate PPIs. The modular design enables customization according to specific user requirements. Trained on a balanced benchmark dataset of mitochondrial proteins, EC2PPI achieved a ROC-AUC of 0.97, significantly outperforming baseline classifiers using only AlphaFold 3 or evolutionary coupling features. To assess feature contributions, I analyzed the separate impact of AlphaFold 3 and evolutionary-coupling inputs on predictive performance: AlphaFold 3 features alone provided strong predictive power, while evolutionary-coupling information contributed an incremental improvement. These findings highlight the potential of combining structural predictions with evolutionary signals to advance computational PPI detection.
### Requirements

EC2PPI is implemented in **Python 3.11** and is best managed via a Conda environment.

A `conda_environment.yaml` file is provided, which contains all required Python dependencies.  
You can create and activate the environment with:

```bash
conda env create -f environment.yaml
conda activate ec2ppi
```
### Usage

EC2PPI is a command-line tool that requires a single **parameter file** (`params.txt`) as input.  
This file specifies dataset paths, feature options, classifier settings, and output directories.

An example can be found in `data/example_setup_file/`

You can run the tool with:

```bash
python -m src.main example_setup_file/example_params.txt
```

### Input: `params.txt`

All runs of EC2PPI are configured through a single parameter file (e.g. `params.txt`).  
This file specifies what kind of run should be performed, which data to use,  
and where results are stored.  

There are **two run modes**:  
- **Training run** — requires positive and negative datasets.  
- **Prediction run** — requires a trained model and a dataset to predict on.  

The parameter file is written in a simple key–value format, one parameter per line.  

#### Required parameters
- `training run` *(bool)* — set `True` if running training.  
- `prediction run` *(bool)* — set `True` if running prediction.  
- `export directory` *(path)* — where trained models and evaluation results are stored.  
- `prediction export filepath` *(path)* — where predictions are saved (for prediction runs).  
- `model import filepath` *(path or "latest")* — trained model to use. `"latest"` uses the most recent model in the export directory.

#### Data-related parameters
For **training runs** you must provide:  
- `positive training complex info table filepath` — CSV file with positive protein pairs.  
- `negative training complex info table filepath` — CSV file with negative protein pairs.  
- (optional) `positive/negative training complex ec directory` — directory with evolutionary coupling data, if required.  
- (optional) `positive/negative training complex af3 directory` — directory with AlphaFold 3 output, if required.  

For **prediction runs** you must provide:  
- `prediction complex info table filepath` — CSV with protein pairs to predict.  
- (optional) `prediction complex ec directory` — EC data directory, if required by chosen features.  
- (optional) `prediction complex af3 directory` — AF3 data directory, if required by chosen features.  

The **info table CSV** must contain at least:  
- `prefix` — unique identifier for each protein pair  
- `uid1` — UniProt ID of protein 1  
- `uid2` — UniProt ID of protein 2  

#### Optional parameters
- `classifier type` — which model to use (`random forest` or `hist gradient boost`).  
- `training parameters` — parameter grid for classifier tuning.  
- `feature list` — list of features to include. Defaults to all non-custom features.  
- `parallelized feature calculation` — enable/disable parallelization.  

#### Notes
- EC and AF3 directories must contain files prefixed with the protein pair’s `prefix`.  
- If custom features are included in the info table, their column names must also appear in the `feature list`. Only numeric values are allowed.  
- Web-server AlphaFold results may need to be reformatted to match the local AlphaFold file structure.  

### Output

All results are written to the directory specified by `export directory` (for training)  
or `prediction export filepath` (for predictions).  

- **Training runs**  
  - Trained model (`.pkl` file) for later reuse  
  - Evaluation metrics (PR and ROC curves + `.json` file)
  - Calculated features (`.csv` file) 

- **Prediction runs**  
  - Predictions stored as a CSV file, including the input protein pairs and their predicted class/probability  
  - If `model import filepath = latest`, results are linked to the most recently trained model  

### Trained model 
In `data/example_input` a RF based model trained on default settings is included. The model was trained on a balanced data set of 1,418 mitochondrial protein pairs.
