# BCALN

BCALN is a research codebase for predicting axillary lymph node status in breast cancer using MRI data and deep learning models.  
The repository provides model implementations, utility functions, and sample data to facilitate testing of the full pipeline.

---

## Overview

The repository includes multiple model variants and supporting utilities:

- Model implementations under different submodules (e.g. `NSLN/`, `SLN/`, `SLNN/`)
- Shared helper functions in `utils/`
- A sample dataset in `sample_data/` for pipeline testing

---

## System requirements

### Operating system
The code has been tested on a CentOS 7 environment.

### Software requirements
- Python 3.8 or later

All required Python packages are listed in `requirements.txt`.

Typical core dependencies include:
- numpy  
- pandas  
- scikit-learn  
- PyTorch  
- MONAI  
- OpenCV  

No non-standard hardware is required.

---

## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Installation typically takes a few minutes, depending on network conditions.

---

## Running with sample data

A sample dataset is provided in the sample_data/ directory. The training and testing pipeline is configuration-driven.

To run the pipeline with the provided sample data:

```bash
python SLNN/train_test_tumor_axillary.py
```

The script internally reads:

- sample_data/dataset.csv
- feature files under sample_data/fd/

Paths and training settings are defined in:

SLNN/cfg_t_a.py

### Output

During execution, the script will:

- Train and/or evaluate the model according to the configuration
- Save checkpoints to the directory specified in the configuration (e.g. ckpts/)
- Log intermediate results and metrics

### Using the code on your own data

To apply the pipeline to your own dataset:

- Prepare feature files in NumPy (.npy) format or pickle (.pkl) format depending on the task.
- Create a CSV file following the same structure as sample_data/dataset.csv.
- Update data paths and parameters in the corresponding configuration file (e.g. cfg_t_a.py).

## Repository structure

```
BCALN/
├── NSLN/
├── SLN/
├── SLNN/
│   ├── train_test_tumor_axillary.py
│   ├── cfg_t_a.py
├── utils/
├── sample_data/
│   ├── dataset.csv
│   └── fd/
│       └── *.npy
├── requirements.txt
└── README.md
```
