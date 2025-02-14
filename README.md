# NFL Draft Assist Tool

This project leverages **Linear Regression**, **Monte Carlo Cross-Validation (MCCV)**, **KMeans Clustering**, and **Manual Thresholding**  to predict the success of NFL prospects. It integrates multiple datasets, including college stats, combine results, player body measurements, and historical NFL performance data.

## Features

- **Linear Regression** used as the primary predictive modeling technique, utilizing it to estimate the success scores of NFL draft prospects based different metrics
- **Monte Carlo Cross Validation** for robust player success prediction
- **KMeans Clustering/Manual Thresholding** to categorize players into tiers (All-Pro, Starter, Backup, etc.)
- **Flask Web App** for visualization of predictions
- **Standardized and Normalized Metrics** for accurate comparisons

## Installation & Setup

### 1. Clone the Repository

```sh
git clone git@github.com:VitalJoseph/vitaljoseph-allegheny-senior-comp-artifact.git
```

### Create and Activate Virtual Environment

```sh
python -m venv artifact_env
source artifact_env/bin/activate
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Run in terminal

```sh
python models/manual_mccv.py
# or
python models/cluster_mccv.py
```

### Run Flask App

```sh
python frontend/app.py
```
