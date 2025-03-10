# Power Consumption Analysis PoC for STM32-F429ZI Ethernet

This folder contains a Proof of Concept (PoC) implementation for performing Side-Channel Analysis (SCA) based on power consumption measurements on the STM32 NUCLEO-144 (STM32-F429ZI) board via its Ethernet interface.

## Overview

This PoC showcases an automated pipeline for power consumption analysis on the STM32-F429ZI platform. Key features include:

- Power trace collection from calibration and bug-induced executions
- Preprocessing of power traces with outlier detection and imputation
- Dimensionality reduction using Principal Component Analysis (PCA)
- Clustering of power traces to detect execution patterns
- Visualization of clustering results and anomaly classification

Note: The signals in this PoC are illustrative examples and differ from those in the paper's reported results.

## File Structure

- `PWR_monitoring.py`: Main Python script that orchestrates power trace extraction, preprocessing, analysis, clustering, and visualization.
- `lecroy3.py`: Helper module for interfacing with LeCroy oscilloscopes to acquire power traces.
- `STM32-F429ZI_PWR_bugs_2024_02_20_10o32_part_aa` to `part_am`: Split archive files containing sample power consumption traces. These are recombined and extracted by the script into a single ZIP file.

## Prerequisites

The script requires the following Python libraries:

- `scikit-learn` (machine learning and clustering)
- `numpy` (numerical operations)
- `pandas` (data manipulation)
- `matplotlib` (visualization)
- `seaborn` (enhanced visualization)
- `scipy` (signal processing and statistics)
- `PyVISA` (oscilloscope communication)
- `paramiko` and `scp` (SSH and file transfer)
- `s_dbw` (clustering validity metric)

Install dependencies using:
```bash
pip install -r requirements.txt