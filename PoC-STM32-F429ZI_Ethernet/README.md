# Power Consumption Analysis PoC for STM32-F429ZI Ethernet

This folder contains the Proof of Concept (PoC) implementation for performing Side-Channel Analysis (SCA) based on power consumption measurements on the STM32 NUCLEO-144 (STM32-F429ZI) board using its Ethernet interface.

## Overview

This PoC demonstrates automated power consumption analysis on the STM32-F429ZI platform. The implementation includes:

- Power trace collection and preprocessing
- Anomaly detection using robust statistical methods
- Dimensionality reduction through Principal Component Analysis (PCA)
- Clustering of power traces to identify different execution patterns
- Classification of potential security vulnerabilities or bugs

## Files Structure

- **PWR_monitoring.py**: Main Python script that handles the entire process including power trace extraction, preprocessing, analysis, and visualization.
- **lecroy3.py**: Helper module for interfacing with LeCroy oscilloscopes (used for power trace acquisition).
- **STM32-F429ZI_PWR_bugs_2024_02_20_10o32_part_aa through part_am**: Split archive containing sample power consumption traces from the STM32-F429ZI board. These files are automatically recombined and extracted by the script.

## Prerequisites

The script requires several Python libraries for signal processing, machine learning, and visualization:

- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- PyVISA (for oscilloscope communication)
- paramiko (for SSH communication)
- s_dbw (for clustering validity assessment)

## Functionality

The PoC implements a comprehensive pipeline for power consumption analysis:

1. **Data Preparation**:
   - Recombines the split archive files into a single ZIP file
   - Extracts power consumption traces from the archive
   - Organizes data into execution, non-execution, and test samples

2. **Signal Preprocessing**:
   - Outlier detection and removal using quantile-based methods
   - Multiple imputation techniques for handling anomalous data points
   - Robust covariance estimation for noise reduction

3. **Feature Extraction**:
   - Principal Component Analysis (PCA) for dimensionality reduction
   - Automatic determination of optimal number of components
   - Variance analysis and cumulative variance visualization

4. **Clustering and Classification**:
   - HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) for identifying execution patterns
   - Cluster validation using silhouette score, Davies-Bouldin index, and Calinski-Harabasz index
   - Confusion matrix generation for evaluating classification performance

5. **Visualization**:
   - 3D cluster visualization of the first three principal components
   - Confusion matrix visualization for bug/vulnerability classification
   - PCA explained variance plots

## Usage

To run the PoC:

```python
python PWR_monitoring.py
