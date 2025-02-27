# -*- coding: utf-8 -*-
"""
Signal Analysis and Clustering Script
"""

# LIBRARIES
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from s_dbw import S_Dbw

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configuration
SHOW_GRAPH = False
n_fuzz = 1

# Device and Path Configuration
device = "pinata"
base_folder = f'/home/analysis/'
list_nfs_path = f'/home/analysis/input_files/entries_lists/{device}'
power_calibration_path = f'/home/analysis/input_files/{device}/PWR_calibration_signals/'
power_entries_nfs_path = f'/home/analysis/input_files/{device}/PWR_operation_signals/'
save_path = '/home/analysis/input_files/output_figs/'

# Mapping Dictionaries
mapeo_strings = {
    '00': 1, '02': 2, '03': 3, '04': 4, '05': 5,
    '06': 6, '07': 7, '08': 8, '09': 9, '0a': 10,
    '0b': 11, '0c': 12, '0d': 13, '0e': 14, '0f': 15,
    'a3': 16, 'a1': 0, 'a0': 17
}

mapeo_bugs = {
    '00': 'E0101', '02': 'E0102', '03': 'E0103', '04': 'E0104', 
    '05': 'E0105', '06': 'E0106', '07': 'E0201', '08': 'E0202', 
    '09': 'E0203', '0a': 'E0204', '0b': 'E0205', '0c': 'E0206', 
    '0d': 'E0207', '0e': 'E0208', '0f': 'E0209', 'a3': 'E0210', 
    'a0': 'SUT00I', 'a1': 'SUT00F'
}

def robust_covariance(signal):
    """
    Detect outliers using robust covariance estimation
    """
    detector = EllipticEnvelope(contamination=0.1, assume_centered=True)
    return detector.fit(signal).predict(signal)

def outlier_detection(input_signal, imputation_model=6):
    """
    Detect and handle outliers in the input signal
    """
    n_samples = input_signal.shape[2]
    outlied_signal = np.zeros_like(input_signal, dtype=float)
    
    for i in range(input_signal.shape[0]):
        quartile_threshold = 50 if i < 2 else 5
        
        for j in range(input_signal.shape[1]):
            # Skip traces with all zeros
            if i < 2 and np.count_nonzero(input_signal[i][j] == 0) == n_samples:
                continue
            
            # Compute quartiles and IQR
            q1 = np.percentile(input_signal[i][j], 25)
            q3 = np.percentile(input_signal[i][j], 75)
            iqr = q3 - q1
            
            # Define fences
            fence_low = q1 - (quartile_threshold * iqr)
            fence_high = q3 + (quartile_threshold * iqr)
            
            # Handle outliers based on imputation model
            signal = pd.DataFrame(input_signal[i][j].copy().reshape(1, -1)).astype(float)
            
            if imputation_model == 2:  # Mean imputation
                signal.T[(input_signal[i][j] <= fence_low)] = np.mean(input_signal[i][j])
                signal.T[(input_signal[i][j] >= fence_high)] = np.mean(input_signal[i][j])
            
            elif imputation_model == 3:  # Median imputation
                signal.T[(input_signal[i][j] <= fence_low)] = np.median(input_signal[i][j])
                signal.T[(input_signal[i][j] >= fence_high)] = np.median(input_signal[i][j])
            
            elif imputation_model == 4:  # LOCF (Last Observation Carried Forward)
                signal.T[(input_signal[i][j] <= fence_low)] = np.nan
                signal.T[(input_signal[i][j] >= fence_high)] = np.nan
                signal = signal.T.fillna(method='bfill').T
            
            elif imputation_model == 5:  # NOCB (Next Observation Carried Backward)
                signal.T[(input_signal[i][j] <= fence_low)] = np.nan
                signal.T[(input_signal[i][j] >= fence_high)] = np.nan
                signal = signal.T.fillna(method='ffill').T
            
            elif imputation_model == 6:  # Linear interpolation
                signal.T[(input_signal[i][j] <= fence_low)] = np.nan
                signal.T[(input_signal[i][j] >= fence_high)] = np.nan
                signal = signal.T.interpolate(method='linear').T
            
            elif imputation_model == 7:  # Spline interpolation
                signal.T[(input_signal[i][j] <= fence_low)] = np.nan
                signal.T[(input_signal[i][j] >= fence_high)] = np.nan
                signal = signal.T.interpolate(method='spline').T
            
            else:  # Zero imputation
                signal.T[(input_signal[i][j] <= fence_low)] = 0
                signal.T[(input_signal[i][j] >= fence_high)] = 0
            
            outlied_signal[i][j] = signal.ravel()
    
    return outlied_signal

def robust_covariance_procedure(outlied_signal):
    """
    Apply robust covariance detection to outlied signals
    """
    n_samples = outlied_signal.shape[2]
    robust_samples = np.zeros_like(outlied_signal, dtype=int)
    
    for test in range(outlied_signal.shape[0]):
        # Remove null traces
        if test < 2:
            zero_traces = np.where(np.count_nonzero(outlied_signal[test] == 0, axis=1) == n_samples)[0]
        
        for i in range(outlied_signal.shape[1]):
            if test == 2 or (test < 2 and i not in zero_traces):
                robust_samples[test][i] = robust_covariance(np.transpose(outlied_signal[test][i, np.newaxis]))
    
    return robust_samples

def pca_technique_application(robust_samples):
    """
    Apply PCA to robust samples and determine optimal number of components
    """
    # Prepare data for PCA
    data = pd.DataFrame(np.transpose(robust_samples[2]))
    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA().fit(data_rescaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Visualize PCA variance explanation
    plt.figure(figsize=(20,6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance (%)')
    plt.title('Variance Explained by PCA Components')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.savefig(f"{save_path}{device}_PCA_variance.png")
    plt.close()
    
    # Determine optimal number of components
    slope = np.diff(cumulative_variance) / np.diff(range(1, len(cumulative_variance) + 1))
    sharp_slope_indices = [i for i in range(1, len(slope)) 
                            if abs(slope[i] - slope[i-1]) < 1e-4]
    
    n_components = sharp_slope_indices[0] + 1 if sharp_slope_indices else len(cumulative_variance)
    
    # Apply PCA with selected components
    print(f"Using {n_components} components in PCA")
    pca = PCA(n_components=n_components, whiten=False)
    
    # Remove null traces
    reshaped_robust_samples = robust_samples.reshape(-1, robust_samples.shape[2])
    zero_traces = np.where(np.count_nonzero(reshaped_robust_samples == 0, axis=1) == robust_samples.shape[2])[0]
    reshaped_robust_samples = np.delete(reshaped_robust_samples, zero_traces, axis=0)
    
    # Transform data
    pca_samples = pca.fit_transform(reshaped_robust_samples)
    
    return pca_samples

def clustering_procedure(pca_samples, entries_list):
    """
    Perform clustering on PCA-transformed data
    """
    n_traces = len(entries_list)
    n_tests = int(np.sqrt(n_traces))
    
    # Prepare data for clustering
    instances = pca_samples.shape[0]
    neighbors = NearestNeighbors(n_neighbors=n_tests).fit(pca_samples)
    distances, _ = neighbors.kneighbors(pca_samples)
    distances = np.sort(distances, axis=0)[:, 1]
    
    # Determine epsilon for clustering
    eps = distances[instances - n_tests - 1] if n_tests > 0 else distances[round(instances * 0.99)]
    
    # Perform HDBSCAN clustering
    min_samples = round(n_tests * 0.8)
    clusterer = HDBSCAN(cluster_selection_epsilon=eps, min_samples=1)
    labels = clusterer.fit_predict(pca_samples)
    
    # Compute cluster metrics
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print("Cluster Counts:", dict(zip(unique_labels, label_counts)))
    
    # Compute clustering metrics
    silhouette = silhouette_score(pca_samples, labels)
    calinski = calinski_harabasz_score(pca_samples, labels)
    davies = davies_bouldin_score(pca_samples, labels)
    sdbw = S_Dbw(pca_samples, labels, method='Halkidi', metric='euclidean')
    
    # Visualization of clustering metrics
    plt.figure(figsize=(16,8))
    plt.plot(range(len(entries_list)), [silhouette]*len(entries_list), label='Silhouette Score')
    plt.plot(range(len(entries_list)), [calinski]*len(entries_list), label='Calinski-Harabasz Score')
    plt.plot(range(len(entries_list)), [davies]*len(entries_list), label='Davies-Bouldin Score')
    plt.plot(range(len(entries_list)), [sdbw]*len(entries_list), label='S_Dbw Score')
    plt.title('Clustering Performance Metrics')
    plt.xlabel('Number of Elements')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f"{save_path}{device}_clustering_metrics.png")
    plt.close()
    
    # Convert numerical labels to bug codes
    unique_entries = np.unique(entries_list)
    true_labels = [mapeo_bugs[entry] for entry in unique_entries]
    
    return true_labels, labels

def processing(input_signal, entries_list):
    """
    Main processing pipeline for signal analysis
    """
    # Outlier detection and imputation
    outlied_signal = outlier_detection(input_signal)
    
    # Robust covariance procedure
    robust_samples = robust_covariance_procedure(outlied_signal)
    
    # PCA application
    pca_samples = pca_technique_application(robust_samples)
    
    # Clustering
    true_labels, predicted_labels = clustering_procedure(pca_samples, entries_list)
    
    return true_labels, predicted_labels

def operation():
    """
    Main operation function for signal processing
    """
    global date
    
    # PWR Signals Collection
    power_exec_name = [f for f in os.listdir(power_calibration_path) 
                       if f.startswith(f"{device}_PWR_exec_")][0]
    power_noexec_name = [f for f in os.listdir(power_calibration_path) 
                         if f.startswith(f"{device}_PWR_noexec")][0]
    power_fuzz_name = f"{device}_PWR_bugs_{date}.csv"
    print("Entries signals files: ", power_fuzz_name)
    
    # Load Power Signals
    power_signal = np.loadtxt(os.path.join(power_calibration_path, power_exec_name), delimiter=',')
    power_cal_traces = power_signal.shape[0]
    n_samples = power_signal.shape[1]
    
    # Prepare Input Signal
    power_input_signal = np.zeros((3, n_traces, n_samples))
    power_input_signal[0, :power_cal_traces] = np.abs(np.loadtxt(os.path.join(power_calibration_path, power_exec_name), delimiter=','))
    power_input_signal[1, :power_cal_traces] = np.abs(np.loadtxt(os.path.join(power_calibration_path, power_noexec_name), delimiter=','))
    power_input_signal[2] = np.abs(np.loadtxt(os.path.join(power_entries_nfs_path, power_fuzz_name), delimiter=','))
    
    # Process Power Signals
    true_labels, predicted_labels = processing(power_input_signal, entries_list)
    
    return true_labels, predicted_labels

# Ensure all required directories exist
def create_directories():
    """
    Create necessary directories for the analysis
    """
    directories = [
        base_folder,
        list_nfs_path,
        power_calibration_path,
        power_entries_nfs_path,
        save_path
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """
    Main function to run the analysis
    """
    global date, n_traces, n_tests, entries_list
    
    # Create required directories
    create_directories()
    
    for iteration in range(n_fuzz):
        print(f"ITERATION {iteration + 1}")
        
        # Example entries
        entries = "a0,a1,00,02,03,04,05,06,07,08,09,0a,0b,0c,0d,0e,0f,a3"
        
        # Predefined date for testing
        date = datetime.datetime.now().strftime("%Y_%m_%d_%Ho%M")
        
        # Load entries list
        with open(os.path.join(list_nfs_path, f"_entries_list_{date}.csv"), 'r') as file:
            entries_list = [line.strip() for line in file]
        
        # Prepare entries list
        number_entries_list = [mapeo_strings[entry] for entry in entries_list]
        
        # Set number of traces and tests
        n_traces = len(entries_list)
        n_tests = int(np.sqrt(n_traces))
        
        # Perform operation
        true_labels, predicted_labels = operation()
        
        # Print final results
        print("Iteration Complete")

# Run the main function
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time} seconds")