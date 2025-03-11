#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:40:14 2023

@author: [Anonymous]
"""

# LIBRARIES
from sklearn.metrics import *
import time
import datetime
import numpy as np
import serial
import matplotlib.pyplot as plt
import csv
import pandas as pd
import pyvisa as visa
from lecroy3 import *
from numpy import loadtxt
from itertools import chain, zip_longest
import seaborn as sns
import scipy.integrate as integrate
from sklearn.impute import SimpleImputer
import scipy.signal  
from scipy.stats import pearsonr, spearmanr
from numpy.random import normal
import random
import scipy.stats as stats
from numpy import exp, savetxt
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift, estimate_bandwidth, OPTICS
from sklearn.utils.multiclass import unique_labels
from s_dbw import S_Dbw
from scipy.stats import norm
import paramiko
from scp import SCPClient
import os.path
from threading import *
import os
from sklearn.metrics import confusion_matrix
from scipy.fft import fft, rfft
import subprocess
import threading 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from collections import Counter
import zipfile

# Warning configuration
import warnings
warnings.filterwarnings('ignore')

SHOW_GRAPH = False

device_ip = "X.X.X.X"  # Anonymized IP
local_ip = "Y.Y.Y.Y"   # Anonymized IP
n_fuzz = 1
device = "STM32-F429ZI"
date = '2024_02_21_09o34'

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the pattern for the parts
zip_part_prefix = os.path.join(script_dir, 'STM32-F429ZI_PWR_bugs_' + date + '_part_')

# Output zip filename to combine parts
zip_filename = os.path.join(script_dir, 'STM32-F429ZI_PWR_bugs_' + date + '.zip')

# Check if all parts exist (parts are named part_aa, part_ab, ..., part_am)
parts_exist = True
for part_suffix in ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an']:
    part_filename = f"{zip_part_prefix}{part_suffix}"
    if not os.path.isfile(part_filename):
        print(f"Error: Part file '{part_filename}' not found.")
        parts_exist = False
        break

# If all parts exist, recombine them
if parts_exist:
    try:
        print(f"Recombining zip parts into one file...")
        with open(zip_filename, 'wb') as f_out:
            for part_suffix in ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an']:
                part_filename = f"{zip_part_prefix}{part_suffix}"
                with open(part_filename, 'rb') as f_in:
                    f_out.write(f_in.read())
        print(f"Recombined file created: {zip_filename}")
    except Exception as e:
        print(f"An error occurred while recombining the zip parts: {e}")
else:
    print("Skipping recombination as some parts are missing.")

# Now, unzip the recombined file if it exists
if os.path.isfile(zip_filename):
    # Destination directory (in the same folder as the script)
    extract_to = os.path.join(script_dir, 'STM32-F429ZI_PWR_bugs_' + date)

    # Create the destination directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Try to unzip the file
    try:
        print(f"Starting to unzip PoC signals...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f'File successfully unzipped to: {extract_to}')
    except zipfile.BadZipFile:
        print(f"Error: '{zip_filename}' is not a valid ZIP file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Define paths based on extraction
    base_path = './STM32-F429ZI_PWR_bugs_' + date + '/'
    list_nfs_path = base_path + device  # Anonymized path
    PWR_calibration_path = base_path # Anonymized path
    PWR_entries_nfs_path = base_path  # Anonymized path
    save_path = base_path  # Anonymized path



# FIXED VARIABLES FOR TESTING
n_files = 3
pca_plot = 0
clust = 0
silh = 0
imp_model = 6

# GLOBAL PARAMETERS
gauss_dict = dict([(1, 0.682), (2, 0.954)])
val = 1
variance_coefficient = []
max_n_components_pca = []
labels_array = []

# Mapping dictionaries for class labels
mapeo_strings = {
    '00': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7, '08': 8,
    '09': 9, '0a': 10, '0b': 11, '0c': 12, '0d': 13, '0e': 14, '0f': 15,
    'a3': 16, 'a1': 0, 'a0': 17
}

mapeo_bugs = {
    '00': 'E0101', '02': 'E0102', '03': 'E0103', '04': 'E0104', '05': 'E0105',
    '06': 'E0106', '07': 'E0201', '08': 'E0202', '09': 'E0203', '0a': 'E0204',
    '0b': 'E0205', '0c': 'E0206', '0d': 'E0207', '0e': 'E0208', '0f': 'E0209',
    'a3': 'E0210', 'a0': 'SUT00I', 'a1': 'SUT00F'
}

def get_class_name(obj):
    return type(obj).__name__

def outlier_detection(input_signal, kind):
    global n_samples, n_files, imp_model
    outlied_signal = [[[0 for z in range(np.shape(input_signal)[2])] for y in range(np.shape(input_signal)[1])] 
                      for x in range(np.shape(input_signal)[0])]
    n_it = np.shape(outlied_signal)[0]
    
    for i in range(n_it):
        quartile_threshold = 50 if i != 1 else 5
        indice_25000_ceros = 0
        
        if i < 2:
            ceros_por_fila = np.count_nonzero(input_signal[i] == 0, axis=1)
            indice_25000_ceros = np.where(ceros_por_fila == n_samples)[0]
        
        for j in range(np.shape(outlied_signal)[1]):
            if i == 2 or (i < 2 and j not in indice_25000_ceros):
                q1 = pd.DataFrame(input_signal[i][j]).quantile(0.25)[0]
                q3 = pd.DataFrame(input_signal[i][j]).quantile(0.75)[0]
                iqr = q3 - q1
                fence_low = q1 - (quartile_threshold * iqr)
                fence_high = q3 + (quartile_threshold * iqr)
                
                if imp_model == 2:  # Mean imputation
                    outlied_signal[i][j] = input_signal[i][j].copy()
                    outlied_signal[i][j][(input_signal[i][j] <= fence_low)] = np.mean(outlied_signal[i][j])
                    outlied_signal[i][j][(input_signal[i][j] >= fence_high)] = np.mean(outlied_signal[i][j])
                elif imp_model == 3:  # Median imputation
                    outlied_signal[i][j] = input_signal[i][j].copy()
                    outlied_signal[i][j][(input_signal[i][j] <= fence_low)] = np.median(outlied_signal[i][j])
                    outlied_signal[i][j][(input_signal[i][j] >= fence_high)] = np.median(outlied_signal[i][j])
                elif imp_model == 4:  # LOCF
                    signal = pd.DataFrame(input_signal[i][j].copy().reshape(1, -1)).astype(float)
                    signal.T[(input_signal[i][j] <= fence_low)] = np.nan
                    signal.T[(input_signal[i][j] >= fence_high)] = np.nan
                    outlied_signal[i][j] = signal.T.fillna(method='bfill').T.to_numpy()
                elif imp_model == 5:  # NOCB
                    signal = pd.DataFrame(input_signal[i][j].copy().reshape(1, -1)).astype(float)
                    signal.T[(input_signal[i][j] <= fence_low)] = np.nan
                    signal.T[(input_signal[i][j] >= fence_high)] = np.nan
                    outlied_signal[i][j] = signal.T.fillna(method='ffill').T.to_numpy()
                elif imp_model == 6:  # Linear interpolation
                    signal = pd.DataFrame(input_signal[i][j].copy().reshape(1, -1)).astype(float)
                    signal.T[(input_signal[i][j] <= fence_low)] = np.nan
                    signal.T[(input_signal[i][j] >= fence_high)] = np.nan
                    outlied_signal[i][j] = signal.T.interpolate(method='linear').T.to_numpy()
                elif imp_model == 7:  # Spline interpolation
                    signal = pd.DataFrame(input_signal[i][j].copy().reshape(1, -1)).astype(float)
                    signal.T[(input_signal[i][j] <= fence_low)] = np.nan
                    signal.T[(input_signal[i][j] >= fence_high)] = np.nan
                    outlied_signal[i][j] = signal.T.interpolate(method='spline').T.to_numpy()
                else:  # Zeroes imputation
                    outlied_signal[i][j] = input_signal[i][j].copy()
                    outlied_signal[i][j][(input_signal[i][j] <= fence_low)] = 0
                    outlied_signal[i][j][(input_signal[i][j] >= fence_high)] = 0
                outlied_signal[i][j] = outlied_signal[i][j].ravel()
    
    outlied_signal = [np.atleast_1d(np.asarray(x, dtype=np.int64)) for x in outlied_signal]
    return outlied_signal

def pca_technique_application(robust_samples, kind):
    global n_traces, n_samples, max_n_components_pca, save_path, device, date, pca_samples
    pca_samples = np.array([[0 for y in range(np.shape(robust_samples)[2])] 
                           for x in range(np.shape(robust_samples)[0] * np.shape(robust_samples)[1])], dtype=object)
    signal_name = ["OP signal", "NOP signal", "Bugs signal"]
    
    # PCA TRAINING
    datos = pd.DataFrame(np.transpose(robust_samples[2]))
    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(datos)
    pca = PCA().fit(data_rescaled)
    xi = np.arange(1, np.shape(robust_samples)[1] + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.rcParams["figure.figsize"] = (10, 6)
    fig, ax = plt.subplots()
    plt.ylim(0.0, 1.10)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylabel('Cumulative variance (%)', fontsize=20)
    plt.title('Components Needed to Explain Variance (PWR)', fontsize=20)
    plt.axhline(y=0.99, color='r', linestyle='-')
    plt.text(0.5, 1.02, '99% cut-off threshold', color='red', fontsize=16)
    ax.grid(axis='both')
    plt.savefig(save_path + device + "_entries_list_" + date + "_PCA_PWR")
    plt.show()
    
    slope = np.diff(y) / np.diff(xi)
    sharp_slope_indices = [i for i in range(1, len(slope)) if np.abs(slope[i] - slope[i - 1]) < 1e-4]
    n_components = sharp_slope_indices[0] + 1 if sharp_slope_indices else None
    
    plt.figure(dpi=100)
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.plot(xi, np.abs(np.gradient(y / (xi / n_traces), xi)))
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid(axis='both')
    plt.xlabel('Number of Components', fontsize=15)
    plt.axvline(x=n_components, color='r', linestyle='-')
    plt.title('Curve Meaning | Explained Variance/(n_element/n_traces) |,\n for {} measuring PWR'.format(device), fontsize=15)
    plt.text(n_components, 0.0, n_components, color='red', fontsize=15)
    plt.savefig(save_path + device + "_entries_list_" + date + "_PCA_gradient_curve_PWR")
    plt.show()
    
    max_n_components_pca = n_components
    reshaped_robust_samples = np.array(robust_samples).reshape(np.shape(robust_samples)[0] * np.shape(robust_samples)[1], 
                                                              np.shape(robust_samples)[2])
    ceros_por_fila = np.count_nonzero(reshaped_robust_samples == 0, axis=1)
    indice_25000_ceros = np.where(ceros_por_fila == n_samples)[0]
    reshaped_robust_samples = np.delete(reshaped_robust_samples, indice_25000_ceros, axis=0)
    
    print("Using {} components in PCA".format(max_n_components_pca))
    pca = PCA(n_components=max_n_components_pca, whiten=False).fit(reshaped_robust_samples)
    pca_samples = pca.transform(reshaped_robust_samples)
    return pca_samples

def clustering_procedure(pca_samples, kind):
    global entries_list, n_tests, cal_n_traces, n_traces, y_pred, gauss_dict, mapeo_strings, mapeo_bugs, val, save_path, device, date
    
    print("Performing clustering...")
    silh_score = [np.nan for x in range(n_traces)]
    calinski_score = np.array([np.nan for x in range(n_traces)])
    davies_score = [np.nan for x in range(n_traces)]
    sdbw_score = [np.nan for x in range(n_traces)]
    labels_array = [[np.nan for y in range(2 * cal_n_traces + n_traces)] for x in range(n_traces)]
    cons_values = int(i)
    max_nans = int(gauss_dict[val] * cons_values)
    
    for index in range(1, n_traces + 1):
        data = pca_samples[:(2 * cal_n_traces + index)]
        instances = np.shape(data)[0]
        neighbors = NearestNeighbors(n_neighbors=n_tests).fit(data)
        distances, _ = neighbors.kneighbors(data)
        distances = np.sort(distances, axis=0)[:, 1]
        eps = distances[instances - n_tests - 1]
        if n_tests == 0:
            eps = distances[round(instances * 0.99)]
        min_samples = round(n_tests * 0.8)
        
        db = HDBSCAN(cluster_selection_epsilon=eps, min_samples=1)
        y_pred = db.fit_predict(data)
        labels_array[index - 1] = db.labels_
        
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise_ = list(db.labels_).count(-1)
        
        unique, counts = numpy.unique(y_pred, return_counts=True)
        print("Clustering Results:", dict(zip(unique, counts)))
        
        fig, ax1 = plt.subplots(figsize=(20, 8))
        x = np.linspace(2 * cal_n_traces + 1, 2 * cal_n_traces + stop_index, stop_index)
        plt.plot(x, silh_score[:stop_index], '-^b', label='Silhouette score')
        plt.plot(x, davies_score[:stop_index], '--g', label='Davies-Bouldin score')
        ax1.set_ylabel("Score", color='g', fontsize=18)
        ax1.set_xlabel("Number of elements per cluster", fontsize=18)
        ax1.tick_params(axis='x', labelsize=18)
        
        ax2 = ax1.twinx()
        plt.plot(x, calinski_score[:stop_index], ':r', label='Calinski-Harabasz score')
        ax2.set_ylabel("Score", color='r', fontsize=18)
           
    bugs_entries_list = [mapeo_bugs[valor] for valor in entries_list]
    etiquetas_unicas_true = np.unique(bugs_entries_list)
    etiquetas_unicas_pred = np.unique(y_pred[2 * cal_n_traces:])
    
    confus_matrix = np.zeros((len(etiquetas_unicas_true), len(etiquetas_unicas_pred)))
    for true_label, pred_label in zip(bugs_entries_list, y_pred[2 * cal_n_traces:]):
        confus_matrix[np.where(etiquetas_unicas_true == true_label)[0][0],
                      np.where(etiquetas_unicas_pred == pred_label)[0][0]] += 1
    
    y_pred_changed = np.copy(y_pred).astype(str)
    for i in range(len(etiquetas_unicas_true)):
        label = np.argmax(confus_matrix[i])
        rep = len(np.where(confus_matrix[i] == np.max(confus_matrix[i]))[0])
        for label in range(rep):
            index = np.where(confus_matrix[i] == np.max(confus_matrix[i]))[0][label]
            if isinstance(etiquetas_unicas_pred[index], np.int64):
                y_pred_changed[y_pred == etiquetas_unicas_pred[index]] = etiquetas_unicas_true[i]
    
    etiquetas_unicas_pred = np.unique(y_pred_changed[2 * cal_n_traces:])
    valores_nuevos = np.setdiff1d(etiquetas_unicas_true, etiquetas_unicas_pred)
    etiquetas_unicas_pred = np.union1d(etiquetas_unicas_pred, valores_nuevos)
    etiquetas_unicas_pred = np.roll(etiquetas_unicas_pred, -np.where(etiquetas_unicas_pred == 'E0101')[0])
    
    confus_matrix = np.zeros((len(etiquetas_unicas_true), len(etiquetas_unicas_pred)))
    for true_label, pred_label in zip(bugs_entries_list, y_pred_changed[2 * cal_n_traces:]):
        confus_matrix[np.where(etiquetas_unicas_true == true_label)[0][0],
                      np.where(etiquetas_unicas_pred == pred_label)[0][0]] += 1
    
    plt.figure(figsize=(20, 12), dpi=100)
    plt.imshow(confus_matrix, cmap=plt.get_cmap('GnBu'), interpolation='nearest')
    for i in range(len(etiquetas_unicas_true)):
        for j in range(len(etiquetas_unicas_pred)):
            plt.text(j, i, str(int(confus_matrix[i, j])), ha="center", va="center", color="black", fontsize=22)
    plt.title('Confusion Matrix (PWR), {}, Time Domain in {}'.format(get_class_name(db), device), fontsize=25)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)
    plt.xticks(np.arange(len(etiquetas_unicas_pred)), etiquetas_unicas_pred, rotation=90)
    plt.yticks(np.arange(len(etiquetas_unicas_true)), etiquetas_unicas_true)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.xlabel('Assigned Labels', fontsize=22)
    plt.ylabel('Actual Labels', rotation=90, verticalalignment='center', fontsize=22)
    plt.savefig(save_path + device + "_entries_list_" + date + "_CONFUSION_MATRIX_TIMEDOMAIN_" + str(get_class_name(db)) + "_PWR.png", bbox_inches='tight', dpi=100)
    plt.show()
    
    categorical_colors = sns.color_palette("tab20")
    fig = plt.figure(figsize=(16, 16), dpi=100)
    ax = fig.add_subplot(projection='3d')
    alpha_value = 0.8
    sorted_labels = sorted(set(y_pred_changed[2 * cal_n_traces + 1:]))
    legend_elements = []
    string_labels = [label for label in sorted_labels if any(char.isalpha() for char in label)]
    numeric_labels = [label for label in sorted_labels if label.isdigit()]
    
    for i in range(len(string_labels)):
        label = string_labels[i]
        color = categorical_colors[i % len(categorical_colors)]
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label, alpha=alpha_value))
    
    if legend_elements:
        ax.legend(handles=legend_elements, title='Clusters', fontsize=18)
    ax.get_legend().get_title().set_fontsize(20)
    
    for i in range(2 * cal_n_traces + 1, len(y_pred)):
        label = y_pred_changed[i]
        x_val = data[i, int(np.round(np.shape(pca_samples)[1] / 2) - 1)]
        y_val = data[i, int(np.round(np.shape(pca_samples)[1] / 2))]
        z_val = data[i, int(np.round(np.shape(pca_samples)[1] / 2) + 1)]
        color = categorical_colors[string_labels.index(label)] if label in string_labels else 'gray'
        ax.scatter(x_val, y_val, z_val, zdir='y', color=color, s=25, marker="o", alpha=alpha_value)
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20, rotation=0)
    plt.title("Clustering considering {} elements (PWR), {}, Time Domain in {}".format(n_traces, str(get_class_name(db)), device), fontsize=25)
    plt.savefig(save_path + device + "_entries_list_" + date + "_" + str(get_class_name(db)) + "_PWR.png", bbox_inches='tight', dpi=100)
    plt.show()
    return etiquetas_unicas_true, etiquetas_unicas_pred

def processing(input_signal, kind):
    globals()[f"{kind}_outlied_signal"] = outlier_detection(input_signal, kind)
    globals()[f"{kind}_robust_samples"] = robust_covariance_procedure(globals()[f"{kind}_outlied_signal"], kind)
    globals()[f"{kind}_pca_samples"] = pca_technique_application(globals()[f"{kind}_robust_samples"], kind)
    globals()[f"{kind}_etiquetas_unicas_true"], globals()[f"{kind}_etiquetas_unicas_pred"] = clustering_procedure(globals()[f"{kind}_pca_samples"], kind)

def operation():
    global PWR_calibration_path, PWR_entries_nfs_path, silh, pca_plot, imp_model, n_traces, n_samples, date, cal_traces, device, PWR_input_signal
    
    PWR_exec_name = [filename for filename in os.listdir(PWR_calibration_path) if filename.startswith(device + "_PWR_exec_")][0]
    PWR_noexec_name = [filename for filename in os.listdir(PWR_calibration_path) if filename.startswith(device + "_PWR_noexec")][0]
    PWR_fuzz_name = device + "_PWR_bugs_" + date + ".csv"
    print("Entries signals files: ", PWR_fuzz_name)
    
    PWR_signal = np.loadtxt(PWR_calibration_path + "/" + PWR_exec_name, delimiter=',')
    PWR_cal_traces = np.shape(PWR_signal)[0]
    n_samples = np.shape(PWR_signal)[1]
    
    PWR_input_signal = np.array([[[0 for z in range(n_samples)] for y in range(n_traces)] for x in range(3)])
    PWR_input_signal[0][:PWR_cal_traces] = np.loadtxt(PWR_calibration_path + "/" + PWR_exec_name, delimiter=',')
    PWR_input_signal[1][:PWR_cal_traces] = np.loadtxt(PWR_calibration_path + "/" + PWR_noexec_name, delimiter=',')
    PWR_input_signal[2] = np.loadtxt(PWR_entries_nfs_path + PWR_fuzz_name, delimiter=',')
    
    processing(PWR_input_signal, "PWR")
    return

def bugs_capture(ssh):
    global device, date, n_traces, n_tests, entries, entries_list
    entries_command = "python3.11 /path/to/code/PWR_data_collection.py "  # Anonymized path
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(entries_command + " " + str(device + "_entries_list_" + date + '.csv'))
    print(ssh_stdout.read().decode())
    print(time.time() - start, "seconds to get fault samples")

def calibration(ssh):
    global calibration_command, cal_n_traces, device
    cal_command = "python3.11 /path/to/code/PWR_data_calibration.py "  # Anonymized path
    command = "a1"
    calibration_command = command
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cal_command + command + " " + str(cal_n_traces) + " " + device)
    print(ssh_stdout.read().decode())
    print(time.time() - start, "seconds to get calibration samples")

def entries_list_creation(entries):
    global entries_list, list_nfs_path, date
    input_info = entries.split(',')
    entries_list = []
    for string in input_info:
        repeated_string = [string] * 20
        entries_list.extend(repeated_string)
    
    now = datetime.datetime.now()
    date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
    date_arr = [x.zfill(2) for x in date_arr]
    date = date_arr[2] + "_" + date_arr[1] + "_" + date_arr[0] + "_" + date_arr[3] + "o" + date_arr[4]
    
    with open(list_nfs_path + "_entries_list_" + date + '.csv', 'w') as txt_file:
        for line in entries_list:
            txt_file.write(line + "\n")
    print("Entries list created: ", device, '_entries_list_', date, '.csv')

start1 = time.time()

# SSH CONNECTION TO DEVICE (Commented out for anonymization)
# ssh = paramiko.client.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(device_ip, username="[user]", password="[password]")

start = time.time()
for it in range(n_fuzz):
    it_start = time.time()
    print("ITERATION", it + 1)
    entries = "a0,a1,00,02,03,04,05,06,07,08,09,0a,0b,0c,0d,0e,0f,a3"
    cal_n_traces = 100
    
    with open(list_nfs_path + "_entries_list_" + date + ".csv") as file:
        entries_list = [line.rstrip() for line in file]
    number_entries_list = [mapeo_strings[valor] for valor in entries_list]
    n_traces = np.shape(entries_list)[0]
    n_tests = int(np.sqrt(n_traces))
    operation()
    print(time.time() - it_start, "seconds to perform fuzzing iteration", it + 1)

print(time.time() - start, "seconds to perform operation")
print(time.time() - start1, "seconds to perform whole process")
