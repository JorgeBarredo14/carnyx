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
# from s_dbw import S_Dbw
from scipy.stats import norm
import paramiko
# from scp import SCPClient
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

# Warning configuration
import warnings
warnings.filterwarnings('ignore')

SHOW_GRAPH = False

device_ip = "X.X.X.X"  # Anonymized IP
local_ip = "Y.Y.Y.Y"   # Anonymized IP
n_fuzz = 1
device = "BBB"

base_path = '/home/atenea/carnyx/'
list_nfs_path = base_path + device  # Anonymized path
PWR_calibration_path = base_path + "calibration/"   # Anonymized path
PWR_entries_nfs_path = base_path + "operation/" # Anonymized path
save_path = base_path               # Anonymized path

# FIXED VARIABLES FOR TESTING
n_files = 3
pca_plot = 0
clust = 0
silh = 0
imp_model = 6

# GLOBAL PARAMETERS
gauss_dict = dict([(1, 0.682), (2, 0.954)])
val = 1
silh_score = []
calinski_score = []
davies_score = []
sdbw_score = []
variance_score = []
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
    './bug_codes/SUT00-SUT0101': 'E0101',
    './bug_codes/SUT00-SUT0102': 'E0102',
    './bug_codes/SUT00-SUT0103': 'E0103',
    './bug_codes/SUT00-SUT0104': 'E0104',
    './bug_codes/SUT00-SUT0105': 'E0105',
    './bug_codes/SUT00-SUT0106': 'E0106',
    './bug_codes/SUT00-SUT0201': 'E0201',
    './bug_codes/SUT00-SUT0202': 'E0202',
    './bug_codes/SUT00-SUT0203': 'E0203',
    './bug_codes/SUT00-SUT0204': 'E0204',
    './bug_codes/SUT00-SUT0205': 'E0205',
    './bug_codes/SUT00-SUT0206': 'E0206',
    './bug_codes/SUT00-SUT0207': 'E0207',
    './bug_codes/SUT00-SUT0208': 'E0208',
    './bug_codes/SUT00-SUT0209': 'E0209',
    './bug_codes/SUT00I': 'SUT00I',
    './bug_codes/SUT00F': 'SUT00F',
}


def generate_beep():
    # Initialize VISA connection
    resource = f'TCPIP0::[Oscilloscope_IP]::INSTR'  # Anonymized IP
    oscilloscope = visa.ResourceManager().open_resource(resource)
    try:
        # Send SCPI command to generate a beep
        oscilloscope.write(":SYST:BEEP")
        time.sleep(0.5)
    finally:
        oscilloscope.close()

def get_class_name(obj):
    return type(obj).__name__

def robust_covariance(signal):  # Stage 3
    detector = EllipticEnvelope(contamination=0.1, assume_centered=True)
    return detector.fit(signal).predict(signal)

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

def robust_covariance_procedure(outlied_signal, kind):
    global n_files, n_traces, n_samples
    robust_samples = np.array([[[0 for z in range(np.shape(outlied_signal)[2])] for y in range(np.shape(outlied_signal)[1])] 
                              for x in range(np.shape(outlied_signal)[0])])
    
    for test in range(np.shape(outlied_signal)[0]):
        ceros_por_fila = np.count_nonzero(outlied_signal[test] == 0, axis=1)
        indice_25000_ceros = np.where(ceros_por_fila == n_samples)[0]
        for i in range(np.shape(outlied_signal)[1]):
            if ceros_por_fila[i] == 0:
                continue
            elif test == 2 or (test < 2 and i not in indice_25000_ceros):
                robust_samples[test][i] = robust_covariance(np.transpose(outlied_signal[test][i, np.newaxis]))
    return robust_samples

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
    
    varianzas = np.array([np.nan for x in range(n_traces)])
    stop_index = n_traces
    for i in range(1, 2):
        print("Searching stop index for sliding window={}...".format(int(i)))
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
            
            silh_score[index - 1] = silhouette_score(data, db.labels_)
            calinski_score[index - 1] = calinski_harabasz_score(data, db.labels_)
            davies_score[index - 1] = davies_bouldin_score(data, db.labels_)
            # sdbw_score[index - 1] = S_Dbw(data, db.labels_, method='Halkidi', metric='euclidean')
        
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
    valid_bug_labels = list(mapeo_bugs.values())
    for i in range(len(etiquetas_unicas_true)):
        label = np.argmax(confus_matrix[i])
        rep = len(np.where(confus_matrix[i] == np.max(confus_matrix[i]))[0])
        for label in range(rep):
            index = np.where(confus_matrix[i] == np.max(confus_matrix[i]))[0][label]
            if isinstance(etiquetas_unicas_pred[index], np.int64):
                y_pred_changed[y_pred == etiquetas_unicas_pred[index]] = etiquetas_unicas_true[i]
    # Assign unassociated predicted labels to "UAN"
    for i in range(len(y_pred_changed)):
        if y_pred_changed[i] not in valid_bug_labels:
            y_pred_changed[i] = 'UAN'
    etiquetas_unicas_pred = np.unique(y_pred_changed[2 * cal_n_traces:])
    valores_nuevos = np.setdiff1d(etiquetas_unicas_true, etiquetas_unicas_pred)
    etiquetas_unicas_pred = np.union1d(etiquetas_unicas_pred, valores_nuevos)
    etiquetas_unicas_pred = np.roll(etiquetas_unicas_pred, -np.where(etiquetas_unicas_pred == 'E0101')[0])
    confus_matrix = np.zeros((len(etiquetas_unicas_true), len(etiquetas_unicas_pred)))
    for true_label, pred_label in zip(bugs_entries_list, y_pred_changed[2 * cal_n_traces:]):
        confus_matrix[np.where(etiquetas_unicas_true == true_label)[0][0],
                      np.where(etiquetas_unicas_pred == pred_label)[0][0]] += 1
    
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    
    # Visualizar la matriz
    im = ax.imshow(confus_matrix, cmap=plt.get_cmap('GnBu'), interpolation='nearest')
    
    # Añadir texto a las celdas
    for i in range(len(etiquetas_unicas_true)):
        for j in range(len(etiquetas_unicas_pred)):
            if int(confus_matrix[i, j]) >= 10:
                ax.text(j, i, str(int(confus_matrix[i, j])), ha="center", va="center", color="white", fontsize=28)
            else:
                ax.text(j, i, str(int(confus_matrix[i, j])), ha="center", va="center", color="black", fontsize=28)
    
    # Configurar título y etiquetas
    ax.set_title('Confusion Matrix (PWR), {},\nTime Domain in {}'.format(get_class_name(db), device), fontsize=28)
    ax.set_xticks(np.arange(len(etiquetas_unicas_pred)))
    ax.set_yticks(np.arange(len(etiquetas_unicas_true)))
    ax.set_xticklabels(etiquetas_unicas_pred, rotation=90, fontsize=28)
    ax.set_yticklabels(etiquetas_unicas_true, fontsize=28)
    ax.set_xlabel('Assigned Labels', fontsize=28)
    ax.set_ylabel('Actual Labels', rotation=90, verticalalignment='center', fontsize=28)
    
    # Configurar una barra de color con la misma anchura pero menor altura
    # Utilizamos el parámetro 'shrink' para hacerla más corta (verticalmente)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)  # shrink=0.5 hace que sea la mitad de alta, pad=0.01 la acerca al gráfico
    cbar.ax.tick_params(labelsize=25)
    
    # Guardar y mostrar la figura
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

def calculate_metrics(y_pred_changed, bugs_entries_list):
    """
    Calcula métricas de evaluación para tres niveles de clasificación:
    1. Anomaly Detection: Error vs No Error
    2. Arithmetic vs Memory Error Detection
    3. Specific Error Detection (error específico)
    
    Args:
        y_pred_changed: Array con las predicciones (después del mapeo)
        bugs_entries_list: Lista con las etiquetas verdaderas
    
    Returns:
        Un diccionario con las métricas para cada nivel de clasificación
    """
    from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, f1_score, confusion_matrix
    import numpy as np
    
    # Asegurarnos de que estamos trabajando solo con los datos de prueba (no calibración)
    test_predictions = []
    for i in range(2 * cal_n_traces, 2 * cal_n_traces + len(bugs_entries_list)):
        if i < len(y_pred_changed):
            test_predictions.append(y_pred_changed[i])
        else:
            test_predictions.append("UAN")
    
    test_true_labels = bugs_entries_list
    
    # Verificar que los arrays tienen la misma longitud
    min_length = min(len(test_predictions), len(test_true_labels))
    test_predictions = test_predictions[:min_length]
    test_true_labels = test_true_labels[:min_length]
    
    # Diccionario para almacenar resultados
    results = {
        "anomaly_detection": {},
        "error_type_detection": {},
        "specific_error_detection": {}
    }
    
    # ============ NIVEL 1: Anomaly Detection (Error vs No Error) ============
    # Mapear etiquetas a binario: 'Error' o 'No Error'
    y_true_anomaly = []
    y_pred_anomaly = []
    
    for i in range(len(test_true_labels)):
        true_label = test_true_labels[i]
        pred_label = test_predictions[i]
        
        # Mapeo de etiquetas verdaderas
        if true_label in ['SUT00I', 'SUT00F']:
            y_true_anomaly.append('No Error')
        else:
            y_true_anomaly.append('Error')
            
        # Mapeo de predicciones (UAN ahora se considera como Error)
        if pred_label in ['SUT00I', 'SUT00F']:
            y_pred_anomaly.append('No Error')
        else:
            # Ahora UAN se considera como Error
            y_pred_anomaly.append('Error')
    
    # Ya no necesitamos filtrar UAN porque ahora se considera como Error
    filtered_y_true = y_true_anomaly
    filtered_y_pred = y_pred_anomaly
    
    if len(set(filtered_y_true)) > 1 and len(set(filtered_y_pred)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_y_true, filtered_y_pred, average='weighted', zero_division=0)
        micro_f1 = f1_score(filtered_y_true, filtered_y_pred, average='micro', zero_division=0)
        mcc = matthews_corrcoef(filtered_y_true, filtered_y_pred)
        
        # Crear matriz de confusión para este nivel
        cm_anomaly = confusion_matrix(filtered_y_true, filtered_y_pred, labels=['Error', 'No Error'])
        
        results["anomaly_detection"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "micro_f1": micro_f1,
            "mcc": mcc,
            "confusion_matrix": cm_anomaly,
            "labels": ['Error', 'No Error']
        }
    else:
        results["anomaly_detection"] = {"error": "No hay suficiente variedad en las etiquetas para calcular métricas"}
    
    # ============ NIVEL 2: Arithmetic vs Memory Error ============
    # Mapear etiquetas a: 'Arithmetic', 'Memory', 'No Error'
    y_true_error_type = []
    y_pred_error_type = []
    
    for i in range(len(test_true_labels)):
        true_label = test_true_labels[i]
        pred_label = test_predictions[i]
        
        # Mapeo de etiquetas verdaderas
        if true_label in ['SUT00I', 'SUT00F']:
            y_true_error_type.append('No Error')
        elif true_label in ['E0101', 'E0102', 'E0103', 'E0104', 'E0105', 'E0106']:
            y_true_error_type.append('Arithmetic')
        elif true_label in ['E0201', 'E0202', 'E0203', 'E0204', 'E0205', 'E0206', 'E0207', 'E0208', 'E0209']:
            y_true_error_type.append('Memory')
        else:
            y_true_error_type.append('Other')  # Por si hay alguna etiqueta no reconocida
            
        # Mapeo de predicciones (UAN se trata como "Other" en este nivel)
        if pred_label in ['SUT00I', 'SUT00F']:
            y_pred_error_type.append('No Error')
        elif pred_label in ['E0101', 'E0102', 'E0103', 'E0104', 'E0105', 'E0106']:
            y_pred_error_type.append('Arithmetic')
        elif pred_label in ['E0201', 'E0202', 'E0203', 'E0204', 'E0205', 'E0206', 'E0207', 'E0208', 'E0209']:
            y_pred_error_type.append('Memory')
        else:
            y_pred_error_type.append('Other')  # UAN y otras etiquetas
    
    # Filtrar casos "Other" para este nivel
    filtered_indices = []
    for i in range(len(y_pred_error_type)):
        if y_pred_error_type[i] != 'Other' and y_true_error_type[i] != 'Other':
            filtered_indices.append(i)
    
    filtered_y_true = []
    filtered_y_pred = []
    for i in filtered_indices:
        filtered_y_true.append(y_true_error_type[i])
        filtered_y_pred.append(y_pred_error_type[i])
    
    if len(set(filtered_y_true)) > 1 and len(set(filtered_y_pred)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_y_true, filtered_y_pred, average='weighted', zero_division=0)
        micro_f1 = f1_score(filtered_y_true, filtered_y_pred, average='micro', zero_division=0)
        mcc = matthews_corrcoef(filtered_y_true, filtered_y_pred)
        
        # Crear matriz de confusión para este nivel
        unique_labels = sorted(list(set(filtered_y_true) | set(filtered_y_pred)))
        cm_error_type = confusion_matrix(filtered_y_true, filtered_y_pred, labels=unique_labels)
        
        results["error_type_detection"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "micro_f1": micro_f1,
            "mcc": mcc,
            "confusion_matrix": cm_error_type,
            "labels": unique_labels
        }
    else:
        results["error_type_detection"] = {"error": "No hay suficiente variedad en las etiquetas para calcular métricas"}
    
    # ============ NIVEL 3: Detección de error específico ============
    # Para el nivel de error específico, tratamos UAN como 'Unknown' en las predicciones
    
    # Preparar las etiquetas y predicciones
    filtered_y_true = test_true_labels
    filtered_y_pred = []
    for pred in test_predictions:
        if pred == 'UAN':
            filtered_y_pred.append('Unknown')  # Renombramos UAN a Unknown para mayor claridad
        else:
            filtered_y_pred.append(pred)
    
    if len(set(filtered_y_true)) > 1 and len(set(filtered_y_pred)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_y_true, filtered_y_pred, average='weighted', zero_division=0)
        micro_f1 = f1_score(filtered_y_true, filtered_y_pred, average='micro', zero_division=0)
        mcc = matthews_corrcoef(filtered_y_true, filtered_y_pred)
        
        # Para la matriz de confusión, usamos solo las etiquetas válidas que aparecen en los datos
        unique_labels = sorted(list(set(filtered_y_true) | set(filtered_y_pred)))
        cm_specific = confusion_matrix(filtered_y_true, filtered_y_pred, labels=unique_labels)
        
        results["specific_error_detection"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "micro_f1": micro_f1,
            "mcc": mcc,
            "confusion_matrix": cm_specific,
            "labels": unique_labels
        }
    else:
        results["specific_error_detection"] = {"error": "No hay suficiente variedad en las etiquetas para calcular métricas"}
    
    # Visualización de los resultados
    print("\n=== RESULTADOS DE EVALUACIÓN ===\n")
    
    categories = [
        ("1. Anomaly Detection (Error vs No Error)", "anomaly_detection"),
        ("2. Error Type Detection (Arithmetic vs Memory)", "error_type_detection"),
        ("3. Specific Error Detection", "specific_error_detection")
    ]
    
    for cat_idx in range(len(categories)):
        title = categories[cat_idx][0]
        key = categories[cat_idx][1]
        
        print(f"\n{title}")
        print("="*50)
        
        if "error" in results[key]:
            print(results[key]["error"])
            continue
            
        print(f"Precision: {results[key]['precision']:.4f}")
        print(f"Recall:    {results[key]['recall']:.4f}")
        print(f"F1 Score:  {results[key]['f1']:.4f}")
        print(f"Micro-F1:  {results[key]['micro_f1']:.4f}")
        print(f"MCC:       {results[key]['mcc']:.4f}")
        
        print("\nMatriz de Confusión:")
        cm = results[key]["confusion_matrix"]
        labels = results[key]["labels"]
        
        # Imprimir la matriz de confusión de manera legible
        header = "   "
        for l_idx in range(len(labels)):
            header += f"{labels[l_idx]:>7}"
        print(header)
        
        for i in range(len(labels)):
            row = f"{labels[i]:3} "
            for j in range(len(labels)):
                row += f"{cm[i, j]:7d}"
            print(row)
    
    # Visualizar las matrices de confusión
    if "confusion_matrix" in results["anomaly_detection"]:
        plt.figure(figsize=(18, 6))
        
        # Anomaly Detection
        plt.subplot(1, 3, 1)
        cm = results["anomaly_detection"]["confusion_matrix"]
        labels = results["anomaly_detection"]["labels"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Anomaly Detection\n(Error vs No Error)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        # Error Type Detection
        if "confusion_matrix" in results["error_type_detection"]:
            plt.subplot(1, 3, 2)
            cm = results["error_type_detection"]["confusion_matrix"]
            labels = results["error_type_detection"]["labels"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.title("Error Type Detection\n(Arithmetic vs Memory)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
        
        # Specific Error Detection - Solo mostramos un resumen visual debido a la potencial cantidad de clases
        if "confusion_matrix" in results["specific_error_detection"]:
            plt.subplot(1, 3, 3)
            cm = results["specific_error_detection"]["confusion_matrix"]
            # Calcular la diagonal y los valores fuera de la diagonal para un resumen visual
            diag = np.diag(cm)
            off_diag = cm.sum() - diag.sum()
            summary_cm = np.array([[diag.sum(), off_diag], [0, 0]])  # Solo para visualización
            sns.heatmap(summary_cm[:1, :], annot=True, fmt="d", cmap="Blues", 
                       xticklabels=["Correct", "Incorrect"], yticklabels=[""])
            plt.title(f"Specific Error Detection\n{diag.sum()} correct, {off_diag} incorrect")
            plt.xlabel("Classification")
        
        plt.tight_layout()
        plt.savefig(save_path + device + "_entries_list_" + date + "_METRICS_SUMMARY.png", bbox_inches='tight', dpi=100)
        plt.show()
        
    return results

def generate_latex_table(metrics_results, device_name, output_path):
    """
    Genera una tabla LaTeX con los resultados de las métricas.
    
    Args:
        metrics_results: Diccionario con los resultados de las métricas
        device_name: Nombre del dispositivo para el título de la tabla
        output_path: Ruta donde guardar el archivo LaTeX
    """
    latex_code = []
    
    # Encabezado de la tabla
    latex_code.append("\\begin{table}[!ht]")
    latex_code.append("\\small")
    latex_code.append("\\centering")
    latex_code.append(f"\\caption{{Clustering Results for {device_name}\\looseness=-1}}")
    latex_code.append(f"\\label{{tab:clust_results_{device_name.lower().replace('-', '_').replace(' ', '_')}}}")
    latex_code.append("\\begin{adjustbox}{max width=\\columnwidth}")
    latex_code.append("\\begin{tabular}{@{}lrrrcc@{}}")
    latex_code.append("\\toprule")
    
    # Sección 1: Anomaly Detection
    latex_code.append("Anomaly Detection & \\begin{tabular}[c]{@{}r@{}}Recall\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}r@{}}Precision\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}r@{}}$F_1$\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Micro-$F_1$\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}c@{}}MCC\\\\ (\\%) \\end{tabular} \\\\ \\midrule")
    
    anomaly_results = metrics_results["anomaly_detection"]
    if "error" not in anomaly_results:
        # Obtener métricas globales
        micro_f1 = anomaly_results["micro_f1"] * 100
        mcc = anomaly_results["mcc"] * 100
        
        # Obtener métricas por clase
        cm = anomaly_results["confusion_matrix"]
        labels = anomaly_results["labels"]
        
        # Calcular precision, recall y F1 por clase
        per_class_metrics = {}
        for i in range(len(labels)):
            label = labels[i]
            # Para cada clase, calculamos:
            # Recall = TP / (TP + FN)
            tp = cm[i, i]
            fn = sum(cm[i, :]) - tp
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            
            # Precision = TP / (TP + FP)
            fp = sum(cm[:, i]) - tp
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[label] = {
                "recall": recall,
                "precision": precision,
                "f1": f1
            }
        
        # Agregar filas para cada clase (Error y No Error)
        # Formato para la primera clase normal y para la segunda clase con multirow
        classes = list(per_class_metrics.keys())
        
        # Primera clase
        latex_code.append(f"{classes[0]} & {per_class_metrics[classes[0]]['recall']:.2f} & {per_class_metrics[classes[0]]['precision']:.2f} & {per_class_metrics[classes[0]]['f1']:.2f} \\\\ %\\cmidrule(r){{1-4}}")
        
        # Segunda clase con multirow para micro-F1 y MCC
        latex_code.append(f"{classes[1]} & {per_class_metrics[classes[1]]['recall']:.2f} & {per_class_metrics[classes[1]]['precision']:.2f} & {per_class_metrics[classes[1]]['f1']:.2f} & \\multirow{{-2}}{{*}}{{{micro_f1:.2f}}} & \\multirow{{-2}}{{*}}{{{mcc:.2f}}} \\\\ \\midrule \\multicolumn{{1}}{{c}}{{}} \\\\ \\midrule")
    else:
        latex_code.append("No sufficient data & - & - & - & - & - \\\\ \\midrule \\multicolumn{1}{c}{} \\\\ \\midrule")
    
    # Sección 2: Arithmetic vs Memory Error Detection
    latex_code.append("\\begin{tabular}[c]{@{}l@{}}Arithmetic vs. Memory\\\\ Error Detection\\end{tabular} & \\begin{tabular}[c]{@{}r@{}}Recall\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}r@{}}Precision\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}r@{}}$F_1$\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Micro-$F_1$\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}c@{}}MCC\\\\ (\\%) \\end{tabular} \\\\ \\midrule")
    
    error_type_results = metrics_results["error_type_detection"]
    if "error" not in error_type_results:
        # Obtener métricas globales
        micro_f1 = error_type_results["micro_f1"] * 100
        mcc = error_type_results["mcc"] * 100
        
        # Calcular métricas por clase
        cm = error_type_results["confusion_matrix"]
        labels = error_type_results["labels"]
        
        # Encontrar las clases "Arithmetic Error" y "Memory Error" si existen
        arith_idx = -1
        memory_idx = -1
        for i in range(len(labels)):
            if labels[i] == "Arithmetic":
                arith_idx = i
            elif labels[i] == "Memory":
                memory_idx = i
        
        per_class_metrics = {}
        for i in range(len(labels)):
            label = labels[i]
            # Calcular métricas por clase
            tp = cm[i, i]
            fn = sum(cm[i, :]) - tp
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            
            fp = sum(cm[:, i]) - tp
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[label] = {
                "recall": recall,
                "precision": precision,
                "f1": f1
            }
        
        # Agregar las filas solo para Arithmetic Error y Memory Error
        if arith_idx >= 0 and memory_idx >= 0:
            # Arithmetic error
            latex_code.append(f"Arithmetic Error & {per_class_metrics['Arithmetic']['recall']:.2f} & {per_class_metrics['Arithmetic']['precision']:.2f} & {per_class_metrics['Arithmetic']['f1']:.2f} \\\\ %\\cmidrule(r){{1-4}}")
            
            # Memory error con multirow
            latex_code.append(f"Memory Error & {per_class_metrics['Memory']['recall']:.2f} & {per_class_metrics['Memory']['precision']:.2f} & {per_class_metrics['Memory']['f1']:.2f} & \\multirow{{-2}}{{*}}{{{micro_f1:.2f}}} & \\multirow{{-2}}{{*}}{{{mcc:.2f}}} \\\\ \\midrule \\multicolumn{{1}}{{c}}{{}} \\\\ \\midrule")
        elif "No Error" in per_class_metrics:
            # Si no están las clases específicas pero sí No Error, mostramos solo No Error
            latex_code.append(f"No Error & {per_class_metrics['No Error']['recall']:.2f} & {per_class_metrics['No Error']['precision']:.2f} & {per_class_metrics['No Error']['f1']:.2f} & {micro_f1:.2f} & {mcc:.2f} \\\\ \\midrule \\multicolumn{{1}}{{c}}{{}} \\\\ \\midrule")
        else:
            latex_code.append("No data & - & - & - & - & - \\\\ \\midrule \\multicolumn{1}{c}{} \\\\ \\midrule")
    else:
        latex_code.append("No sufficient data & - & - & - & - & - \\\\ \\midrule \\multicolumn{1}{c}{} \\\\ \\midrule")
    
    # Sección 3: Specific Error Detection
    latex_code.append("\\begin{tabular}[c]{@{}l@{}}Specific Error\\\\ Detection \\end{tabular} & \\begin{tabular}[c]{@{}r@{}}Recall\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}r@{}}Precision\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}r@{}}$F_1$\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Micro-$F_1$\\\\ (\\%) \\end{tabular} & \\begin{tabular}[c]{@{}c@{}}MCC\\\\ (\\%) \\end{tabular} \\\\ \\midrule")
    
    specific_results = metrics_results["specific_error_detection"]
    if "error" not in specific_results:
        # Obtener métricas globales
        micro_f1 = specific_results["micro_f1"] * 100
        mcc = specific_results["mcc"] * 100
        
        # Calcular métricas por clase
        cm = specific_results["confusion_matrix"]
        labels = specific_results["labels"]
        
        # Todas las posibles clases específicas (añadimos incluso las que no aparecen)
        all_specific_errors = [
            'E0101', 'E0102', 'E0103', 'E0104', 'E0105', 'E0106',
            'E0201', 'E0202', 'E0203', 'E0204', 'E0205', 'E0206', 
            'E0207', 'E0208', 'E0209', 'E0210'
        ]
        
        per_class_metrics = {}
        # Primero inicializamos todas con ceros
        for error in all_specific_errors:
            per_class_metrics[error] = {
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0
            }
        
        # Ahora calculamos para las que realmente aparecen
        for i in range(len(labels)):
            label = labels[i]
            if label in all_specific_errors:
                tp = cm[i, i]
                fn = sum(cm[i, :]) - tp
                recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                
                fp = sum(cm[:, i]) - tp
                precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_metrics[label] = {
                    "recall": recall,
                    "precision": precision,
                    "f1": f1
                }
        
        # Agregar filas para cada error específico
        for i in range(len(all_specific_errors)):
            error = all_specific_errors[i]
            if i < len(all_specific_errors) - 1:
                # Para todas menos la última, formato normal
                latex_code.append(f"{error} & {per_class_metrics[error]['recall']:.2f} & {per_class_metrics[error]['precision']:.2f} & {per_class_metrics[error]['f1']:.2f} \\\\ %\\cmidrule(r){{1-4}}")
            else:
                # Para la última, agregar multirow con micro-F1 y MCC
                latex_code.append(f"{error} & {per_class_metrics[error]['recall']:.2f} & {per_class_metrics[error]['precision']:.2f} & {per_class_metrics[error]['f1']:.2f} & \\multirow{{-{len(all_specific_errors)}}}{{*}}{{{micro_f1:.2f}}} & \\multirow{{-{len(all_specific_errors)}}}{{*}}{{{mcc:.2f}}} \\\\ \\bottomrule")
    else:
        latex_code.append("No sufficient data & - & - & - & - & - \\\\ \\bottomrule")
    
    # Cerrar la tabla
    latex_code.append("\\end{tabular}")
    latex_code.append("\\end{adjustbox}")
    latex_code.append("\\end{table}")
    
    # Unir todo y guardar en el archivo
    latex_table = "\n".join(latex_code)
    
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"Tabla LaTeX guardada en: {output_path}")
    
    return latex_table

def processing(input_signal, kind):
    globals()[f"{kind}_outlied_signal"] = outlier_detection(input_signal, kind)
    globals()[f"{kind}_robust_samples"] = robust_covariance_procedure(globals()[f"{kind}_outlied_signal"], kind)
    globals()[f"{kind}_pca_samples"] = pca_technique_application(globals()[f"{kind}_robust_samples"], kind)
    globals()[f"{kind}_etiquetas_unicas_true"], globals()[f"{kind}_etiquetas_unicas_pred"] = clustering_procedure(globals()[f"{kind}_pca_samples"], kind)

def operation():
    global PWR_calibration_path, PWR_entries_nfs_path, silh, pca_plot, imp_model, n_traces, n_samples, date, cal_traces, device, PWR_input_signal
    
    PWR_exec_name = [filename for filename in os.listdir(PWR_calibration_path) if filename.startswith(device + "_exec_")][0]
    PWR_noexec_name = [filename for filename in os.listdir(PWR_calibration_path) if filename.startswith(device + "_noexec")][0]
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

start = time.time()
for it in range(n_fuzz):
    it_start = time.time()
    print("ITERATION", it + 1)
    # entries = "a0,a1,00,02,03,04,05,06,07,08,09,0a,0b,0c,0d,0e,0f,a3"
    cal_n_traces = 100
    # calibration(ssh)
    # bugs_capture(ssh) # BUGS CAPTURING
    
    # now = datetime.datetime.now()
    # date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
    # date_arr = [x.zfill(2) for x in date_arr]
    # date = date_arr[2] + "_" + date_arr[1] + "_" + date_arr[0] + "_" + date_arr[3] + "o" + date_arr[4]
    
    date = "2025_05_14_10o11"

    with open(list_nfs_path + "_entries_list_" + date + ".csv") as file:
        entries_list = [line.rstrip() for line in file]
    # number_entries_list = [mapeo_strings[valor] for valor in entries_list]
    n_traces = np.shape(entries_list)[0]
    n_tests = int(np.sqrt(n_traces))
    operation()
    print(time.time() - it_start, "seconds to perform fuzzing iteration", it + 1)
    # metrics_results = calculate_metrics(y_pred_changed, bugs_entries_list)
    # latex_table = generate_latex_table(
    #     metrics_results,
    #     device.upper(),  # Nombre del dispositivo (BBB ? "BBB")
    #     save_path + device + "_entries_list_" + date + "_METRICS_TABLE.tex"
    # )


print(time.time() - start, "seconds to perform operation")
print(time.time() - start1, "seconds to perform whole process")
