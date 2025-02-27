#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Capture Script for STM32-F429ZI
"""

import datetime
import socket
import time
import numpy as np
from scipy.stats import pearsonr
import subprocess
import matplotlib.pyplot as plt
from tektronik_mso56 import *

# Configuration Parameters
command = 0xa1
comando = str(command)[1:3]
n_traces = int(100)
whole_n_traces = n_traces
device = "STM32-F429ZI"

# Generate timestamp
now = datetime.datetime.now()
date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
date_arr = [x.zfill(2) for x in date_arr]
date = date_arr[0] + "_" + date_arr[1] + "_" + date_arr[2] + "_" + date_arr[3] + "o" + date_arr[4]

# Remote Device Configuration
remote_ip = "10.205.1.97"
remote_port = 6000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Oscilloscope Configuration
osc_ip = "10.205.1.96"
channel = ['CH6']  # Only Power channel

# Scope Settings
voltDiv = 1e-3  # V/div
timeDiv = 15e-6  # s/div
sampleRate = 1.25e9  # S/div
n_samples = int(10 * timeDiv * sampleRate)

# File Paths
calibration_path = '/home/atenea/ATENEA_Calibration_Clustering/input_files/calibration/' 
comando_rst = 'python3.11 /nfs/general/pinata_rst_relee.py'

def NOP_execution(channel, n_traces):
    """
    Capture idle (no operation) signals
    """
    global device, date, delta_t0, n_samples
 
    delta_t0 = 0.9455416202545166
    PWR_NOP_signal = np.array([[0 for z in range(n_samples)] for y in range(whole_n_traces)])
    
    # Initialize Oscilloscope
    scope = Tektronik_MSO56()
    scope.connect(IPAddress=osc_ip)
    
    try:
        numTotal = 0
        
        while numTotal < n_traces:
            print(f"Idle signal capture, trace {numTotal+1}")
            start_time = time.time()
            
            while True:
                if (time.time() - start_time) >= delta_t0:
                    break
                     
            PWR_NOP_signal[numTotal] = scope.getWaveform(channel=channel[0])
            
            print(f"(PWR) Data output from Ethernet (oscilloscope): {len(PWR_NOP_signal[numTotal])}") 
            numTotal += 1

        # Save signals to CSV
        np.savetxt(
            f"{calibration_path}{device}_PWR_noexec_{date}.csv", 
            np.c_[PWR_NOP_signal], 
            delimiter=','
        )
        
    except Exception as ex:
        print("ERROR: ", ex)
        scope.clear()

def OP_execution(channel, n_traces, command):
    """
    Capture operation signals
    """
    global sock, device, date, remote_ip, remote_port, n_samples
    
    PWR_OP_signal = np.array([[0 for z in range(n_samples)] for y in range(whole_n_traces)])
    
    # Initialize Oscilloscope
    scope = Tektronik_MSO56()
    scope.connect(IPAddress=osc_ip)
    scope.setTriggerMode("MANUAL")
            
    try:
        time.sleep(1)
        start_time = time.time()
        data = bytes([command])
        sock.sendto(data, (remote_ip, remote_port))
        
        # Receive initial data
        data_output, addr = sock.recvfrom(1024)
        print(f"Data output from ETH ({device}): {data_output}")
                    
        numTotal = 0
        
        while numTotal < n_traces:
            print(f"Command {command}, trace {numTotal+1}")
            scope.write("*CLS")

            start_time = time.time()
            data = bytes([command])
            sock.sendto(data, (remote_ip, remote_port))
            sock.settimeout(5)
            
            try:
                # Receive data with timeout
                data_output, addr = sock.recvfrom(1024)
                print(f"Data output from serial ({device}): {data_output}")
                
            except socket.timeout:
                print("Timeout receiving data.")
                data_output = b''

            # Capture Power Signal
            PWR_OP_signal[numTotal] = scope.getWaveform(channel=channel[0])
            
            # Reset device if no data received
            if data_output == b'':
                print(f"Resetting {device}...")
                reset = subprocess.run(comando_rst, shell=True, capture_output=True, text=True)
                time.sleep(1)
            
            print(f"(PWR) Data output from Ethernet (oscilloscope): {len(PWR_OP_signal[numTotal])}")
            
            # Compute Pearson correlation between consecutive traces
            if numTotal > 0:
                print(f"(PWR) Pearson: {pearsonr(PWR_OP_signal[numTotal], PWR_OP_signal[numTotal-1])[0]*100}")
            
            numTotal += 1
            
        # Save signals to CSV
        np.savetxt(
            f"{calibration_path}{device}_PWR_exec_{comando}_{date}.csv", 
            np.c_[PWR_OP_signal], 
            delimiter=','
        )
                                
    except Exception as ex:
        print("ERROR: ", ex)

def main():
    """
    Main execution function
    """
    # Perform no-operation signal capture
    NOP_execution(channel, n_traces)
    
    # Perform operation signal capture
    OP_execution(channel, n_traces, command)
    
    # Close socket
    sock.close()

if __name__ == "__main__":
    main()