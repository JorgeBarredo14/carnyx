# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:27:10 2023

@author: jbarredo
"""

# LIBRARIES

import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import pyvisa as visa
from lecroy3 import *
from numpy import loadtxt
from itertools import chain
from itertools import zip_longest
import seaborn as sns
import scipy.integrate as integrate
import scipy.signal  
import scipy.stats as stats
from numpy import exp
from numpy import savetxt
from scipy.stats import pearsonr
import datetime
import random
import sys
import os
import json
import paramiko

# Import the required packages
import numpy as np
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
import matplotlib.pyplot as plt

ip_address = "10.205.1.95"
bbb_ip = "10.205.1.101"
channel = 'C3'

# n_traces = int(sys.argv[1])
# list_name = sys.argv[2]
# device = sys.argv[3]
device = "BBB"
n_traces = 340
list_name = device + '_entries_list_2025_05_14_10o11.csv'
file_date = list_name[-20:-4]
    
cal_entry = './Bugs_RPi4/SUT00I.bin'
cal_path = '/home/atenea/carnyx/calibration/'
entries_path = '/home/atenea/carnyx/'
results_entries_path = '/home/atenea/carnyx/operation/'
list_file = open(entries_path + list_name, 'r')
Lines = list_file.readlines()
 
count = 0
entries_list = [0 for x in range(n_traces)]
entries = [0 for x in range(len(entries_list))]

# Strips the newline character
for line in Lines:
    entries_list[count] = line.strip()
    # entries[count] = entries_list[count]
    entries[count] = entries_list[count]
    count += 1

print("Capturing signals from commands in: ", list_name)
       
def OP_execution(IPAddress, channel, n_traces, entries):

    global cal_entry
    global file_date
    global entries_path
    global delta_t0
    global n_samples
    global sampleRate
    global device
    
    ## If loadLecroyPanelEnable is True, loads a pre-registered config file to avoid setting scope with command lines
    ## Tips: you can do manual setting of scope and afterwrads save panel config with lecroy3.py, with command: >>python lecroy3.py -s <panelname>.lss
    loadLecroyPanelEnabled = False
    lecroyPanelFileName = "config/xoodyakHash.lss"

    ## Init Scope
    scope = Lecroy()
    scope.connect(IPAdrress = IPAddress)
    # scope.displayOff() #Disable the plotting of traces during data collection 

    ## Set/Get Scope parameters
    if(loadLecroyPanelEnabled):
        scope.loadLecroyPanelFromFile(lecroyPanelFileName)
        voltDiv = float(scope.getVoltsDiv(channel))
        timeDiv = float(scope.getTimeDiv())
    else:
        # scope.setTriggerDelay("0");
        voltDiv = 50e-3#V/div
        timeDiv = 5e-6#s/div
        sampleRate = 10e9#S/s
        # voltDiv = 800e-3#V/div
        # timeDiv = 50e-9#s/div
        # sampleRate = 500e6#S/s
        scope.setSampleRate(sampleRate)
        scope.setVoltsDiv(channel, str(voltDiv))
        scope.setTimeDiv(str(timeDiv))


    # exec_name = [filename for filename in os.listdir(cal_path) if filename.startswith("exec_")][0]
    # reference_signal = np.loadtxt(cal_path + "/" + exec_name,delimiter=',')
    
    # final_n_samples = np.shape(reference_signal)[1]
    n_samples = int(1*timeDiv*sampleRate)
    OP_signal = np.array([[0 for z in range(n_samples)] for y in range(n_traces)])
    
    # dif_entries = np.unique(entries).tolist()
    # dif_entries[0] = b'\x00'
    # pearson_comp = np.array([-10 for x in range(np.shape(dif_entries)[0])], dtype=float)
    # prev_entry = np.array([-10 for x in range(np.shape(dif_entries)[0])]) 
    
    start1 = time.time()
    ssh = paramiko.client.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.connect(bbb_ip, username="debian", password="temppwd")
    ssh.connect(bbb_ip, username="root", password="root")

    print(time.time()-start1, "seconds to start ssh connection")
    
    try:
        ## Set Scope trigger
        print(entries)
        scope.clearSweeps()
        # time.sleep(1)        
        scope.setTriggerMode("SINGLE") # INICIAR ADQUISICIÓN            
        scope.waitLecroy()
        start_time = time.time()
        
        entry=entries[0]
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(entry)
        
        # epsilon0_time = time.time()
        # epsilon0 = epsilon0_time - start_time
        # print("Epsilon0: ", epsilon0, "seconds")
        ## Get data from Scope
        #voltageGain, voltageOffset, timeInterval, timeOffset = scope.getWaveformDescryption(channel, use2BytesDataFormat=True)
        channel_out, channel_out_interpreted = scope.getNativeSignalBytes(channel,n_samples,False,3) # RECEPCIÓN Y DETENCIÓN ADQUISICIÓN
        data_output = ssh_stdout.read().decode()
        
        i = 0
        # ind = 0
        # repeat = 0
        # repeat_prev = 0
        # continue_entry = 0
        # reset = 0
        # data_output=b'-1'
               
        # inf_gone_back_tolerance = 0.80
        # sup_gone_back_tolerance = 1.20
        # inf_tolerance = 0.90
        # sup_tolerance = 1.10
        
        while i < len(entries):
            
            # if len(data_output)==0 and reset==0:
            #     reset=1
            #     print("SOFTWARE RESET ({})" .format(i))
            #     # ser.write(bytes.fromhex('c2')) # SOFTWARE RESET
            #     data_output=b'-1'
            # else:
                # time.sleep(1)
                # if reset==1:
                #     # time.sleep(10)
                #     reset=0
                entry=entries[i] 
                # # REPEAT VERIFICATION
                # if repeat==1:
                #     print("Entry {} REPEATED ({})" .format(entry, i))
                #     repeat=0
                # elif repeat_prev==1:
                #     print("Entry {} GONE BACK TO {}" .format(entry, i))
                # else:
                #     print("Entry {} ({})" .format(entry, i))
                print("Entry {} ({})" .format(entry, i+1))    
                # time.sleep(1)
                scope.setTriggerMode("SINGLE") # INICIAR ADQUISICIÓN
                scope.clearSweeps()
                scope.waitLecroy()
                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(entry)
                # print(ssh_stdout.read().decode())
                start_time = time.time()
                epsilon0_time = time.time()
                epsilon0 = epsilon0_time - start_time
                # print("Epsilon0: ", epsilon0, "seconds")
                ## Get data from Scope
                #voltageGain, voltageOffset, timeInterval, timeOffset = scope.getWaveformDescryption(channel, use2BytesDataFormat=True)
                channel_out, channel_out_interpreted = scope.getNativeSignalBytes(channel,n_samples,False,3) # RECEPCIÓN Y DETENCIÓN ADQUISICIÓN
                
                epsilon1_start = time.time()
                data_output = ssh_stdout.read().decode()
                print("Data output from serial (BBB):", data_output)
        
                OP_signal[i] = channel_out_interpreted
                print("(PWR) Data output from Ethernet (oscilloscope):", len(OP_signal[i]))
                
                delta_t0_time = time.time()
                delta_t0 = delta_t0_time - start_time
                epsilon1 = delta_t0_time-epsilon1_start
                # print("Epsilon1: ", epsilon1, "seconds")
                # print("?t0: " + str(delta_t0) + " seconds")
                
                # PEARSON SEQUENCE
                if i>0:
                    print("(PWR) Pearson: ", pearsonr(OP_signal[i], OP_signal[i-1])[0])
                    if pearsonr(OP_signal[i], OP_signal[i-1])[0] >= 0.9999:
                        scope.clearSweeps()
                        scope.waitLecroy()
                        # scope.setTriggerMode("AUTO") # INICIAR ADQUISICIÓN
                        time.sleep(5)
                        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cal_entry)
                        # scope.setTriggerMode("SINGLE") # INICIAR ADQUISICIÓN
                        # i-=1

                # ind = dif_entries.index(entry)              
              
                # if repeat_prev == 1:
                #         pearson_comp[ind] = pearsonr(OP_signal[i], reference_signal[0])[0]   
                #         i = continue_entry
                #         repeat_prev = 0
                # else:    
                #     if pearson_comp[ind] == -10:
                #         pearson_comp[ind] = pearsonr(OP_signal[i], reference_signal[0])[0]
                #         prev_entry[ind] = i
                #         i+=1
                #     # elif (pearsonr(OP_signal[i], reference_signal[0])[0] > sup_gone_back_tolerance*pearson_comp[ind]) or (pearsonr(OP_signal[i], reference_signal[0])[0] < inf_gone_back_tolerance*pearson_comp[ind]):
                #     #     continue_entry = i
                #     #     i = prev_entry[ind]
                #     #     repeat_prev = 1
                #     # elif (pearsonr(OP_signal[i], reference_signal[0])[0] > inf_tolerance*pearson_comp[ind]) and (pearsonr(OP_signal[i], reference_signal[0])[0] < sup_tolerance*pearson_comp[ind]):
                #     #     pearson_comp[ind] = pearsonr(OP_signal[i], reference_signal[0])[0]
                #     #     prev_entry[ind] = i
                #     #     i+=1
                #     else:
                        # pearson_comp[ind] = pearsonr(OP_signal[i], reference_signal[0])[0]
                        # prev_entry[ind] = i
                i+=1
                        # repeat = 1
                    
                # print("Pearson array: ", pearson_comp)
                time.sleep(1)
            
        np.savetxt(results_entries_path + device + '_PWR_bugs_' + file_date + '.csv', np.c_[OP_signal], delimiter=',')
        # scope.setTriggerMode("STOP") #Lecroy arm
        scope.clearSweeps()
        scope.disconnect()
        ssh.close()
                            
        # plt.figure()
        # plt.plot(OP_signal)
        # plt.show()
                                
    except Exception as ex:
        print("ERROR: ", ex)
        scope.disconnect()

OP_execution(ip_address, channel, n_traces, entries)