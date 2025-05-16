# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 08:52:24 2023

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
import paramiko

# Import the required packages
import numpy as np
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
import matplotlib.pyplot as plt


sampleRate=20e9
bbb_ip = "10.205.1.101"
cal_command = "./bug_codes/SUT00I"
ip_address = "10.205.1.95"
channel = 'C3'

# commands = [b'\x00', b'\x02', b'\x03', b'\x04', b'\x05', b'\x06', b'\x07', b'\x08', b'\x09', b'\x0A', b'\x0B', b'\x0C', b'\x0D', b'\x0E', b'\x0F', b'\xA3']
# commands = [b'\x08', b'\x09', b'\x0A', b'\x0B', b'\x0C', b'\x0D', b'\x0E', b'\x0F', b'\xA3']

# command = sys.argv[1]
device = 'BBB'
n_traces = 100
# commands = [bytes.fromhex(sys.argv[1])]
# n_traces = int(sys.argv[2])
now = datetime.datetime.now()
date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
date_arr = [x.zfill(2) for x in date_arr]
date = date_arr[2] + "_" + date_arr[1] + "_" + date_arr[0] + "_" + date_arr[3] + "o" + date_arr[4]


#commands = [b'\x00']
# command_OP = b'\x00'

cal_path = '/home/atenea/carnyx/calibration/'

def NOP_execution(scope, channel, n_traces):
    
    global date
    global delta_t0
    global cal_path
    global final_n_samples
    global sampleRate    
 
    # n_samples = int(9*timeDiv*sampleRate) #time when trigger goes off * sampling frequency (maximum is 10*timeDiv*sampleRate?)
    NOP_signal = np.array([[0 for z in range(final_n_samples)] for y in range(n_traces)])
    
    # Main loop
    numTotal = 0
    
    ## Set Scope trigger
    scope.clearSweeps()
    scope.setTriggerMode("AUTO") # INICIAR ADQUISICION
    scope.waitLecroy()
        
    try:
        while numTotal < n_traces:
            print("Idle signal capture, trace", numTotal+1)
            start_time = time.time() # INICIAR TIMER
            while True:
                  if (time.time()-start_time)>=(delta_t0):
                      break # FINALIZAR TIMER
            channel_out, channel_out_interpreted = scope.getNativeSignalBytes(channel,n_samples,False,3) # FIN ADQUISICIÓN      
            #data_output = ser.read(16)
            # print("Data output from serial (BBB):", data_output)
            # NOP_signal[numTotal] = np.abs(channel_out_interpreted)
            # NOP_signal[numTotal] = np.abs(rfft(channel_out_interpreted))
            NOP_signal[numTotal] = channel_out_interpreted
            #print("Data output from Ethernet (oscilloscope):", len(NOP_signal[numTotal])) 
            numTotal+=1

        np.savetxt(cal_path + device + '_noexec_' + date + '.csv', np.c_[NOP_signal], delimiter=',')
        # scope.setTriggerMode("STOP") #Lecroy arm
        scope.clearSweeps() 
        scope.resetLecroy()
            
            # plt.figure()
            # plt.plot(NOP_signal)
            # plt.show()  
        
    except Exception as ex:
        print("ERROR: ", ex)
        scope.disconnect()
        
def OP_execution(scope, channel, n_traces):

    global date
    global delta_t0
    global final_n_samples
    global sampleRate
    global bbb_ip
    global cal_command
    
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('echo 67 > /sys/class/gpio/export')
            
    try:
        ## Set Scope trigger
         scope.clearSweeps()        
         time.sleep(1)
         scope.setTriggerMode("SINGLE") # INICIAR ADQUISICIÓN            
         scope.waitLecroy()
         start_time = time.time()
         ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cal_command)
         # print(ssh_stdout.read().decode())
         
         # epsilon0_time = time.time()
         # epsilon0 = epsilon0_time - start_time
         # print("Epsilon0: ", epsilon0, "seconds")
         ## Get data from Scope
         #voltageGain, voltageOffset, timeInterval, timeOffset = scope.getWaveformDescryption(channel, use2BytesDataFormat=True)
         channel_out, channel_out_interpreted = scope.getNativeSignalBytes(channel,n_samples,False,3) # RECEPCIÓN Y DETENCIÓN ADQUISICIÓN
         # final_n_samples=np.shape(np.abs(rfft(channel_out_interpreted)))[0]
         final_n_samples = n_samples
         OP_signal = np.array([[0 for z in range(int(final_n_samples))] for y in range(n_traces)])
          
         # Main loop
         numTotal = 0
         
         while numTotal < n_traces:
             print("Command {}, trace {}" .format(cal_command,numTotal+1))
             # time.sleep(1)
             ## Set Scope trigger
             scope.setTriggerMode("SINGLE") # INICIAR ADQUISICIÓN
             scope.clearSweeps()
             scope.waitLecroy()
             ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cal_command)
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
     
             # OP_signal[numTotal] = np.abs(channel_out_interpreted)
             # OP_signal[numTotal] = np.abs(rfft(channel_out_interpreted))
             OP_signal[numTotal] = channel_out_interpreted
             print("Data output from Ethernet (oscilloscope):", len(OP_signal[numTotal]))
                  
             delta_t0_time = time.time()
             delta_t0 = delta_t0_time - start_time
             epsilon1 = delta_t0_time-epsilon1_start
             # print("Epsilon1: ", epsilon1, "seconds")
             # print("?t0: " + str(delta_t0) + " seconds")
             
             print("Pearson: ", pearsonr(OP_signal[numTotal], OP_signal[0])[0]*100)
             # if pearsonr(OP_signal[numTotal], OP_signal[0])[0]*100 > 50:
             numTotal+=1
             time.sleep(1)
             # numTotal+=1
             
         np.savetxt(cal_path + device + '_exec_' + date + '.csv', np.c_[OP_signal], delimiter=',')
         # scope.setTriggerMode("STOP") #Lecroy arm
         scope.clearSweeps()
                            
                # plt.figure()
                # plt.plot(OP_signal)
                # plt.show()
                                
    except Exception as ex:
        print("ERROR: ", ex)
        scope.disconnect()

# OPENING SSH CONNECTION TO BEAGLEBONE BLACK
start1 = time.time()
ssh = paramiko.client.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(bbb_ip, username="debian", password="temppwd")
ssh.connect(bbb_ip, username="root", password="root")

print(time.time()-start1, "seconds to start ssh connection")

start = time.time()
# CALIBRATION SIGNALS CAPTURE

# command = "10"

print(time.time()-start, "seconds to get calibration samples")

## If loadLecroyPanelEnable is True, loads a pre-registered config file to avoid setting scope with command lines
## Tips: you can do manual setting of scope and afterwrads save panel config with lecroy3.py, with command: >>python lecroy3.py -s <panelname>.lss
loadLecroyPanelEnabled = False
lecroyPanelFileName = "config/xoodyakHash.lss"

## Init Scope
scope = Lecroy()
scope.connect(IPAdrress = ip_address)
# scope.displayOff() #Disable the plotting of traces during data collection 

## Set/Get Scope parameters
if(loadLecroyPanelEnabled):
    scope.loadLecroyPanelFromFile(lecroyPanelFileName)
    voltDiv = float(scope.getVoltsDiv(channel))
    timeDiv = float(scope.getTimeDiv())
else:
    scope.setTriggerDelay("-23e-6");
    voltDiv = 50e-3#V/div
    timeDiv = 5e-6#s/div
    sampleRate = 10e9#S/s
    # voltDiv = 800e-3#V/div
    # timeDiv = 50e-9#s/div
    # sampleRate = 500e6#S/s
    scope.setSampleRate(sampleRate)
    scope.setVoltsDiv(channel, str(voltDiv))
    scope.setTimeDiv(str(timeDiv))

n_samples = int(1*timeDiv*sampleRate) #time when trigger goes off * sampling frequency (maximum is 10*timeDiv*sampleRate?)
# final_n_samples = int(0.6*n_samples)
# final_n_samples = int(0.5*n_samples)+1

OP_execution(scope, channel, n_traces)
NOP_execution(scope, channel, n_traces)
scope.disconnect()
