# Power Signal Measurement Script

import time
import numpy as np
import serial
import csv
import pandas as pd
import pyvisa as visa
from lecroy3 import *
import scipy.stats as stats
import datetime
import subprocess
import sys

# Configuration Parameters
sample_rate = 1e9
ip_address = "10.205.1.18"
port_list = ['/dev/ttyUSB0']
channel = ['C4']  # Only PWR channel

# Command-line arguments
commands = [bytes.fromhex(sys.argv[1])]
total_traces = int(sys.argv[2])
device = sys.argv[3]

n_traces = total_traces
now = datetime.datetime.now()
date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
date_arr = [x.zfill(2) for x in date_arr]
date = date_arr[2] + "_" + date_arr[1] + "_" + date_arr[0] + "_" + date_arr[3] + "o" + date_arr[4]

power_calibration_path = '/nfs/general/' + device + '/PWR_calibration_signals/'

reset_command = 'python3.11 /nfs/general/device_reset.py'

def no_operation_execution(ip_address, port_list, channel, n_traces):
    global device
    global date
    global delta_t0
    global total_traces
    global n_samples
    global sample_rate    
 
    # Enable loading pre-registered config file (optional)
    load_lecroy_panel_enabled = False
    lecroy_panel_filename = "config/device_config.lss"

    # Device serial port settings
    ser = serial.Serial(baudrate=115200, timeout=1)
    for p in port_list:
        try:
            ser.port = p
            ser.open()
            break
        except(serial.serialutil.SerialException):
            pass
            raise ValueError(f"Serial cannot be opened. Verify device is listed in {port_list}.")

    # Initialize Oscilloscope
    scope = Lecroy()
    scope.connect(IPAdrress = ip_address)

    # Set Scope parameters
    scope.setTriggerDelay("0")
    volts_div = 100e-3  # V/div
    time_div = 5e-6  # s/div
    sample_rate = 1e9  # S/s
    scope.setSampleRate(sample_rate)
    scope.setVoltsDiv(channel[0], str(volts_div))
    scope.setTimeDiv(str(time_div))

    n_samples = int(5 * time_div * sample_rate)
    pwr_no_op_signal = np.array([[0 for z in range(n_samples)] for y in range(total_traces)])
    
    # Main loop
    num_total = 0
    
    # Set Scope trigger
    scope.clearSweeps()
    scope.setTriggerMode("AUTO")
    scope.waitLecroy()
        
    try:
        while num_total < n_traces:
            print(f"Idle signal capture, trace {num_total+1}")
            start_time = time.time()
            while True:
                 if (time.time() - start_time) >= delta_t0:
                     break
            
            channel_out, pwr_no_op_signal[num_total] = scope.getNativeSignalBytes(channel[0], n_samples, False, 3)
        
            data_output = ser.read(16)
            print("Data output from serial (Device):", data_output)
            print("(PWR) Data output from Ethernet (oscilloscope):", len(pwr_no_op_signal[num_total]))
            num_total += 1

        # Save signals to CSV
        np.savetxt(power_calibration_path + device + '_PWR_noexec_' + date + '.csv', 
                   np.c_[pwr_no_op_signal], delimiter=',')
        
        # Clean up
        scope.clearSweeps() 
        scope.resetLecroy()
        scope.disconnect()
        ser.close()
        
    except Exception as ex:
        print("ERROR: ", ex)
        scope.disconnect()
        ser.close()

def operation_execution(ip_address, port_list, channel, n_traces, commands):
    global device    
    global date
    global delta_t0
    global n_samples
    global sample_rate
    
    # Enable loading pre-registered config file (optional)
    load_lecroy_panel_enabled = False
    lecroy_panel_filename = "config/device_config.lss"

    # Device serial port settings
    ser = serial.Serial(baudrate=115200, timeout=1)
    for p in port_list:
        try:
            ser.port = p
            ser.open()
            break
        except(serial.serialutil.SerialException):
            pass
            raise ValueError(f"Serial cannot be opened. Verify device is listed in {port_list}.")

    # Initialize Oscilloscope
    scope = Lecroy()
    scope.connect(IPAdrress = ip_address)

    # Set Scope parameters
    if load_lecroy_panel_enabled:
        scope.loadLecroyPanelFromFile(lecroy_panel_filename)
        volts_div = float(scope.getVoltsDiv(channel))
        time_div = float(scope.getTimeDiv())
    else:
        scope.setTriggerDelay("0")
        volts_div = 100e-3  # V/div
        time_div = 5e-6  # s/div
        sample_rate = 1e9  # S/s
        scope.setSampleRate(sample_rate)
        scope.setVoltsDiv(channel[0], str(volts_div))
        scope.setTimeDiv(str(time_div))

    n_samples = int(5 * time_div * sample_rate)
    pwr_op_signal = np.array([[0 for z in range(n_samples)] for y in range(total_traces)])
            
    try:
        # Set Scope trigger
        scope.clearSweeps()
        for i in range(len(commands)):
            time.sleep(1)
            scope.setTriggerDelay("0")
            scope.setTriggerMode("SINGLE")
            scope.waitLecroy()
            start_time = time.time()
            command = commands[i]
            ser.write(command)
            
            # Get data from Scope
            channel_out, channel_out_interpreted = scope.getNativeSignalBytes(channel[0], n_samples, False, 3)
            data_output = ser.read(16)
                    
            # Main loop
            num_total = 0
            
            while num_total < n_traces:
                print(f"Command {command.hex()}, trace {num_total+1}")
                
                # Set Scope trigger
                scope.setTriggerDelay("0")
                scope.setTriggerMode("SINGLE")
                scope.clearSweeps()
                scope.waitLecroy()
                start_time = time.time()
                ser.write(command)
                
                # Get data from Scope
                channel_out, pwr_op_signal[num_total] = scope.getNativeSignalBytes(channel[0], n_samples, False, 3)
                
                delta_t0_start = time.time()
                data_output = ser.read(16)
                print("Data output from serial (Device):", data_output)
                
                # Reset device if no data received
                if data_output == b'':
                    ser.close()
                    print("Resetting Device...")
                    reset = subprocess.run(reset_command, shell=True, capture_output=True, text=True)
                    time.sleep(1)
                    try:
                        ser.open()
                        time.sleep(1)
                    except(serial.serialutil.SerialException):
                        pass
                        raise ValueError(f"Serial cannot be opened. Verify device is listed in {port_list}.")

                print("(PWR) Data output from Ethernet (oscilloscope):", len(pwr_op_signal[num_total]))
                
                delta_t0_time = time.time()
                delta_t0 = delta_t0_time - start_time
                
                # Calculate correlation with previous signal
                if num_total > 0:
                    print("(PWR) Pearson Correlation: ", 
                          stats.pearsonr(pwr_op_signal[num_total], pwr_op_signal[num_total-1])[0]*100)
                
                num_total += 1
                
            # Save signals to CSV
            np.savetxt(power_calibration_path + device + '_PWR_exec_' + sys.argv[1] + "_" + date + '.csv', 
                       np.c_[pwr_op_signal], delimiter=',')
        
        # Clean up
        scope.clearSweeps()
        scope.disconnect()
        ser.close()
                            
    except Exception as ex:
        print("ERROR: ", ex)
        scope.disconnect()
        ser.close()

# Execute both no-operation and operation functions
operation_execution(ip_address, port_list, channel, n_traces, commands)
no_operation_execution(ip_address, port_list, channel, n_traces)