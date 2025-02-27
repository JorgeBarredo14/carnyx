# -*- coding: utf-8 -*-
"""
Operation Signal Capture Script
"""

# LIBRARIES
import time
import numpy as np
import serial
import sys
import subprocess
import scipy.stats as stats
from lecroy3 import *

# Configuration Parameters
sample_rate = 1e9
ip_address = "10.205.1.18"
port_list = ['/dev/ttyUSB0']
channel = ['C4']  # Only PWR channel

# Process input arguments
list_name = sys.argv[1]
file_date = list_name[-20:-4]
device = list_name[0:-34]
    
# File Paths
list_path = '/nfs/general/entries_lists/'
power_entries_nfs_path = '/nfs/general/' + device + '/PWR_operation_signals/'

reset_command = 'python3.11 /nfs/general/pinata_reset.py'
 
# List to store hexadecimal numbers
entries_list = []

# Read entries from file
with open(list_path + list_name, "r") as file:
    for line in file:
        # Remove whitespace and convert hex to bytes
        hex_number = line.strip()
        decimal_number = bytes.fromhex(hex_number)
        entries_list.append(decimal_number)

print(f"Capturing signals from commands in: {list_name}")

n_traces = np.shape(entries_list)[0]
       
def operation_execution(ip_address, port_list, channel, n_traces, entries_list):
    """
    Execute operation signal capture for given commands
    """
    global file_date
    global power_entries_nfs_path
    global sample_rate
        
    # Optional: Load pre-registered scope configuration
    load_lecroy_panel_enabled = False

    # Device serial port settings
    ser = serial.Serial(baudrate=115200, timeout=2)
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

    # Calculate number of samples
    n_samples = int(5 * time_div * sample_rate)
    
    # Initialize signal arrays
    power_op_signal = np.array([[0 for z in range(n_samples)] for y in range(n_traces)])
  
    # Prepare Scope trigger
    print(entries_list)
    scope.clearSweeps()
    scope.setTriggerDelay("0")
    scope.setTriggerMode("SINGLE")
    scope.waitLecroy()
    
    start_time = time.time()
    command = entries_list[0]
    ser.write(command)

    # Initial data collection
    channel_out, channel_out_interpreted = scope.getNativeSignalBytes(channel[0], n_samples, False, 3)
    data_output = ser.read(16)
    
    i = 0
    
    while i < len(entries_list):
        try:
            command = entries_list[i]  
            
            # Command processing
            print(f"Command {command.hex()} ({i+1})")
                
            # Set Scope trigger
            scope.setTriggerDelay("0")
            scope.setTriggerMode("SINGLE")
            scope.clearSweeps()
            scope.waitLecroy()
            start_time = time.time()
            ser.write(command)
            
            # Get data from Scope
            channel_out, power_op_signal[i] = scope.getNativeSignalBytes(channel[0], n_samples, False, 3)
            
            # Read serial data
            data_output = ser.read(16)
            print(f"Data output from serial (Device): {data_output}")
            
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
                    raise ValueError(f"Serial cannot be opened. Verify device is listed in {port_list}.")
                    
            print(f"(PWR) Data output from Ethernet (oscilloscope): {len(power_op_signal[i])}")
            
            # Calculate time delta
            delta_t0_time = time.time()
            delta_t0 = delta_t0_time - start_time
            
            # Pearson correlation for consecutive signals
            if i > 0:
                print(f"(PWR) Pearson Correlation: {stats.pearsonr(power_op_signal[i], power_op_signal[i-1])[0]*100}")

            i += 1

        except Exception as ex:
            print("ERROR: ", ex)
            if data_output == b'\x00':
                i += 1
            else:
                scope.disconnect()
                ser.close()
                break
            
    # Save signals to CSV
    np.savetxt(power_entries_nfs_path + device + '_PWR_bugs_' + file_date + '.csv', 
               np.c_[power_op_signal], delimiter=',')
    
    # Clean up
    scope.clearSweeps()
    scope.disconnect()
    ser.close()

# Execute the operation
operation_execution(ip_address, port_list, channel, n_traces, entries_list)