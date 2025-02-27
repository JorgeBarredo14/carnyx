#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:36:14 2024
"""

import socket
import threading
import time
import numpy as np
from scipy.stats import pearsonr
import datetime
import subprocess
from tektronik_mso56 import *
import sys
import paramiko

# Command line inputs
# n_traces = int(sys.argv[1])
# list_name = sys.argv[1]
list_name = "STM32-F429ZI_entries_list_2024_02_20_10o32.csv"
file_date = list_name[-20:-4]
device = list_name[0:-34]

# OSCILLOSCOPE CONFIGURATION

# Oscilloscope IP address and monitored channels
osc_ip = "10.205.1.96"
channel = ['CH2', 'CH6']

# REMOTE DEVICE CONFIGURATION

# Remote device IP address and port
remote_ip = "10.205.1.97"
remote_port = 6000
# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

## Initialize Scope
## Set/Get Scope parameters
voltDiv = 1e-3  # V/div

# Open connection with the oscilloscope by its IP address
scope = Tektronik_MSO56()
scope.connect(IPAddress=osc_ip)

timeDiv = 15e-6  # s/div
voltDiv = 200e-3  # V/div

sampleRate = 1.25e9  # S/div
scope.setSampleRate(sampleRate)

n_samples = int(10 * timeDiv * sampleRate)

calibration_path = '/home/user/Calibration_Clustering/input_files/calibration/'

comando_rst = 'python /nfs/general/reset_script.py'
rst_hex = 0xa5

list_path = '/home/user/Calibration_Clustering/input_files/entries_lists/'
entries_nfs_path = '/home/user/Calibration_Clustering/input_files/entries_results/'

# RASPBERRY PI RESET CONFIGURATION
raspi_ip = "10.205.1.66"
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(raspi_ip, port=22, username="pi", password="raspberry")
comando_rst = 'python /home/pi/reset_script.py'

# List to store hexadecimal numbers as integers
entries_list = []

# Open the file and read line by line
with open(list_path + list_name, "r") as file:
    for line in file:
        # Remove newlines and whitespace around the number
        hex_number = line.strip()

        # Convert hexadecimal number to integer
        decimal_number = bytes.fromhex(hex_number)

        # Add the integer to the list
        entries_list.append(decimal_number)

print("Capturing signals from commands in: ", list_name)

n_traces = np.shape(entries_list)[0]

def OP_execution(channel, n_traces, entries_list):

    global device
    global file_date
    global entries_nfs_path
    global delta_t0
    global n_samples
    global sampleRate
    global comando_rst

    PWR_OP_signal = np.array([[0 for z in range(n_samples)] for y in range(n_traces)])

    ## Set Scope trigger
    print(entries_list)
    start_time = time.time()
    command = entries_list[0]
    data = command
    sock.sendto(data, (remote_ip, remote_port))
    data_output, addr = sock.recvfrom(1024)
    print("Data output from serial ({}): {}".format(device, data_output))

    i = 0

    while i < len(entries_list):

        try:

            command = entries_list[i]

            # REPEAT VERIFICATION
            print("Command {} ({})".format(command.hex(), i + 1))

            start_time = time.time()

            data = command
            sock.sendto(data, (remote_ip, remote_port))
            sock.settimeout(5)

            try:
                data_output, addr = sock.recvfrom(1024)
                print("Data output from serial ({}): {}".format(device, data_output))

                epsilon0_time = time.time()
                epsilon0 = epsilon0_time - start_time
                PWR_OP_signal[i] = scope.getWaveform(channel=channel[1])

                epsilon1_start = time.time()

            except socket.timeout:
                print("Timeout is over. Resetting {}".format(device))

                PWR_OP_signal[i] = scope.getWaveform(channel=channel[1])

                stdin, stdout, stderr = client.exec_command(comando_rst)
                time.sleep(10)

            print("(PWR) Data output from Ethernet (oscilloscope):", len(PWR_OP_signal[i]))

            delta_t0_time = time.time()
            delta_t0 = delta_t0_time - start_time
            epsilon1 = delta_t0_time - epsilon1_start

            if i > 0:
                print("(PWR) Pearson: ", pearsonr(PWR_OP_signal[i], PWR_OP_signal[i - 1])[0] * 100)

            i += 1

        except Exception as ex:
            print("ERROR: ", ex)
            if data_output == b'\x00':
                i += 1
            else:
                continue

    np.savetxt(entries_nfs_path + device + '_PWR_bugs_' + file_date + '.csv', np.c_[PWR_OP_signal], delimiter=',')
    client.close()

OP_execution(channel, n_traces, entries_list)
