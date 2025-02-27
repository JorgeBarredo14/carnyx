# Power Consumption Analysis PoC

This repository contains various tools and scripts for performing **Side-Channel Analysis (SCA)** based on power consumption. It includes resources to perform automated power consumption analysis on different platforms.

## Directory Structure

The repository has the following structure:

- **`Piñata/`**: Contains files for performing automated SCA (Side-Channel Analysis) on the **Riscure Piñata** platform.
- **`STM32-F429ZI/`**: Contains files for performing automated SCA on **STM32 NUCLEO-144** boards (STM32-F429ZI).
- **`PoC-STM32-F429ZI_Ethernet/`**: Contains a Python script for Proof of Concept (PoC) of metrics measured on the STM32 NUCLEO-144 board.
- **`requirements.txt`**: Python dependencies required to run the scripts in the repository.

## Overview

This repository provides tools for capturing and analyzing power consumption traces, which can be used for performing **Side-Channel Attacks (SCA)**. The main goal is to provide automated scripts for power consumption measurement and analysis.

### Contents:

1. **Piñata**:  
   This folder contains resources for conducting SCA on the **Riscure Piñata** platform, including scripts and configuration files.

2. **STM32-F429ZI**:  
   This folder is dedicated to performing SCA on the **STM32 NUCLEO-144 (STM32-F429ZI)**. It includes tools for capturing power traces and performing initial analysis.

3. **PoC-STM32-F429ZI_Ethernet**:  
   This folder contains a **Python script** that acts as a Proof of Concept (PoC) for the metrics measured during power consumption analysis on the STM32 platform.

## Installation

To set up the environment for running the scripts, you need to install the required Python dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
