# CARNYX: A Framework for Vulnerability Detection via Power Consumption Analysis in Low-End Embedded Systems
This repository contains various tools and scripts for performing **Side-Channel Analysis (SCA)** based on power consumption, belonging to CARNYX framework. It includes resources to perform automated power consumption analysis on different platforms.

## Directory Structure
The repository has the following structure:
- **`Piñata/`**: Contains files for performing automated SCA (Side-Channel Analysis) on the **Riscure Piñata** platform.
- **`STM32-F429ZI/`**: Contains files for performing automated SCA on **STM32 NUCLEO-144** boards (STM32-F429ZI).
- **`PoC-PWR_Piñata/`**: Contains a Python script for Proof of Concept (PoC) of metrics measured on the Riscure Piñata board.
- **`PoC-STM32-F429ZI_serial/`**: Contains a Python script for Proof of Concept (PoC) of metrics measured on the STM32 NUCLEO-144 board, using the serial port.
- **`PoC-STM32-F429ZI_Ethernet/`**: Contains a Python script for Proof of Concept (PoC) of metrics measured on the STM32 NUCLEO-144 board, using the Ethernet interface.
- **`requirements.txt`**: Python dependencies required to run the scripts in the repository.

## Overview
This repository provides tools for capturing and analyzing power consumption traces, which can be used for performing **Side-Channel Attacks (SCA)**. The main goal is to provide automated scripts for power consumption measurement and analysis.

### Contents:
1. **Piñata**:  
   This folder contains resources for conducting SCA on the **Riscure Piñata** platform, including scripts and configuration files.
2. **STM32-F429ZI**:  
   This folder is dedicated to performing SCA on the **STM32 NUCLEO-144 (STM32-F429ZI)**. It includes tools for capturing power traces and performing initial analysis.
3. **PoC-PWR_Piñata**:  
   This folder contains a Python script for Proof of Concept (PoC) of metrics measured on the Riscure Piñata board.
4. **PoC-STM32-F429ZI_serial**:  
   This folder contains Python scripts for PoC of metrics measured on the STM32 NUCLEO-144 board using the serial port.
5. **PoC-STM32-F429ZI_Ethernet**:  
   This folder contains Python scripts for PoC of metrics measured on the STM32 NUCLEO-144 board using the Ethernet interface.

## Installation
To set up the environment for running the scripts, you need to install the required Python dependencies. You can do this by running the following command:
```bash
pip install -r requirements.txt
```

## Usage
For detailed usage instructions, please refer to the README files within each specific folder.

## Requirements
- Python 3.6 or higher
- Dependencies listed in requirements.txt
- Hardware: Riscure Piñata board or STM32 NUCLEO-144 (STM32-F429ZI)
- Riscure current probe
- Oscilloscope or power monitoring equipment
