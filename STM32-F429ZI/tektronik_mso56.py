import sys
import numpy as np
import struct
import time

import pyvisa as visa #import the visa library

DEBUG_MODE = False

class Tektronik_MSO56():

    def __init__(self):
        print("[*] Tektronik MSO56 SETUP")
        self.rm = visa.ResourceManager()
        self._scope = None
    
    def __del__(self):
        self.disconnect()
        
    def connect(self,IPAddress = "10.205.1.96"):
        command = "TCPIP0::" + IPAddress + "::INSTR"
        print("[0] " + command)
        self._scope = self.rm.open_resource(command)
        self._scope.timeout = 5000
        self._scope.clear()
        #self._scope.write("COMM_FORMAT OFF,BYTE,BIN")
        # self._scope.write(r"""vbs 'app.settodefaultsetup' """)
        command = "*IDN?"
        print("[0] " + command)
        ret = self._scope.query(command)
        print("[!] Connected scope:", ret)
        
    def disconnect(self):
        self._scope.close()
        
    def clear(self):
        self._scope.write("*CLS")
        
    def query(self, command):
        return self._scope.query(command)
    
    def write(self, command):
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        
    def read(self):
        return self._scope.read()
    
    
    ###
    # Enable a channel
    # @channel
    # String for the channel, can be:
    # "CH1", "CH2", "CH3"...
    
    def enableChannel(self, channel):
        command = "SELect:" + channel + " ON"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        
    ###
    # Disable a channel
    # @channel
    # String for the channel, can be:
    # "CH1", "CH2", "CH3"...
    
    def disableChannel(self, channel):
        command = "SELect:" + channel + " OFF"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        
    ###
    # Check enabled channels
    #
    # @ return
    # List containing enabled channels
    ###
    
    def checkChannels(self, channels=6):
        command = "SEL?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return ["CH" + str(i + 1) for i, val in enumerate(np.array(self._scope.query(command).strip().split(";")[-channels:], dtype='int')) if val == 1]
    

    ###
    # Read the trigger position value
    #
    # @ return
    # Integer with the value of the trigger position
    ###
    def getTriggerPosition(self, channel):
        command = channel + ":POSition?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return self._scope.query(command).strip()        
   
    
    
    ###
    # Defines the source channel for the trigger
    # @channel
    # String for the analysed channel, can be:
    # "CH1", "CH2",...
    # By default CH1
    
    def setTriggerSource(self, channel="CH1"):
        command = "TRIGger:A:EDGE:SOUrce " + channel
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    ###
    # Read the trigger source value
    #
    # @ return
    # String with the value of the trigger source
    ###
    def getTriggerSource(self):
        command = "TRIGger:A:EDGE:SOUrce?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return self._scope.query(command).strip()        
     
 
    
    ###
    # Defines the trigger level for a certain channel.
    # @channel
    # String for the analysed channel, can be:
    # "CH1", "CH2",...
    # @level
    # Integer for the level, can be:
    # By default zero
    
    def setTriggerLevel(self, channel, level=0):
        command = "TRIGger:A:LEV:" + channel + " " + str(level)
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    ###
    # Read the trigger level value
    # @channel
    # String for the analysed channel, can be:
    # "CH1", "CH2, ...
    # By default "CH1"y.
    # @ return
    # Integer with the value of the trigger level for that channel, in volts
    ###
    def getTriggerLevel(self, channel="CH1"):
        command = "TRIGger:A:LEV:" + channel + "?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return self._scope.query(command).strip()   
    
    ###
    # Defines the trigger position (trigger offset) - the time interval between trigger point and reference
    # point to analize the signal some time before or after the trigger event.
    # @channel
    # String for the analysed channel, can be:
    # "CH1", "CH2",...
    # @position
    # Integer for the position, can be:
    # By default zero
    
    def setTriggerPosition(self, channel, position=0):
        command = channel + ":POSition" + str(position)
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    ###
    # Read the trigger mode value
    #
    # @ return
    # Integer with the value of the trigger mode
    ###
    def getTriggerMode(self):
        command = "HOR:MODE?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return self._scope.query(command).strip()        
    
    
    ###
    # Change the trigger mode
    # @mode
    # String for the trigger mode, can be:
    # NORMAL, AUTO
    
    def setTriggerMode(self, mode):
        command = "TRIGger:A:MODE " + mode
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    ###
    # Read the trigger mode value
    #
    # @ return
    # Integer with the value of the trigger mode
    ###
    def getTriggerMode(self):
        command = "TRIGger:A:MODE?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return self._scope.query(command).strip()
         
    ###
    # Change the time division
    #
    # @timeDiv
    # Float with the time division, can be:
    # "1e9", "1e-6"...
    #
    ###
    def setTimeDiv(self, timeDiv):
        command = "HORizontal:SCAle " + str(timeDiv)
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        
    ###
    # Read the time division value
    #
    #
    # @ return
    # Integer with the value of the time per division
    ###
    def getTimeDiv(self):
        command = "HOR:SCA?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return int(float(self._scope.query(command).strip().lower()))
    
    ###
    # Change the sample rate
    #
    # @SampleRate
    # Float with the sample rate, can be:
    # 1.25e9, 500e6"...
    #
    ###
    def setSampleRate(self, SampleRate):
        command = "HOR:SAMPLER " + str(SampleRate)
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        
    ###
    # Read the sample rate value
    #
    # @ return
    # Integer with the value of the sample rate
    ###
    def getSampleRate(self):
        command = "HOR:SAMPLER?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return int(float(self._scope.query(command).strip().lower()))

    ###
    # Change the volts division of one of the channels
    #
    # @channel
    # String with the name of the channel, can be:
    # "CH1", "CH2", "CH3", "CH4",...
    #
    # @voltDiv
    # Float with the new value of volts per div, can be:
    # 100e-3, 1...
    ###
    
    def setVoltsDiv(self, channel, voltDiv):
        command = channel + ':SCA ' + str(voltDiv)
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
    
    ###
    # Read the volts division value
    #
    # @channel
    # String with the desired channel, can be:
    # "CH1", "CH2", "CH3"...
    #
    # @ return
    # Integer with the value of the volts per division for that channel
    ###
    def getVoltsDiv(self, channel):
        command = channel + ":SCA?"
        if(DEBUG_MODE):
            print("[0] " + command)
        return int(float(self._scope.query(command).strip().lower()))

    ###
    # Read Waveform
    #
    # @channel
    # String where, can be:
    # "CH1", "CH2", "CH3"...
    #
    # @ return
    # Integer with the value of the volts per division for that channel
    ###
    def getWaveform(self, channel, encode="ASCIi", bits=16):
        self._scope.write("DATa:SOUrce {}" .format(channel))
        self._scope.write("DATa:ENCdg {}" .format(encode))
        self._scope.write("WFMOutpre:BIT_Nr {}" .format(bits))
        string_numeros = self._scope.query("CURVe?").strip()

        # Dividir el string en subcadenas separadas por comas y convertirlas en números
        return np.array([int(numero) for numero in string_numeros.split(',')])
        
