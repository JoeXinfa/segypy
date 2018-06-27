# -*- coding: utf-8 -*-

"""
A Python module for reading/writing of SEG-Y formatted files

I found the version 0.3 dated 2005/10/3, written by Thomas Mejer Hansen.
http://segymat.sourceforge.net/segypy
License : GNU LESSER GENERAL PUBLIC LICENSE

The revision between 2017/1/1 and 2017/7/7 is named version 0.4.

I started using Git for version control in July 2017.
"""

__version__ = '0.5.1'

import struct
import collections
from datetime import datetime
import numpy as np
import warnings
#import os

endian = '>' # Big Endian
#endian = '<' # Little Endian
#endian = '=' # Native

# Table for data type to character size
dtype2csize = {
  'int32'  : 4,
  'uint32' : 4,
  'int16'  : 2,
  'uint16' : 2,
  'float32': 4,
  'double' : 8,
  'char'   : 1,
  'uchar'  : 1}

# Table for data type to character type
dtype2ctype = {
  'int32'  : 'l',
  'uint32' : 'L',
  'int16'  : 'h',
  'uint16' : 'H',
  'float32': 'f',
  'double' : 'd',
  'char'   : 'c',
  'uchar'  : 'B'}

bytes_STFH = 3200
bytes_SBFH = 400
bytes_SFH = 3600
bytes_STH = 240

nspb = 100000000 # 100 million samples per block for unpack

# Initialize SEGY binary file header 
SH_def = collections.OrderedDict()
SH_def["Job"] = {"pos": 3200, "type": "int32", "def": 0}
SH_def["Line"] = {"pos": 3204, "type": "int32", "def": 0}
SH_def["Reel"] = {"pos": 3208, "type": "int32", "def": 0}
SH_def["DataTracePerEnsemble"] = {"pos": 3212, "type": "int16", "def": 0}
SH_def["AuxiliaryTracePerEnsemble"] = {"pos": 3214, "type": "int16", "def": 0}
SH_def["dt"] = {"pos": 3216, "type": "uint16", "def": 1000}
SH_def["dtOrig"] = {"pos": 3218, "type": "uint16", "def": 0}
SH_def["ns"] = {"pos": 3220, "type": "uint16", "def": 0}
SH_def["nsOrig"] = {"pos": 3222, "type": "uint16", "def": 0} 
SH_def["DataSampleFormat"] = {"pos": 3224, "type": "int16", "def": 5}
SH_def["DataSampleFormat"]["descr"] = {0: {
  1: "IBM Float", 
  2: "32 bit Integer", 
  3: "16 bit Integer", 
  8: "8 bit Integer"}}

SH_def["DataSampleFormat"]["descr"][1] = {
  1: "4-byte IBM floating point", 
  2: "4-byte, two's complement integer", 
  3: "2-byte, two's complement integer", 
  4: "4-byte, fixed-point with gain (obsolete)",
  5: "4-byte IEEE floating point",
  8: "1-byte Integer"}

SH_def["DataSampleFormat"]["bps"] = {0: {
  1: 4, 
  2: 4, 
  3: 2, 
  8: 1}}

SH_def["DataSampleFormat"]["bps"][1] = {
  1: 4, 
  2: 4, 
  3: 2, 
  5: 4, 
  8: 1}

SH_def["DataSampleFormat"]["datatype"] = {0: {
  1: 'ibm', 
  2: 'int32', 
  3: 'int16', 
  8: 'uchar'}}

SH_def["DataSampleFormat"]["datatype"][1] = {
  1: 'ibm', 
  2: 'int32', 
  3: 'int16', 
  5: 'float32', 
  8: 'uchar'}

SH_def["EnsembleFold"] = {"pos": 3226, "type": "int16", "def": 0}
SH_def["TraceSorting"] = {"pos": 3228, "type": "int16", "def": 0}
SH_def["VerticalSumCode"] = {"pos": 3230, "type": "int16", "def": 0}
SH_def["SweepFrequencyEnd"] = {"pos": 3234, "type": "int16", "def": 0}
SH_def["SweepLength"] = {"pos": 3236, "type": "int16", "def": 0}
SH_def["SweepType"] = {"pos": 3238, "type": "int16", "def": 0}
SH_def["SweepChannel"] = {"pos": 3240, "type": "int16", "def": 0}
SH_def["SweepTaperlengthStart"] = {"pos": 3242, "type": "int16", "def": 0}
SH_def["SweepTaperLengthEnd"] = {"pos": 3244, "type": "int16", "def": 0} 
SH_def["TaperType"] = {"pos": 3246, "type": "int16", "def": 0}
SH_def["CorrelatedDataTraces"] = {"pos": 3248, "type": "int16", "def": 0}
SH_def["BinaryGain"] = {"pos": 3250, "type": "int16", "def": 0}
SH_def["AmplitudeRecoveryMethod"] = {"pos": 3252, "type": "int16", "def": 0}
SH_def["MeasurementSystem"] = {"pos": 3254, "type": "int16", "def": 0} 
SH_def["ImpulseSignalPolarity"] = {"pos": 3256, "type": "int16", "def": 0}
SH_def["VibratoryPolarityCode"] = {"pos": 3258, "type": "int16", "def": 0}
SH_def["Unassigned1"] = {"pos": 3260, "type": "int16", "n": 120, "def": 0}
SH_def["SegyFormatRevisionNumber"] = {"pos": 3500, "type": "uint16", "def": 100}
SH_def["FixedLengthTraceFlag"] = {"pos": 3502, "type": "uint16", "def": 0} 
SH_def["NumberOfExtTextualHeaders"] = {"pos": 3504, "type": "uint16", "def": 0}
SH_def["Unassigned2"] = {"pos": 3506, "type": "int16", "n": 47, "def": 0} 

# Initialize SEGY trace header
# Use ordered dict so can pack and cat headers in order to a big byte string
# for file.write together. Purpose is to avoid call file.write repeatedly
# for each header. This way, the 'pos' key may become unnecessary...
STH_def = collections.OrderedDict()
STH_def["TraceSequenceLine"] = {"pos": 0 , "type": "int32"}
STH_def["TraceSequenceFile"] = {"pos": 4 , "type": "int32"}
STH_def["FieldRecord"] = {"pos": 8 , "type": "int32"}
STH_def["TraceNumber"] = {"pos": 12 , "type": "int32"}
STH_def["EnergySourcePoint"] = {"pos": 16 , "type": "int32"} 
STH_def["cdp"] = {"pos": 20 , "type": "int32"}
STH_def["cdpTrace"] = {"pos": 24 , "type": "int32"}
STH_def["TraceIdenitifactionCode"] = {"pos": 28 , "type": "uint16"}
STH_def["TraceIdenitifactionCode"]["descr"] = {0: {
  1: "Seismic data", 
  2: "Dead", 
  3: "Dummy", 
  4: "Time Break", 
  5: "Uphole", 
  6: "Sweep", 
  7: "Timing", 
  8: "Water Break"}}
STH_def["TraceIdenitifactionCode"]["descr"][1] = {
  -1: "Other",
   0: "Unknown",
   1: "Seismic data",
   2: "Dead",
   3: "Dummy",
   4: "Time break",
   5: "Uphole",
   6: "Sweep",
   7: "Timing",
   8: "Waterbreak",
   9: "Near-field gun signature",
  10: "Far-field gun signature",
  11: "Seismic pressure sensor",
  12: "Multicomponent seismic sensor - Vertical component",
  13: "Multicomponent seismic sensor - Cross-line component",
  14: "Multicomponent seismic sensor - In-line component",
  15: "Rotated multicomponent seismic sensor - Vertical component",
  16: "Rotated multicomponent seismic sensor - Transverse component",
  17: "Rotated multicomponent seismic sensor - Radial component",
  18: "Vibrator reaction mass",
  19: "Vibrator baseplate",
  20: "Vibrator estimated ground force",
  21: "Vibrator reference",
  22: "Time-velocity pairs"}
STH_def["NSummedTraces"] = {"pos": 30 , "type": "int16"}
STH_def["NStackedTraces"] = {"pos": 32 , "type": "int16"}
STH_def["DataUse"] = {"pos": 34 , "type": "int16"}
STH_def["DataUse"]["descr"] = {0: {
  1: "Production", 
  2: "Test"}}
STH_def["DataUse"]["descr"][1] = STH_def["DataUse"]["descr"][0]
STH_def["offset"] = {"pos": 36 , "type": "int32"}
STH_def["ReceiverGroupElevation"] = {"pos": 40 , "type": "int32"}
STH_def["SourceSurfaceElevation"] = {"pos": 44 , "type": "int32"}
STH_def["SourceDepth"] = {"pos": 48 , "type": "int32"}
STH_def["ReceiverDatumElevation"] = {"pos": 52 , "type": "int32"}
STH_def["SourceDatumElevation"] = {"pos": 56 , "type": "int32"}
STH_def["SourceWaterDepth"] = {"pos": 60 , "type": "int32"}
STH_def["GroupWaterDepth"] = {"pos": 64 , "type": "int32"}
STH_def["ElevationScalar"] = {"pos": 68 , "type": "int16"}
STH_def["SourceGroupScalar"] = {"pos": 70 , "type": "int16"}
STH_def["SourceX"] = {"pos": 72 , "type": "int32"}
STH_def["SourceY"] = {"pos": 76 , "type": "int32"}
STH_def["GroupX"] = {"pos": 80 , "type": "int32"}
STH_def["GroupY"] = {"pos": 84 , "type": "int32"}
STH_def["CoordinateUnits"] = {"pos": 88 , "type": "int16"}
STH_def["CoordinateUnits"]["descr"] = {1: {
  1: "Length (meters or feet)",
  2: "Seconds of arc"}}
STH_def["CoordinateUnits"]["descr"][1] = {
  1: "Length (meters or feet)",
  2: "Seconds of arc",
  3: "Decimal degrees",
  4: "Degrees, minutes, seconds (DMS)"}
STH_def["WeatheringVelocity"] = {"pos": 90 , "type": "int16"}
STH_def["SubWeatheringVelocity"] = {"pos": 92 , "type": "int16"}
STH_def["SourceUpholeTime"] = {"pos": 94 , "type": "int16"}
STH_def["GroupUpholeTime"] = {"pos": 96 , "type": "int16"}
STH_def["SourceStaticCorrection"] = {"pos": 98 , "type": "int16"}
STH_def["GroupStaticCorrection"] = {"pos": 100 , "type": "int16"}
STH_def["TotalStaticApplied"] = {"pos": 102 , "type": "int16"}
STH_def["LagTimeA"] = {"pos": 104 , "type": "int16"}
STH_def["LagTimeB"] = {"pos": 106 , "type": "int16"}
STH_def["DelayRecordingTime"] = {"pos": 108 , "type": "int16"}
STH_def["MuteTimeStart"] = {"pos": 110 , "type": "int16"}
STH_def["MuteTimeEND"] = {"pos": 112 , "type": "int16"}
STH_def["ns"] = {"pos": 114 , "type": "uint16"}
STH_def["dt"] = {"pos": 116 , "type": "uint16"}
STH_def["GainType"] = {"pos": 119 , "type": "int16"}
STH_def["GainType"]["descr"] = {0: {
  1: "Fixes", 
  2: "Binary",
  3: "Floating point"}}
STH_def["GainType"]["descr"][1] = STH_def["GainType"]["descr"][0]
STH_def["InstrumentGainConstant"] = {"pos": 120 , "type": "int16"}
STH_def["InstrumentInitialGain"] = {"pos": 122 , "type": "int16"}
STH_def["Correlated"] = {"pos": 124 , "type": "int16"}
STH_def["Correlated"]["descr"] = {0: {
  1: "No", 
  2: "Yes"}}
STH_def["Correlated"]["descr"][1] = STH_def["Correlated"]["descr"][0]
STH_def["SweepFrequenceStart"] = {"pos": 126 , "type": "int16"}
STH_def["SweepFrequenceEnd"] = {"pos": 128 , "type": "int16"}
STH_def["SweepLength"] = {"pos": 130 , "type": "int16"}
STH_def["SweepType"] = {"pos": 132 , "type": "int16"}
STH_def["SweepType"]["descr"] = {0: {
  1: "linear", 
  2: "parabolic",
  3: "exponential",
  4: "other"}}
STH_def["SweepType"]["descr"][1] = STH_def["SweepType"]["descr"][0]
STH_def["SweepTraceTaperLengthStart"] = {"pos": 134 , "type": "int16"}
STH_def["SweepTraceTaperLengthEnd"] = {"pos": 136 , "type": "int16"}
STH_def["TaperType"] = {"pos": 138 , "type": "int16"}
STH_def["TaperType"]["descr"] = {0: {
  1: "linear", 
  2: "cos2c",
  3: "other"}}
STH_def["TaperType"]["descr"][1] = STH_def["TaperType"]["descr"][0]
STH_def["AliasFilterFrequency"] = {"pos": 140 , "type": "int16"}
STH_def["AliasFilterSlope"] = {"pos": 142 , "type": "int16"}
STH_def["NotchFilterFrequency"] = {"pos": 144 , "type": "int16"}
STH_def["NotchFilterSlope"] = {"pos": 146 , "type": "int16"}
STH_def["LowCutFrequency"] = {"pos": 148 , "type": "int16"}
STH_def["HighCutFrequency"] = {"pos": 150 , "type": "int16"}
STH_def["LowCutSlope"] = {"pos": 152 , "type": "int16"}
STH_def["HighCutSlope"] = {"pos": 154 , "type": "int16"}
STH_def["YearDataRecorded"] = {"pos": 156 , "type": "int16"}
STH_def["DayOfYear"] = {"pos": 158 , "type": "int16"}
STH_def["HourOfDay"] = {"pos": 160 , "type": "int16"}
STH_def["MinuteOfHour"] = {"pos": 162 , "type": "int16"}
STH_def["SecondOfMinute"] = {"pos": 164 , "type": "int16"}
STH_def["TimeBaseCode"] = {"pos": 166 , "type": "int16"}
STH_def["TimeBaseCode"]["descr"] = {0: {
  1: "Local", 
  2: "GMT", 
  3: "Other"}}
STH_def["TimeBaseCode"]["descr"][1] = {
  1: "Local", 
  2: "GMT", 
  3: "Other", 
  4: "UTC"}
STH_def["TraceWeightningFactor"] = {"pos": 168 , "type": "int16"}
STH_def["GeophoneGroupNumberRoll1"] = {"pos": 170 , "type": "int16"}
STH_def["GeophoneGroupNumberFirstTraceOrigField"] = {"pos": 172 , "type": "int16"}
STH_def["GeophoneGroupNumberLastTraceOrigField"] = {"pos": 174 , "type": "int16"}
STH_def["GapSize"] = {"pos": 176 , "type": "int16"}
STH_def["OverTravel"] = {"pos": 178 , "type": "int16"}
STH_def["OverTravel"]["descr"] = {0: {
  1: "down (or behind)", 
  2: "up (or ahead)",
  3: "other"}}
STH_def["OverTravel"]["descr"][1] = STH_def["OverTravel"]["descr"][0]
STH_def["cdpX"] = {"pos": 180 , "type": "int32"}
STH_def["cdpY"] = {"pos": 184 , "type": "int32"}
STH_def["Inline3D"] = {"pos": 188 , "type": "int32"}
STH_def["Crossline3D"] = {"pos": 192 , "type": "int32"}
STH_def["ShotPoint"] = {"pos": 192 , "type": "int32"}
STH_def["ShotPointScalar"] = {"pos": 200 , "type": "int16"}
STH_def["TraceValueMeasurementUnit"] = {"pos": 202 , "type": "int16"}
STH_def["TraceValueMeasurementUnit"]["descr"] = {1: {
  -1: "Other", 
  0: "Unknown (should be described in Data Sample Measurement Units Stanza) ", 
  1: "Pascal (Pa)", 
  2: "Volts (V)", 
  3: "Millivolts (v)", 
  4: "Amperes (A)", 
  5: "Meters (m)", 
  6: "Meters Per Second (m/s)", 
  7: "Meters Per Second squared (m/&s2)Other", 
  8: "Newton (N)", 
  9: "Watt (W)"}}
STH_def["TransductionConstantMantissa"] = {"pos": 204 , "type": "int32"}
STH_def["TransductionConstantPower"] = {"pos": 208 , "type": "int16"}
STH_def["TransductionUnit"] = {"pos": 210 , "type": "int16"}
STH_def["TransductionUnit"]["descr"] = STH_def["TraceValueMeasurementUnit"]["descr"]
STH_def["TraceIdentifier"] = {"pos": 212 , "type": "int16"}
STH_def["ScalarTraceHeader"] = {"pos": 214 , "type": "int16"}
STH_def["SourceType"] = {"pos": 216 , "type": "int16"}
STH_def["SourceType"]["descr"] = {1: {
  -1: "Other (should be described in Source Type/Orientation stanza)",
   0: "Unknown",
   1: "Vibratory - Vertical orientation",
   2: "Vibratory - Cross-line orientation",
   3: "Vibratory - In-line orientation",
   4: "Impulsive - Vertical orientation",
   5: "Impulsive - Cross-line orientation",
   6: "Impulsive - In-line orientation",
   7: "Distributed Impulsive - Vertical orientation",
   8: "Distributed Impulsive - Cross-line orientation",
   9: "Distributed Impulsive - In-line orientation"}}
STH_def["SourceEnergyDirectionMantissa"] = {"pos": 218 , "type": "int32"}
STH_def["SourceEnergyDirectionExponent"] = {"pos": 222 , "type": "int16"}
STH_def["SourceMeasurementMantissa"] = {"pos": 224 , "type": "int32"}
STH_def["SourceMeasurementExponent"] = {"pos": 228 , "type": "int16"}
STH_def["SourceMeasurementUnit"] = {"pos": 230 , "type": "int16"}
STH_def["SourceMeasurementUnit"]["descr"] = {1: {
  -1: "Other (should be described in Source Measurement Unit stanza)",
   0: "Unknown",
   1: "Joule (J)",
   2: "Kilowatt (kW)",
   3: "Pascal (Pa)",
   4: "Bar (Bar)",
   4: "Bar-meter (Bar-m)",
   5: "Newton (N)",
   6: "Kilograms (kg)"}}
STH_def["UnassignedInt1"] = {"pos": 232 , "type": "int32"}
STH_def["UnassignedInt2"] = {"pos": 236 , "type": "int32"}

def image(Data):
  """
  i Data : 2D array, nsample by ntrace?
  o Plot window
  Plot image of 2D array
  """
  import matplotlib.pyplot as plt
  plt.imshow(Data)
  plt.title('segypy test')
  plt.grid(True)
  plt.show()

def wiggle(Data, dt=4, skipt=1, maxval=8, lwidth=.1):
  """
  i Data : 2D array, nsample by ntrace?
  i dt : float, trace sampling interval, e.g. 4ms or 5m.
  i skipt : integer, number of traces to skip plot
  i maxval : float, amplitude scalar
  i lwidth : float, line width
  o Plot window
  Plot traces in wiggle form
  """
  import matplotlib.pyplot as plt
    
  ntraces, ns = Data.shape
  t = range(ns) * dt # vertical axis, time or depth

  for i in range(0, ntraces, skipt):
    trace = Data[:,i]
    #trace[0] = 0
    #trace[ns-1] = 0  
    plt.plot(i+trace/maxval, t, color='black', linewidth=lwidth)
    #for a in range(len(trace)):
    #  if (trace[a] < 0):
    #    trace[a] = 0 
    #plt.fill(i+Data[:,i]/maxval, t, color='k', facecolor='g')
    plt.fill(i+trace/maxval, t, 'k', linewidth=0)
  plt.grid(True)
  plt.show()

def getDefaultSegyHeader():
  """
  o SH : dictionary, Segy binary file header
  Get default Segy header values.
  """
  SH = {}
  for key in SH_def.keys(): 
    keyDict = SH_def[key] # is a dictionary
    if 'def' in keyDict :
      val = keyDict['def']
    else :
      val = 0 # If no default value, set zero.
    SH[key] = val
  return SH

def getDSF_fromDataType(Data):
  """
  i Data : array, numpy array
  o dsf : integer, code number for data sample format
  get data sample format from array data type
  """
  if Data.dtype == 'float64' :
    warnings.warn('Cast data type float64 to float32.')
    Data = Data.astype('float32', copy=False)

  if Data.dtype == 'int32' :
    dsf = 2
  elif Data.dtype == 'int16' :
    dsf = 3
  elif Data.dtype == 'float32' :
    dsf = 5 
  elif Data.dtype == 'int8' :
    dsf = 8
  else :
    raise TypeError('Cannot handle data type:', Data.dtype)

  return dsf

def setSegyHeaders(SH, ntraces=None, ns=None, dt=None, dsf=None):
  """
  i SH : dictionary, Segy binary file header with default values
  i ntraces : integer, number of traces
  i ns : integer, number of samples per trace
  i dt : integer, sample interval in microsecond or millimeter
  i dsf : integer, data sample format code
  o SH : dictionary, Segy binary file header
  Set Segy file header values. Prepare for write Segy.
  """
  if ntraces is not None:
    SH["ntraces"] = int(ntraces)
  if ns is not None:
    SH["ns"] = int(ns) 
  if dt is not None:
    SH["dt"] = int(dt) 
  if dsf is not None:
    SH['DataSampleFormat'] = dsf

def setSegyTraceHeaders(mySTH=None, ntraces=100, ns=100, dt=1000):
  """
  i mySTH : dictionary, user-supplied entries [key][trace]
  i ntraces : integer, number of traces
  i ns : integer, number of samples per trace
  i dt : integer, sample interval in microsecond or millimeter
  o STH : dictionary, Segy trace header
  Set Segy trace header values. Prepare for write Segy.
  """
  STH = {}
  for key in STH_def.keys(): 
    STH[key] = np.zeros(ntraces, dtype=np.int32)
      
  for a in range(ntraces):      
    STH["TraceSequenceLine"][a] = a + 1
    STH["TraceSequenceFile"][a] = a + 1
    STH["FieldRecord"][a] = 1000
    STH["TraceNumber"][a] = a + 1
    STH["ns"][a] = ns
    STH["dt"][a] = dt

  # Overwrite using user-supplied header values
  if mySTH is not None:
    for key in mySTH.keys():
      print('Custom added key to trace header:', key)
      for a in range(ntraces):
        STH[key][a] = mySTH[key][a]

  return STH

def getSegyTraceHeader(SH, data, TH_name, TH_dict=None, itrace=0):
  """
  i SH : dictionary, Segy binary file header
  i data : byte object, from read the file in binary mode
  i TH_name : string, Trace header name
  i TH_dict : dictionary, Trace header byte position and data type
  i itrace : integer, the trace number to read.
  o TH_value : array, numpy, header value for each trace
  Get the value for a trace header TH_name.
  """

  if TH_dict is None :
    TH_pos = STH_def[TH_name]["pos"]
    TH_format = STH_def[TH_name]["type"]
  else :
    TH_pos = TH_dict[TH_name]["pos"]
    TH_format = TH_dict[TH_name]["type"]

  print('Reading trace header:', TH_name, TH_pos+1, TH_format)
  ntraces = SH["ntraces"]
  ns = SH["ns"]
  bps = getBytePerSample(SH)
  traceByteSize = bytes_STH + ns * bps

  if itrace == 0 : # get all traces
    TH_value = np.zeros(ntraces, dtype=TH_format)
    for itrace in range(ntraces):
      pos = bytes_SFH + traceByteSize * itrace + TH_pos
      TH_value[itrace] = getValue(data, pos, TH_format, endian)
  else : # get one trace
    pos = bytes_SFH + traceByteSize * (itrace - 1) + TH_pos
    TH_value = getValue(data, pos, TH_format, endian)

  # If dt in STH is zero, read from SBFH.
  if TH_name == "dt" :
    if TH_value[0] == 0 :
      TH_value[:] = SH["dt"]

  return TH_value

def getSegyTraceHeaders(SH, data, TH_dict=None, itrace=0):
  """
  i SH : dictionary, Segy binary file header
  i data : byte object, from read the file in binary mode
  i TH_dict : dictionary, Trace header byte position and data type
  i itrace : integer, the trace number to read.
  o SegyTraceHeaders : dictionary, For each key-value pair, key is
    the header name in TH_dict, value is array of size ntrace.
  Get the value for trace headers in TH_dict.
  """
  if TH_dict is None:
    TH_read = STH_def
  else:
    TH_read = TH_dict

  SegyTraceHeaders = {}
  for key in TH_read.keys():    
    TH_value = getSegyTraceHeader(SH, data, key, TH_read, itrace)
    SegyTraceHeaders[key] = TH_value
  return SegyTraceHeaders

def readSegy(filename, TH_dict=None, TH_only=False) :
  """
  i filename : string, Segy filename
  i TH_dict : dictionary, Trace header byte position and data type
    This controls read only useful trace headers, thus saves time
    from reading all of the 91 headers.
  i TH_only : bool, flag for reading headers only.
  o Data : 2D array, nTrace by nSamplePerTrace
  o SH : dictionary, Segy binary file header
  o SegyTraceHeaders : dictionary, Segy trace header
  Read Segy file.
  """
  print("Reading file:", filename)

  SH, data = getSegyHeader(filename)
  print('Done read file headers at', datetime.now())

  SegyTraceHeaders = getSegyTraceHeaders(SH, data, TH_dict)
  print('Done read trace headers at', datetime.now())

  if TH_only is True :
    return SH, SegyTraceHeaders  
  else : 
    bps = getBytePerSample(SH)
    ndummy_samples = int(bytes_STH / bps)
    index = bytes_SFH
    filesize = len(data)
    nd = int((filesize - bytes_SFH) / bps)

    revno = getRevisionNumber(SH)
    dsf = SH["DataSampleFormat"]
    DataDescr = SH_def["DataSampleFormat"]["descr"][revno][dsf]
    print("DataSampleFormat = " + str(dsf) + ', ' + DataDescr)

    dtype = SH_def["DataSampleFormat"]["datatype"][revno][dsf]
    Data = getValue(data, index, dtype, endian, nd)
    print('Done read trace samples at', datetime.now())
  
    ntraces = SH["ntraces"]
    nsDummyTrace = SH['ns'] + ndummy_samples
    Data = np.reshape(Data, (ntraces, nsDummyTrace))
    #print(Data.dtype)
    #Data = Data.astype('float32') # cast all types to 32-bit float

    # Strip off header dummy data
    Data = Data[:,ndummy_samples:nsDummyTrace]
    
    # Deal with 8-bit integer
    if dsf == 8:
      #for i in np.arange(ntraces):
      #  for j in np.arange(SH['ns']):
      #    if Data[i][j] > 128:
      #      Data[i][j] = Data[i][j] - 256
      Data[Data > 128] -= 256
  
    return Data, SH, SegyTraceHeaders  

def readSegyTrace(filename, TH_dict=None, itrace=1):
  """
  i filename : string, Segy filename
  i TH_dict : dictionary, Trace header byte position and data type
  i itrace : integer, the trace number to read
  o SegyTraceHeaders : dictionary, trace headers
  o SegyTraceData : array, 1D, trace data
  """
  SH, data = getSegyHeader(filename)
  SegyTraceHeaders = getSegyTraceHeaders(SH, data, TH_dict, itrace)

  bps = getBytePerSample(SH)
  ns = SH['ns'] # number of samples per trace
  bytesTrace = bytes_STH + ns * bps
  index = bytes_SFH + (itrace - 1) * bytesTrace + bytes_STH
  revno = getRevisionNumber(SH)
  dsf = SH["DataSampleFormat"]
  dtype = SH_def["DataSampleFormat"]["datatype"][revno][dsf]
  SegyTraceData = getValue(data, index, dtype, endian, ns)
  return SegyTraceHeaders, SegyTraceData

def getSegyHeader(filename):
  """
  i filename : string, Segy filename
  o SH : dictionary, Segy binary file header
  o data : byte object, from read the file in binary mode
  Read Segy binary file headers.
  """
  data = open(filename, 'rb').read()
  SH = {'filename': filename}
  for key in SH_def.keys(): 
    pos = SH_def[key]["pos"]
    fmt = SH_def[key]["type"]
    SH[key] = getValue(data, pos, fmt, endian)   

  bps = getBytePerSample(SH)
  ns = SH['ns']
  traceByteSize = bytes_STH + ns * bps
  filesize = len(data)
  ntraces = (filesize - bytes_SFH) / traceByteSize
  SH["ntraces"] = int(ntraces) 

  return SH, data

def writeSegy(filename, Data, dt=1000, STFH='', mySTH=None):
  """
  i filename : string, output filename
  i Data : float array 2D, ns by ntraces
  i dt : float, sample interval in microsecond or millimeter
  i STFH : string, Segy textual file header
  i mySTH : dictionary, [key][trace], value for trace header  
  o file in disk
  Write Segy file to disk
  """
  print("Writing file:", filename)

  ns, ntraces = Data.shape
  print('Number of Traces =', ntraces)
  print('Number of Samples per Trace =', ns)

  # Prepare the Segy binary file header
  SH = getDefaultSegyHeader() 
  dsf = getDSF_fromDataType(Data)
  setSegyHeaders(SH, ntraces, ns, dt, dsf)

  # Prepare the Segy trace header
  STH = setSegyTraceHeaders(mySTH, ntraces, ns, dt)

  writeSegyStructure(filename, Data, STFH, SH, STH)

def writeSegyStructure(filename, Data, STFH, SH, STH):
  """
  i filename : string, output Segy filename
  i Data : float array 2D, ns by ntraces
  i STFH : string, Segy textual file header
  i SH : dictionary, Segy binary file header
  i STH : dictionary, Segy trace header
  o file in disk
  Write Segy file to disk
  """
  #bufsize = 0
  #f = open(filename, 'wb', bufsize)
  f = open(filename, 'wb')

  # Write textual file header
  f.write(STFH.encode('ascii')) # ASCII format/encoding
  #f.write(STFH.encode('cp500')) # EBCDIC format/encoding

  # Variables used from the input SH
  revno = getRevisionNumber(SH)
  dsf = SH['DataSampleFormat']
  ntraces = SH['ntraces']
  ns = SH['ns']

  DataDescr = SH_def["DataSampleFormat"]["descr"][revno][dsf]
  print("Write DataSampleFormat = " + str(dsf) + ' ' + DataDescr)

  # Write binary file header
  SH_ByteArray = bytearray()
  for key in SH_def.keys():
    dtype = SH_def[key]["type"]
    value = SH[key]
    bytesObject = packValue(value, dtype, endian)
    SH_ByteArray.extend(bytesObject)
  index = bytes_STFH
  f.seek(index)
  f.write(SH_ByteArray)

  index = index + bytes_SBFH
  dtype = SH_def['DataSampleFormat']['datatype'][revno][dsf]
  ctype = dtype2ctype[dtype]
  bps = SH_def['DataSampleFormat']['bps'][revno][dsf]
  traceByteSize = bytes_STH + ns * bps

  for itrace in range(ntraces):

    if itrace % 10000 == 0 :
      print('Total traces %i, Writing trace %i, Progress = %6.2f' %
        (ntraces, itrace, itrace/ntraces*100))

    # When should do flush and sync? Not system automatic?
    #if itrace % 40000 == 0 :
    #  f.flush() # flush internal buffer to OS buffer
    #  os.fsync(f.fileno()) # push OS buffer to disk      

    TraceByteArray = bytearray()
    for key in STH_def.keys(): # ordered dictionary
      dtype = STH_def[key]["type"]
      value = STH[key][itrace]
      bytesObject = packValue(value, dtype, endian)
      TraceByteArray.extend(bytesObject)

    aray1d = Data[:, itrace]
    cformat = endian + str(ns) + ctype
    aray1dPack = struct.pack(cformat, *aray1d)
    TraceByteArray.extend(aray1dPack)

    # Write to file once per trace
    f.seek(index)
    f.write(TraceByteArray)
    index += traceByteSize

  f.close()

def packValue(value, dtype='int32', endian='>', number=1):
  """
  i value : one number of an array, value to be packed.
  i dtype : string, data type, e.g. int32, int16, uint16.
  i endian : character, byte order, e.g. native, big-endian, little-endian.
  i number : integer, the number of numbers in value 
  o bytesObject : bytes object, returned by struct.pack.
  Convert Python values to C struct represented as Python bytes object.
  This is used in handling binary data stored in files.
  This is used to pack each head and concatenate to a bytes array before
  calling file write once for all headers.
  Call fh.seek and fh.write every time for each header is too costly.
  """
  ctype = dtype2ctype[dtype]
  cformat = endian + ctype * number
  bytesObject = struct.pack(cformat, value)
  return bytesObject

def getValue(data, index, dtype='int32', endian='>', number=1):
  """
  i data : bytes object, returned by read file in binary mode.
  i index : integer, the starting byte location
  i dtype : string, data type, e.g. int32, int16, uint16.
  i endian : character, byte order
  i number : integer, the number of numbers in data
  o Value : value, when only one number.
  o Value : tuple, struct.unpack returns a tuple
  """
  if (dtype == 'ibm'): # IBM float data
    Value = np.empty(number, dtype='float32')
    for i in np.arange(number):
      index_ibm = i * 4 + index
      Value[i] = ibm2ieee(data[index_ibm:index_ibm+4])
  else: # all other types of data

    ctype = dtype2ctype[dtype]
    csize = dtype2csize[dtype]
    Value = np.empty(number, dtype=dtype)

    # Unpack all at once
    # For files of size close to memory, MemoryError.
    #cformat = endian + ctype * number
    #index_end = index + csize * number
    #Value = struct.unpack(cformat, data[index:index_end])

    nblock = int(number / nspb) + 1
    for i in range(nblock) :
      if nblock > 1 :
        print('Total blocks %i, Current block %i' % (nblock, i+1))
      if i == (nblock - 1) :
        ns = number % nspb
      else :
        ns = nspb
      si1 = nspb * i # sample index 1
      si2 = si1 + ns # sample index 2
      bi1 = index + si1 * csize # byte index 1
      bi2 = bi1 + ns * csize    # byte index 2
      cformat = endian + ctype * ns
      Value[si1:si2] = struct.unpack(cformat, data[bi1:bi2])

  if number == 1:
    return Value[0]
  else:
    return Value

def ibm2ieee(ibm_float):
  """
  i ibm_float : float, in IBM format
  o ieee_float : float, in IEEE format
  Convert float IBM to IEEE.
  """
  dividend = float(16**6)
  if ibm_float == 0:
    return 0.0
  istic, a, b, c = struct.unpack('>BBBB', ibm_float)
  if istic >= 128:
    sign = -1.0
    istic = istic - 128
  else:
    sign = 1.0
  mant = float(a<<16) + float(b<<8) + float(c)
  ieee_float = sign * 16**(istic-64) * (mant / dividend)
  return ieee_float

def getBytePerSample(SH):
  """
  i SH : dictionary, Segy binary file header
  o bps : integer, bytes per data sample
  """
  revno = getRevisionNumber(SH)
  dsf = SH["DataSampleFormat"]
  bps = SH_def["DataSampleFormat"]["bps"][revno][dsf]
  return bps

def getRevisionNumber(SH):
  """
  i SH : dictionary, Segy binary file header
  o revno : integer, Segy revision number
  Get Segy revision number.
  This method is adapted for Segy rev 1 released in May 2002.
  """
  revno = SH["SegyFormatRevisionNumber"]
  if SH["SegyFormatRevisionNumber"] == 0 :
    # Gocad
    #revno = 0
    # Some software only process binary file headers at 3200-3260.
    # They leave the revno number at 3501-3502 as zero, even when
    # the Segy file is actually rev1, i.e. data sample format is IEEE.
    revno = 1
  elif SH["SegyFormatRevisionNumber"] == 100 :
    revno = 1
  elif SH["SegyFormatRevisionNumber"] == 1 :
    # Petrel
    revno = 1
  elif SH["SegyFormatRevisionNumber"] == 256 :
    # SeisSpace and RokDoc
    revno = 1
  else :
    raise ValueError("Unknown revno number:", revno)
  return revno
