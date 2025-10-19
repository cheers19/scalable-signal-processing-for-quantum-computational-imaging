import numpy
import numba
import Haim_Camera_Functions as HCF
import Haim_File_Opening_Functions as HFOF
import Haim_Saving_Functions as HSF
import Haim_Filtering_Functions as HFF
import time

# Raw analysis option:
# 1 - without timing
# 2 - with timing
Raw_Analysis_Option = 2

# the folder in which the raw data is at
FolderNameRawData = r'/data/visitor/mi1495/id01/20240924/PROCESSED_DATA/Raw_Phi_45_58' + '/'
#FolderNameRawData = r'E:\Adi\Quantum_magnification\SSD1\PDC13' + '/'
FolderNameRawData = r'F:\Spring8_MacShare\PDC11' + '/'

# the folder in which the processed data will be saved
FolderNameAnalysed = r'/data/visitor/mi1495/id01/20240924/PROCESSED_DATA/Phi_45_58_Time_Hist' + '/'
FolderNameAnalysed = r'D:\Haim\ESRF Sep 2024\Data Analysis Tests' + '/'
CalibFolder = '/asap3/petra3/gpfs/p09/2023/data/11016211/shared/Calibration/'
CalibFolder = r'/data/visitor/mi1495/id01/20240924/SCRIPTS/Advapix_Calibration'+'/'
CalibFolder = r'D:\Haim\Working_Folder_Spring8_2024\CalibFiles'+'/'

# the index of the first and final files in the FolderNameRawData folder
StartFileIndex = 0
EndFileIndex = -1 # write -1 to load all the files till the end of the folder

AnalysedFileSufix = 'TimeWind_1us_PhotonMax_5_Energy_9_5___11_5.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_8_11___17_20_No_PDC.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_7_10___18_21_PDC_11_BotL_UpR.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_7_10___18_21_PDC_13_UpL_BotR.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_18_21___18_21_NoEnergyConservation_PDC_13_BotL_UpR.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_18_21___18_21_NoEnergyConservation_PDC_11_BotL_UpR.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_18_21___18_21_NoEnergyConservation_PDC_12_BotL_UpR.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_18_21___18_21_NoEnergyConservation_PDC_11_SameRegion.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_18_21___18_21_NoEnergyConservation_PDC_11_SameChip.pkl'
AnalysedFileSufix = 'TimeWind_1us_PhotonMax_10_Energy_11_16___18_21_NoEnergyConservation_PDC_11_SameRegion.pkl'





TimeWindow = 1000 # in nano seconds
Min_Number_of_Photons = 2
Max_Number_of_Photons = 10

# the following energy limits 1,2 and (x,y) coordinate limits 1,2 are used to exclude photons which are definitly not SPDC. for example if 2 photons arrive at the sime half of the camera they aren't SPDC

# EnergyMin1 and EnergyMax1 are the minimum and maximum energy in area 1 and the same applies for the index 2
EnergyMin1 = 8000
EnergyMax1 = 11000
EnergyMin2 = 17000
EnergyMax2 = 20000

# Ady's parameters
EnergyMin1 = 7000
EnergyMax1 = 10000
EnergyMin2 = 18000
EnergyMax2 = 21000

Pump_Energy = 28000
Tolerance = 2000

# NO energy conserving pairs
EnergyMin1 = 18000
EnergyMax1 = 21000
EnergyMin2 = 18000
EnergyMax2 = 21000
Tolerance = 14000

# NO energy conserving pairs
EnergyMin1 = 11000
EnergyMax1 = 16000
EnergyMin2 = 18000
EnergyMax2 = 21000
Tolerance = 14000

XMin1 = 0
XMax1 = 254
XMin2 = 257
XMax2 = 512

YMin1 = 0
YMax1 = 256
YMin2 = 80
YMax2 = 190


XMin1 = 160-100
XMax1 = 280-100
XMin2 = 425
XMax2 = 490

YMin1 = 70+50#70-50
YMax1 = 140+50#140-50
YMin2 = 80
YMax2 = 190



# Ady's parameters: bottom-left and upper-right
XMin1 = 160
XMax1 = 220
XMin2 = 456
XMax2 = 508

YMin1 = 32
YMax1 = 100
YMin2 = 108
YMax2 = 196

# same region
XMin1 = 160
XMax1 = 220
XMin2 = 160
XMax2 = 220

YMin1 = 108
YMax1 = 196
YMin2 = 108
YMax2 = 196
'''
# same chip
XMin1 = 160
XMax1 = 220
XMin2 = 160
XMax2 = 220

YMin1 = 8
YMax1 = 78
YMin2 = 90
YMax2 = 160
'''
'''
# Ady's parameters: upper-left and bottom-right
XMin1 = 140
XMax1 = 220
XMin2 = 456
XMax2 = 508

YMin1 = 108
YMax1 = 206
YMin2 = 32
YMax2 = 100
'''

Number_of_CPUs_to_Use = 14

numba.set_num_threads(Number_of_CPUs_to_Use)
if Raw_Analysis_Option == 1:
    HCF.Load_and_Process_All_Raw_Data(FolderNameRawData, StartFileIndex, EndFileIndex,CalibFolder, TimeWindow, Max_Number_of_Photons, Min_Number_of_Photons, FolderNameAnalysed, AnalysedFileSufix,EnergyMin1, EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2,YMin2, YMax2)
elif Raw_Analysis_Option == 2:
    # Load_and_Process_All_Raw_Data_with_Timings export the time tag of photon arrival to PDC_Events_TimeTagsALL. it saves the difference between 2 TOAs between 2 PDC photons
    SpectrumFig, PDC_Events_TimeTagsALL = HCF.Load_and_Process_All_Raw_Data_with_Timings(FolderNameRawData, StartFileIndex, EndFileIndex,CalibFolder, TimeWindow, Max_Number_of_Photons, Min_Number_of_Photons, FolderNameAnalysed, AnalysedFileSufix,EnergyMin1, EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2,YMin2, YMax2, Pump_Energy, Tolerance)
    # Prep_For_Enrgy_Histogram creates a histogram of PDC_Events_TimeTagsALL
    Hist, Hist_Bins = HFF.Prep_For_Enrgy_Histogram(PDC_Events_TimeTagsALL, 1.5625, 0, 1000)
    TimeThreshold = 10
    BinsTimeFilter = Hist_Bins<=10
    Counts_Below_Threshold = Hist[BinsTimeFilter]
    Counts_Above_Threshold = Hist[~BinsTimeFilter]
    MeanBackground = Counts_Above_Threshold.mean()
    Counts_Below_Threshold = Counts_Below_Threshold[(Counts_Below_Threshold-MeanBackground)>0]
    Signal = Counts_Below_Threshold.sum()
    Noise = MeanBackground*Counts_Below_Threshold.shape[0]
    print('Number of pairs above time threshold of {0:} ns is:{1:}'.format(TimeThreshold, numpy.count_nonzero(PDC_Events_TimeTagsALL>TimeThreshold)))
    print('Number of pairs below time threshold of {0:} ns is:{1:}'.format(TimeThreshold, numpy.count_nonzero(PDC_Events_TimeTagsALL <= TimeThreshold)))
    print('Mean number of pairs above threshold per bin: {0:}'.format(MeanBackground))
    print('Total PDC events: {0:}'.format(Signal))
    print('SNR is: {0:}'.format(Signal/Noise))



# ******* Experiments with Loading raw data ******
'''
import Haim_Advacam_Functions as Haim_Test
from matplotlib import pyplot as plt

TimeWindow = 1000
Min_Number_of_Photons = 2
Max_Number_of_Photons = 5

SortedRawData, PhotonCounter, BegInd, EndInd, Filtered_Photons_per_Frame, Filtered_Frames_Indexes, Filtered_RawData, PixelEnergy, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Combined_Photons, Photons_Coordinates,Photons_XY, BegInd_Photons, EndInd_Photons = Haim_Test.Load_and_Process_Single_File(r'E:\Adi\Quantum_magnification\SSD1\PDC13' + '/','PDC13_r00000.t3p',r'D:\Haim\Working_Folder_Spring8_2024\CalibFiles'+'/', TimeWindow, Max_Number_of_Photons,Min_Number_of_Photons)
BegInd,EndInd,PhotonCounter, Photons_per_Frame, Filtered_Frames_Indexes, Filtered_RawData, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Filtered_Photons_per_Frame = Haim_Test.Analyze_Pixels_and_Frames_Numba_New(SortedRawData, TimeWindow, Min_Number_of_Photons, Max_Number_of_Photons)
# Calc time differences of pixels of each photon and plot histogram
Photons_Time_Diff, Photons_Start_Time=Haim_Test.Pixels_Time_Differences(Filtered_RawData, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter,Filtered_Photons_per_Frame)
Bins = numpy.arange(-1.56/2,200,1.56)
plt.hist(Photons_Time_Diff,Bins)
'''