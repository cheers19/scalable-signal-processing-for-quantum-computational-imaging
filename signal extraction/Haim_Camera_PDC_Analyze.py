import Haim_Camera_Functions as HCF
import Haim_Filtering_Functions as HFF
import Haim_Saving_Functions as HSF
import Haim_File_Opening_Functions as HFOF
import numba

# Analysis options:
#       1- combine 3D arrays to single file
#       2- filter and combine 3D arrays to single file
#       3- Analyze single file for energy conservation and plot image
AnalysisOption = 3
# IMPORTANT!!!!!!!!!! - Should define folder and file name in the relevant AnalysisOption - !!!!!!!!!


StartFileIndex = 0
EndFileIndex = -1 # write -1 to load all the files till the end of the folder

Number_of_CPUs_to_Use = 8

#PDC parameters:
EnergyMin1 = 8000
EnergyMax1 = 11000
EnergyMin2 = 17000
EnergyMax2 = 20000


PumpEnergy = 28000
EnergyTolerance = 2000

XMin1 = 160-100
XMax1 = 280-100
XMin2 = 425
XMax2 = 490

YMin1 = 70-50
YMax1 = 140-50
YMin2 = 80
YMax2 = 190


# idler in 4nd square (x1,y1)
EnergyMin1 = 8900
EnergyMax1 = 11900
EnergyMin2 = 8900
EnergyMax2 = 11900

Pump_Energy = 20856
Tolerance = 2000

XMin2 = 282
XMax2 = 502

YMin2 = 10
YMax2 = 230

XMin1 = 128
XMax1 = 251
YMin1 = 128
YMax1 = 251

# wider energy conservation and including the last raw and column of pixels
XMin1 = 0
XMax1 = 255
XMin2 = 256
XMax2 = 512

YMin1 = 0
YMax1 = 255
YMin2 = 0
YMax2 = 255

EnergyMin1 = 6500
EnergyMax1 = 14500
EnergyMin2 = 6500
EnergyMax2 = 14500

Pump_Energy = 20856
Tolerance = 8144

numba.set_num_threads(Number_of_CPUs_to_Use)

# First version - combines all the data before analyzing PDC. Change to analyzing each file
#Total_Data = HCF.Load_and_Combine_3D_Array_Data(FolderNameAnalysed, StartFileIndex, EndFileIndex)
if AnalysisOption == 1:
    FolderNameAnalysed = r'D:\Haim\ESRF Sep 2024\Raw_Analysis_Output' + '/'
    FolderNameAnalysed = r'D:\Haim\ESRF Sep 2024\raw_analysis_PDC13_ESRF24' + '/'

    FolderNameOutput = r'D:\Haim\ESRF Sep 2024\Data Analysis Tests\ESRF Sep24' + '/'
    FilteredFileName = 'HaimandEdy'
    FilteredFileName = 'combinedPDC13_Energy_6p5_14p5_IncludingCorners_ESRF24'

    #FolderNameAnalysed = r'D:\Haim\Working_Folder_Spring8_2024\Data_Analysis_Tests\PDC12_First_Step' + '\\'
    #FolderNameOutput = r'D:\Haim\Working_Folder_Spring8_2024\Data_Analysis_Tests\PDC12_Second_Step' + '\\'
    #FilteredFileName = 'PDC_11plus12_Combined'
    FileList = HFOF.ListFilesInFolder(FolderNameAnalysed)
    FileList.sort()
    if (EndFileIndex < 0) | (EndFileIndex > len(FileList)):
        EndFileIndex = len(FileList)
    Combined_Data = HCF.Load_and_Combine_3D_Array_Data(FolderNameAnalysed, StartFileIndex, EndFileIndex)
    HSF.SaveDataToFile(Combined_Data,{'Folder Name': FolderNameOutput, 'File Name': FilteredFileName + '_{}_{}'.format(StartFileIndex,EndFileIndex)})

elif AnalysisOption == 2:
    FolderNameAnalysed = r'/data/visitor/mi1495/id01/20240924/PROCESSED_DATA/PDC6' + '/'
    FolderNameOutput = r'/data/visitor/mi1495/id01/20240924/PROCESSED_DATA/PDC6_3D_Array' + '/'
    FilteredFileName = 'PDC6_Combined_Wide_Band'

    Combined_Filtered_Data = HCF.Load_Analyize_and_Combine_3D_PDC_Data(FolderNameAnalysed, StartFileIndex, EndFileIndex,EnergyMin1,EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2, YMin2, YMax2,PumpEnergy, EnergyTolerance)
    FileList = HFOF.ListFilesInFolder(FolderNameAnalysed)
    FileList.sort()
    if (EndFileIndex < 0) | (EndFileIndex > len(FileList)):
        EndFileIndex = len(FileList)
    HSF.SaveDataToFile(Combined_Filtered_Data,{'Folder Name': FolderNameOutput, 'File Name': FilteredFileName + '_{}_{}'.format(StartFileIndex,EndFileIndex)})

elif AnalysisOption == 3:
    FolderNameAnalysed = r'D:\Haim\ESRF Sep 2024\Data Analysis Tests' + '/'
    FolderNameAnalysed = r'D:\Haim\ESRF Sep 2024\Data Analysis Tests\ESRF Sep24' + '/'

    FilteredFileName = 'HaimandEdy_0_4008'
    FilteredFileName = 'combinedPDC13_Energy_6p5_14p5_IncludingCorners_ESRF24_0_273'

    Loaded_Data = HFOF.LoadDataFromFile({'Folder Name': FolderNameAnalysed, 'File Name': FilteredFileName})
    #Combined_Filtered_Data = HCF.Load_Analyize_and_Combine_3D_PDC_Data(FolderNameAnalysed, StartFileIndex, EndFileIndex,EnergyMin1, EnergyMax1, XMin1, XMax1, YMin1,YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2,YMin2, YMax2, PumpEnergy, EnergyTolerance)
    Energy_and_Region_Filtration = HFF.Timing_Of_Function(HFF.Filter_Pairs_Array_Numba,'energy and regions filtration',True,Loaded_Data,EnergyMin1,EnergyMax1,XMin1,XMax1,YMin1,YMax1,EnergyMin2,EnergyMax2,XMin2,XMax2,YMin2,YMax2,False)[0]
    Energy_Conservation = HFF.Generate_EnergyConserving_Array(Energy_and_Region_Filtration, PumpEnergy, EnergyTolerance)[0]
    Figs = HCF.PDC_Ploting(Energy_Conservation,Binning=4,ROIX=[300,400],ROIY=[50, 150])
    Figs[1][1].set_title('In the right {}-{} eV'.format(EnergyMin2,EnergyMax2))
    Figs[1][1].images[0].set_clim(1,20)
#Figs2 = HCF.PDC_Ploting(Total_Data[:1000000,:,:],Binning=2,ROIX=[300,400],ROIY=[50, 150])
#Figs2[1].set_title('Without energy or spatial filtration')

#Use the function HCF.Prep_For_Enrgy_Histogram and EPF.Custom_Plot_7
