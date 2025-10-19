import numpy
import time
import Haim_File_Opening_Functions as HFOF
import Haim_Filtering_Functions as HFF
import Haim_Ploting_Functions as HPF
import Haim_Saving_Functions as HSF
import numba
import sys

# Functions:
def ScanArea(DataArray, Energy_Region1, Energy_Region2, PumpEnergy, BinningSteps, Fixed_Area_Coordinates,Fixed_Area_Size, Scan_Rows_Binned, Scan_Columns_Binned, Scan_Steps):
    """
    A function that scans over an area
    Parameters:
        Energy_Region1 - numpy array with 2 cells with min\max energy in region 1. Region 1 is the fixed region.
        Energy_Region2 - numpy array with 2 cells with min\max energy in region 2. Region 2 is the scanning region.
        BinningSteps - numpy array with 2 cells. Binned number of rows and columns.
        Fixed_Area_Coordinates - numpy array with 2 cells. Row and column of fixed area in binned image.
        Fixed_Area_Size - numpy array with 2 cells. Number of row and number of columns.
        Scan_Rows_Binned - numpy array with 2 cells. Start and end rows for scan in binned image. NOT includes the end value.
        Scan_Columns_Binned - numpy array with 2 cells. Start and end columns for scan in binned image. NOT includes the end value.
        Scan_Steps - numpy array with 2 cells. Number of rows and number of columns in each step.
    """
    Scan_Row_Step = Scan_Steps[0]
    Scan_Column_Step = Scan_Steps[1]
    Scan_Begin_Row = Scan_Rows_Binned[0] * BinningSteps[0]
    Scan_End_Row = Scan_Rows_Binned[1] * BinningSteps[0]
    Scan_Begin_Column = Scan_Columns_Binned[0] * BinningSteps[1]
    Scan_End_Column = Scan_Columns_Binned[1] * BinningSteps[1]
    Rows_Scan = numpy.arange(start=Scan_Begin_Row,step=Scan_Row_Step,stop=Scan_End_Row,dtype=numpy.uint16)
    Columns_Scan = numpy.arange(start=Scan_Begin_Column,step=Scan_Column_Step,stop=Scan_End_Column,dtype=numpy.uint16)

    Fixed_Begin_Row = Fixed_Area_Coordinates[0] * BinningSteps[0]
    Fixed_End_Row = Fixed_Begin_Row +  Fixed_Area_Size[0] - 1 # Minus 1 at the end because the filtering function takes also the last row
    Fixed_Begin_Column = Fixed_Area_Coordinates[1] * BinningSteps[1]
    Fixed_End_Column = Fixed_Begin_Column +  Fixed_Area_Size[1] -1 # Minus 1 at the end because the filtering function takes also the last column
    ResultsList = list()
    print('Fixed region is:\n\tRows {}-{}\n\tColumns{}-{}\n'.format(Fixed_Begin_Row,Fixed_End_Row,Fixed_Begin_Column,Fixed_End_Column))
    for Current_Row in Rows_Scan:
        Current_Row_End = Current_Row+Scan_Row_Step-1 # Minus 1 at the end because the filtering function takes also the last row
        print('Scanning rows {}-{} (inclusive)'.format(Current_Row,Current_Row_End))
        for Current_Column in Columns_Scan:
            Current_Column_End = Current_Column+Scan_Column_Step-1 # Minus 1 at the end because the filtering function takes also the last column
            print('Scanning columns {}-{} (inclusive)'.format(Current_Column,Current_Column_End),end='\r')
            FilteredArray_First = HFF.Filter_Pairs_Array_Numba(DataArray, Energy_Region1[0], Energy_Region1[1], Fixed_Begin_Column, Fixed_End_Column, Fixed_Begin_Row, Fixed_End_Row,Energy_Region2[0], Energy_Region2[1], Current_Column, Current_Column_End, Current_Row, Current_Row_End, False)[0]
            if type(FilteredArray_First) != int:
                FilteredArray_Second = HFF.Generate_EnergyConserving_Array(FilteredArray_First, PumpEnergy, 250)[0]
                if type(FilteredArray_Second) != int:
                    ResultsList.append(FilteredArray_Second)
        print('')


            # Testing
            # DataArray[Current_Row,Current_Column] = int(str(Current_Row)+str(Current_Column))
    return ResultsList

# INPUT: Choose what script do.
#        1 - single fixed area and show results
#        2 - scan with fixed area
ScriptOption = 2
# INPUT: Choose if to save result of single fixed area
SaveSingleScan = False
# INPUT: Choose how many CPUs to use
Number_of_CPUs_to_Use = 10
numba.set_num_threads(Number_of_CPUs_to_Use)

# INPUT: folder and file name of 3D numpy array with combined and filtered data from script 'Edy_Unite_Recombined_List'
Folder = r'D:\Haim\ESRF Sep 2024\Data Analysis Tests' + '/'
File = 'HaimandEdy_0_4008' # Without object
# File = 'PDC_3D_Array_PDC044-PDC046_12-11-2023' # With object

# INPUT: scan values
Energy_Region1 = numpy.array([17000,20000],dtype=numpy.uint16) # Fixed region.
Energy_Region2 = numpy.array([8000,11000],dtype=numpy.uint16) # Scanning region.
Binning_Steps = numpy.array([1,1],dtype=numpy.uint8) # In real pixels units
Fixed_Area_Coordinates = numpy.array([18,22],dtype=numpy.uint16) # Rows, Columns in BINNED image
Fixed_Area_Size = numpy.array([20,20],dtype=numpy.uint16) # In real pixels units
Fixed_Area_Scan_Row_Range = numpy.array([0,256],dtype=numpy.uint16) # Fixed area scan row range in binned units (Binning_Steps)
Fixed_Area_Scan_Column_Range = numpy.array([380,512],dtype=numpy.uint16) # Fixed area scan column range in binned units (Binning_Steps)
Fixed_Area_Scan_Step_Binned = numpy.uint8(20) # Scan step with fixed area in binned pixels units (Binning_Steps). It is how many times there is Binning_Steps in Fixed_Area_Size. It is 1 if they of the same size
Scan_Rows_Binned = numpy.array([0,256],dtype=numpy.uint16) # Scanning area rows range in binned pixels units according to Binning_Steps
Scan_Columns_Binned = numpy.array([0,300],dtype=numpy.uint16) # Scanning area columns range in binned pixels units according to Binning_Steps
Scan_Steps = numpy.array([40,40],dtype=numpy.uint16) # Scanning area step size in real pixels units

PumpEnergy = 28000
EnergyTolerance = 2000

# Load 3D numpy array:
DescriptionText = 'loading 3D array'
ArrayData = HFF.Timing_Of_Function(HFOF.LoadDataFromFile,DescriptionText,True,{'Folder Name':Folder, 'File Name':File})

ScanInfo = dict()
ScanInfo['Energy_Region1'] = Energy_Region1
ScanInfo['Energy_Region2'] = Energy_Region2
ScanInfo['PumpEnergy'] = PumpEnergy
ScanInfo['Binning_Steps'] = Binning_Steps
ScanInfo['Fixed_Area_Coordinates'] = Fixed_Area_Coordinates
ScanInfo['Fixed_Area_Size'] = Fixed_Area_Size
ScanInfo['Scan_Rows_Binned'] = Scan_Rows_Binned
ScanInfo['Scan_Columns_Binned'] = Scan_Columns_Binned
ScanInfo['Scan_Steps'] = Scan_Steps
ScanInfo['EnergyTolerance'] = EnergyTolerance
ScanInfo['Fixed_Area_Scan_Row_Range'] = Fixed_Area_Scan_Row_Range
ScanInfo['Fixed_Area_Scan_Column_Range'] = Fixed_Area_Scan_Column_Range
ScanInfo['Fixed_Area_Scan_Step_Binned'] = Fixed_Area_Scan_Step_Binned

if ScriptOption == 1:
    # Perform scanning:
    StartTime = time.perf_counter()
    ScanResults = HFF.ScanArea(ArrayData,Energy_Region1,Energy_Region2,PumpEnergy,Binning_Steps,Fixed_Area_Coordinates,Fixed_Area_Size,Scan_Rows_Binned,Scan_Columns_Binned,Scan_Steps,EnergyTolerance)
    EndTime = time.perf_counter()
    print('Scan took: {0:.3f} seconds'.format(EndTime-StartTime))
    # Saving results to file
    if SaveSingleScan:
        ParamWrite = dict()
        ParamWrite['Folder Name'] = '/asap3/petra3/gpfs/p09/2022/data/11013046/processed/Scans_for_Radii/'
        ParamWrite['File Name'] = 'Fixed_Region_{}-{}eV_Point_Test7'.format(Energy_Region1[0],Energy_Region1[1])
        DataToSave = dict()

        DataToSave['Results'] = [ScanResults] # Turn even single fixed square to a list. For future data analysis

        DataToSave['Scan Info'] = ScanInfo
        DataToSave['Time'] = EndTime - StartTime
        HSF.SaveDataToFile(DataToSave, ParamWrite)
    # Plotting number of pairs in each square
    SNR_Plot = HFF.Calc_Background(ScanResults)
    HPF.Custom_Plot_6(SNR_Plot, XLabel='Arbitrary index of small areas on the left', PlotSize=[0.2, 0.15, 0.7, 0.8])
    # Showing image
    BinnedImage = HFF.Pairs_List_to_Image(ScanResults)
    AreaAxes = HPF.Custom_Plot_5(BinnedImage, CLimMax=6, CLimMin=0, PlotSize=[0.2, 0.15, 0.7, 0.8], Title='Scaning results')[1]

    Background = 4

    # Image with correct count rate (by subtracting background)
    # This part works only for squares of 4 over 8 pixels
    NewBinnedImage = HFF.SubtractFromAreas(ScanResults, Const=Background)
    NewAxes = HPF.Custom_Plot_5(NewBinnedImage, CLimMax=10, CLimMin=0, PlotSize=[0.2, 0.15, 0.7, 0.8],Title='Without background')[1]

    # Selecting only squares with number of pairs above the background
    RingParts = list()
    for Sub_Ind in range(len(ScanResults)):
        if ScanResults[Sub_Ind].shape[0] > Background:
            RingParts.append(ScanResults[Sub_Ind])

    # Calculating centers of intensity and drawing a line
    BinnedImage = HFF.Pairs_List_to_Image(RingParts, VBin=4, HBin=8)
    AreaAxes = HPF.Custom_Plot_5(BinnedImage, CLimMax=6, CLimMin=0, PlotSize=[0.2, 0.15, 0.7, 0.8], Title='Areas above background')[1]
    X1,Y1,X2,Y2 = HFF.Calc_Center_Mass(BinnedImage,Center=16)
    HPF.Plot_Line_Between_Points(X1,Y1, X2,Y2,AreaAxes)

elif ScriptOption == 2:
    # Scanning with fixed region:
    ScanResults = list()
    # Check input while calling script and updating scan parameters:
    UserInput = sys.argv
    if len(UserInput) > 1 :
        Fixed_Area_Scan_Row_Range[0] = numpy.uint16(UserInput[1])
        Fixed_Area_Scan_Row_Range[1] = numpy.uint16(UserInput[2])
        Fixed_Area_Scan_Column_Range[0] = numpy.uint16(UserInput[3])
        Fixed_Area_Scan_Column_Range[1] = numpy.uint16(UserInput[4])
        Fixed_Area_Scan_Step_Binned = numpy.uint8(UserInput[5])
        ScanInfo['Fixed_Area_Scan_Row_Range'] = Fixed_Area_Scan_Row_Range
        ScanInfo['Fixed_Area_Scan_Column_Range'] = Fixed_Area_Scan_Column_Range
        ScanInfo['Fixed_Area_Scan_Step_Binned'] = Fixed_Area_Scan_Step_Binned
    if len(UserInput) > 6:
        Energy_Region1[0] = numpy.uint16(UserInput[6])
        Energy_Region1[1] = numpy.uint16(UserInput[7])
        Energy_Region2[0] = numpy.uint16(UserInput[8])
        Energy_Region2[1] = numpy.uint16(UserInput[9])
        ScanInfo['Energy_Region1'] = Energy_Region1
        ScanInfo['Energy_Region2'] = Energy_Region2

    StartTime = time.perf_counter()
    for Fix_Area_Row_Step in range(Fixed_Area_Scan_Row_Range[0],Fixed_Area_Scan_Row_Range[1],Fixed_Area_Scan_Step_Binned): # NOT includes the last value. The range is in binned steps (Binning_Steps). If Fixed_Area_Size larger than binning, should increase the step
        for Fix_Area_Column_Step in range(Fixed_Area_Scan_Column_Range[0],Fixed_Area_Scan_Column_Range[1],Fixed_Area_Scan_Step_Binned): # NOT includes the last value. The range is in binned steps (Binning_Steps). If Fixed_Area_Size larger than binning, should increase the step
            Fixed_Area_Coordinates = numpy.array([Fix_Area_Row_Step, Fix_Area_Column_Step], dtype=numpy.uint16)
            ScanResults.append(ScanArea(ArrayData,Energy_Region1,Energy_Region2,PumpEnergy,Binning_Steps,Fixed_Area_Coordinates,Fixed_Area_Size,Scan_Rows_Binned,Scan_Columns_Binned,Scan_Steps))
    EndTime = time.perf_counter()
    print('Scan took: {0:.3f} seconds'.format(EndTime - StartTime))

    # ****** Input: folder where to save the result: ********
    ParamWrite = dict()
    ParamWrite['Folder Name'] = r'D:\Haim\ESRF Sep 2024\Data Analysis Tests'+'/'

    # ****** Input: file name of results: *******
    ParamWrite['File Name'] = 'Area_Scan_for_Ring_Large_Steps_{}_{}eV'.format(Energy_Region1[0],Energy_Region1[1])
    if len(UserInput) > 10:
        ParamWrite['File Name'] += '_{}'.format(UserInput[10])
    DataToSave = dict()
    DataToSave['Results'] = ScanResults
    DataToSave['Scan Info'] = ScanInfo
    DataToSave['Time'] = EndTime-StartTime
    HSF.SaveDataToFile(DataToSave,ParamWrite)

#temp1 = numpy.vstack(ScanResults[0])
#temp2 = HFF.Prepare_For_Plotting_Array_V2(temp1)
#temp3 = HFF.BinFrame(temp2, {'vertical binning': 4, 'horizontal binning': 8})
#HPF.Custom_Plot_5(temp3,CLimMax=7,PlotSize=[0.2,0.15,0.7,0.8])


# Show frame with large area filtration:
'''
FilteredArray_First = HFF.Filter_Pairs_Array_Numba(ArrayData, Energy_Region2[0], Energy_Region2[1], 0, 131, 0, 132,Energy_Region1[0],Energy_Region1[1], 132, 263, 0, 132,False)[0]
FilteredArray_Second = HFF.Generate_EnergyConserving_Array(FilteredArray_First,PumpEnergy, 250)[0]

TestFrame = HFF.Prepare_For_Plotting_Array_V2(FilteredArray_Second)
TestBinnedFrame = HFF.BinFrame(TestFrame, {'vertical binning': 4, 'horizontal binning': 8})
'''
#HPF.Custom_Plot_5(TestBinnedFrame,CLimMax=1200,PlotSize=[0.2,0.15,0.7,0.8])
