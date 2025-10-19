import numpy
import time
import numba
import pandas
from numba import jit,prange,vectorize,guvectorize,uint8,uint16,boolean,uint32,int64,float64,float32
import Haim_Ploting_Functions as HPF
import Haim_Saving_Functions as HSF
import Haim_File_Opening_Functions as HFOF
import Haim_Filtering_Functions as EFF

def Load_and_Process_Single_File(FolderNameRawData, FileNameRawData,CalibFolder, TimeWindow, Max_Number_of_Photons, Min_Number_of_Photons):
    print('Loading file: ...{}'.format(FolderNameRawData[-20:]+FileNameRawData))
    StartTime = time.perf_counter()

    # Test - faster data read:
    if FileNameRawData.rsplit('.',maxsplit=1)[1]=='t3pa':
        # RawData = numpy.loadtxt(FolderNameRawData + FileNameRawData, dtype=numpy.uint32, skiprows=1, usecols=(1, 2, 3, 4, 5))
        RawData_pandas = pandas.read_csv(FolderNameRawData + FileNameRawData, sep='\t', dtype=numpy.uint32, skiprows=0, engine='pyarrow',usecols=['Matrix Index', 'ToA', 'ToT', 'FToA', 'Overflow'])
        RawData = RawData_pandas.values
        print('Finished loading file. It took {0:.2f} sec.'.format(time.perf_counter()-StartTime))
    elif FileNameRawData.rsplit('.',maxsplit=1)[1]=='t3p':
        RawData = Read_Advacam_Binary(FolderNameRawData + FileNameRawData)
        print('Finished loading file. It took {0:.2f} sec.'.format(time.perf_counter() - StartTime))
    CounterResInd = Find_Counter_Reset(RawData, 2**28)
    SortedRawData = SortRawData(RawData,CounterResInd)

    BegInd,EndInd,PhotonCounter, Photons_per_Frame, Filtered_Frames_Indexes, Filtered_RawData, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Filtered_Photons_per_Frame = Analyze_Pixels_and_Frames_Numba_New(SortedRawData, TimeWindow, Min_Number_of_Photons, Max_Number_of_Photons)

    PixelEnergy = HFF.Timing_Of_Function(Convert_to_Energy,'converting ToT to energy',True,Filtered_RawData,0,CalibFolder)

    Combined_Photons, Photons_Coordinates, BegInd_Photons, EndInd_Photons = HFF.Timing_Of_Function(Combine_SinglePhoton_InFrame_Numba, 'combining pixels to photons', True, Filtered_RawData, Filtered_PhotonCounter, PixelEnergy, Filtered_BegInd, Filtered_EndInd, Filtered_Photons_per_Frame)

    Photons_XY = HFF.Timing_Of_Function(Matrix_Index_to_XY_Numba,'converting to XY values',True, Photons_Coordinates)
    return SortedRawData, PhotonCounter, BegInd, EndInd, Filtered_Photons_per_Frame, Filtered_Frames_Indexes,Filtered_RawData,PixelEnergy, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Combined_Photons, Photons_Coordinates, Photons_XY, BegInd_Photons, EndInd_Photons

def Load_Raw_Data(FolderName, StartFileIndex, EndFileIndex):
    FileList = HFOF.ListFilesInFolder(FolderName)
    HFOF.Advacam_Prepare_Raw_ASCII_List(FileList)

    if (EndFileIndex < 0) | (EndFileIndex > len(FileList)):
        EndFileIndex = len(FileList)

    for FileInd in range(StartFileIndex, EndFileIndex):
        FileName = FileList[FileInd]
        if FileName.rsplit('.',maxsplit=1)[1]=='t3pa':
            # RawData = numpy.loadtxt(FolderNameRawData + FileNameRawData, dtype=numpy.uint32, skiprows=1, usecols=(1, 2, 3, 4, 5))
            RawData_pandas = pandas.read_csv(FolderName + FileName, sep='\t', dtype=numpy.uint32, skiprows=0, engine='pyarrow',usecols=['Matrix Index', 'ToA', 'ToT', 'FToA', 'Overflow'])
            RawData = RawData_pandas.values
        elif FileName.rsplit('.',maxsplit=1)[1]=='t3p':
            RawData = Read_Advacam_Binary(FolderName + FileName)
        if FileInd == StartFileIndex:            
            CounterResInd = Find_Counter_Reset(RawData, 2**28)
            SortedRawData = SortRawData(RawData,CounterResInd)
            TotalRawData = SortedRawData
            TestRawData = RawData
            print('First file {} pixels'.format(RawData.shape[0]))
        else:
            CounterResInd = Find_Counter_Reset(RawData, 2**28)
            SortedRawData = SortRawData(RawData,CounterResInd)
            
            TotalRawData = numpy.vstack((TotalRawData,SortedRawData))
            TestRawData = numpy.vstack((TestRawData, RawData))
            print('adding {} pixels to total of {} pixels'.format(RawData.shape,TotalRawData.shape))
        

    return TotalRawData, TestRawData

def Process_RawData(RawData,CalibFolder, TimeWindow, Max_Number_of_Photons, Min_Number_of_Photons):
    """
    A function that recieves raw data with 5 columns (matrix index, toa, tot,ftoa,overflow) and returns photons with energies and x,y coordinates
    """
    print('Processing raw data with {0:,} pixels...'.format(RawData.shape[0]))
    StartTime = time.perf_counter()

    # Test - faster data read:

    CounterResInd = Find_Counter_Reset(RawData, 2**28)
    SortedRawData = SortRawData(RawData,CounterResInd)

    BegInd,EndInd,PhotonCounter, Photons_per_Frame, Filtered_Frames_Indexes, Filtered_RawData, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Filtered_Photons_per_Frame = Analyze_Pixels_and_Frames_Numba_New(SortedRawData, TimeWindow, Min_Number_of_Photons, Max_Number_of_Photons)

    PixelEnergy = HFF.Timing_Of_Function(Convert_to_Energy,'converting ToT to energy',True,Filtered_RawData,0,CalibFolder)

    Combined_Photons, Photons_Coordinates, BegInd_Photons, EndInd_Photons = HFF.Timing_Of_Function(Combine_SinglePhoton_InFrame_Numba, 'combining pixels to photons', True, Filtered_RawData, Filtered_PhotonCounter, PixelEnergy, Filtered_BegInd, Filtered_EndInd, Filtered_Photons_per_Frame)

    Photons_XY = HFF.Timing_Of_Function(Matrix_Index_to_XY_Numba,'converting to XY values',True, Photons_Coordinates)
    return SortedRawData, PhotonCounter, BegInd, EndInd, Filtered_Photons_per_Frame, Filtered_Frames_Indexes,Filtered_RawData,PixelEnergy, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Combined_Photons, Photons_Coordinates, Photons_XY, BegInd_Photons, EndInd_Photons

def Convert_ToA_to_nanosec(ToA,FToA):
    Time =  ToA.astype(numpy.float32)*25-FToA.astype(numpy.float32)*(25/16)
    return Time

def Convert_ASCII_to_Binary():
    pass

def Read_Advacam_Binary(FileName):
    Advapix_dtype = numpy.dtype([('MatInd', '<u4'), ('ToA', '<u8'), ('OverF', 'u1'), ('FToA', 'u1'), ('ToT', '<u2')])
    Data = numpy.fromfile(FileName,Advapix_dtype, sep="")
    RawData = numpy.zeros((Data.shape[0],5),dtype=numpy.uint32)
    RawData[:, 0] = Data['MatInd']
    RawData[:, 1] = Data['ToA']
    RawData[:, 2] = Data['ToT']
    RawData[:, 3] = Data['FToA']
    RawData[:, 4] = Data['OverF']
    return RawData

@jit(nopython=True, nogil=True, parallel=True)
def Convert_2D_to_3D_array(PhotonsEnergy, XY_Data, BegInd, EndInd):
    """
    A function that converts 2D arrays of x,y coordinates and energies of photons to 3D array
    return a 3D with x,y,energy.
    Energy is in eV
    """
    Max_Frame_Size = numpy.max(EndInd-BegInd)
    Number_of_Frames = EndInd.shape[0]
    Out_3D = numpy.zeros((Number_of_Frames,Max_Frame_Size,3),numpy.float32)

    for FrameInd in prange(Number_of_Frames):
        FrameLength = EndInd[FrameInd]-BegInd[FrameInd]
        Out_3D[FrameInd,:FrameLength,:2] = XY_Data[BegInd[FrameInd]:EndInd[FrameInd],:]
        Out_3D[FrameInd,:FrameLength,2] = PhotonsEnergy[BegInd[FrameInd]:EndInd[FrameInd]]*1000
    return Out_3D.astype(numpy.uint32)

def Load_and_Combine_3D_Array_Data(FolderNameAnalysed, StartFileIndex, EndFileIndex):
    FileList = HFOF.ListFilesInFolder(FolderNameAnalysed)
    FileList.sort()

    if (EndFileIndex < 0) | (EndFileIndex > len(FileList)):
        EndFileIndex = len(FileList)

    Total_Data = HFOF.LoadDataFromFile({'Folder Name':FolderNameAnalysed,'File Name':FileList[StartFileIndex]})
    print('Loaded {}'.format(FolderNameAnalysed[-20:]+FileList[StartFileIndex]))
    for FileInd in range(StartFileIndex+1,EndFileIndex):
        Loaded_Data = HFOF.LoadDataFromFile({'Folder Name':FolderNameAnalysed,'File Name':FileList[FileInd]})
        print('Loaded {}'.format(FolderNameAnalysed[-20:] + FileList[FileInd]))
        if Loaded_Data.shape[1]>Total_Data.shape[1]:
            New_Total_Data = numpy.zeros((Total_Data.shape[0],Loaded_Data.shape[1],3),dtype=numpy.uint32) # Change from uint16
            New_Total_Data[:,:Total_Data.shape[1],:] = Total_Data[:,:,:]
            Total_Data = numpy.vstack((New_Total_Data, Loaded_Data))
        elif Loaded_Data.shape[1]<Total_Data.shape[1]:
            New_Loaded_Data = numpy.zeros((Loaded_Data.shape[0], Total_Data.shape[1], 3), dtype=numpy.uint32) # Change from uint16
            New_Loaded_Data[:,:Loaded_Data.shape[1],:] = Loaded_Data[:,:,:]
            Total_Data = numpy.vstack((Total_Data,New_Loaded_Data))
        else:
            Total_Data = numpy.vstack((Total_Data,Loaded_Data))

    return Total_Data

def Load_Analyize_and_Combine_3D_PDC_Data(FolderNameAnalysed, StartFileIndex, EndFileIndex,EnergyMin1,EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2, YMin2, YMax2,PumpEnergy, EnergyTolerance):
    """
    A function that loads files with data in the form of 3D numpy arrays, does first filtration with function  Filter_Pairs_Array_Numba and combines to one large 3D array.
    """
    FileList = HFOF.ListFilesInFolder(FolderNameAnalysed)
    FileList.sort()

    if (EndFileIndex < 0) | (EndFileIndex > len(FileList)):
        EndFileIndex = len(FileList)

    Loaded_Data = HFOF.LoadDataFromFile({'Folder Name':FolderNameAnalysed,'File Name':FileList[StartFileIndex]})
    Energy_and_Region_Filtration = HFF.Timing_Of_Function(HFF.Filter_Pairs_Array_Numba, 'energy and regions filtration', True, Loaded_Data, EnergyMin1,EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2, YMin2, YMax2,False)[0]
    #Energy_Conservation = HFF.Generate_EnergyConserving_Array(Energy_and_Region_Filtration, PumpEnergy, EnergyTolerance)[0]
    #Total_PDC_Data = Energy_Conservation
    Total_PDC_Data = Energy_and_Region_Filtration
    print('Loaded {}'.format(FolderNameAnalysed[-20:]+FileList[StartFileIndex]))
    for FileInd in range(StartFileIndex+1,EndFileIndex):
        Loaded_Data = HFOF.LoadDataFromFile({'Folder Name':FolderNameAnalysed,'File Name':FileList[FileInd]})
        Energy_and_Region_Filtration = HFF.Timing_Of_Function(HFF.Filter_Pairs_Array_Numba, 'energy and regions filtration', True, Loaded_Data,EnergyMin1, EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2,YMin2, YMax2, False)[0]
        NewRowNumber = Energy_and_Region_Filtration.shape[1]
        if NewRowNumber > Total_PDC_Data.shape[1]:

            New_Total_Data = numpy.zeros((Total_PDC_Data.shape[0],NewRowNumber,3),dtype=numpy.uint32)
            New_Total_Data[:,:Total_PDC_Data.shape[1],:] = Total_PDC_Data[:,:,:]
            Total_PDC_Data = numpy.vstack((New_Total_Data, Energy_and_Region_Filtration))
        elif NewRowNumber < Total_PDC_Data.shape[1]:
            New_Loaded_Data = numpy.zeros((Energy_and_Region_Filtration.shape[0], Total_PDC_Data.shape[1], 3), dtype=numpy.uint32)
            New_Loaded_Data[:,:NewRowNumber,:] = Energy_and_Region_Filtration[:,:,:]
            Total_PDC_Data = numpy.vstack((Total_PDC_Data,New_Loaded_Data))
        else:
            Total_PDC_Data = numpy.vstack((Total_PDC_Data,Energy_and_Region_Filtration))

        #Energy_Conservation = HFF.Generate_EnergyConserving_Array(Energy_and_Region_Filtration, PumpEnergy, EnergyTolerance)[0]
        #Total_PDC_Data = numpy.vstack((Total_PDC_Data,Energy_Conservation))
        print('Loaded and analyzed {}'.format(FolderNameAnalysed[-20:] + FileList[FileInd]))

    return Total_PDC_Data

def Load_and_Process_All_Raw_Data(FolderNameRawData, StartFileIndex, EndFileIndex,CalibFolder, TimeWindow, Max_Number_of_Photons, Min_Number_of_Photons, FolderNameAnalysed, AnalysedFileSufix,EnergyMin1, EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2,YMin2, YMax2):
    # The file has 6 columns: Index, Matrix Index, ToA, ToT, FToA, Overflow

    FileList = HFOF.ListFilesInFolder(FolderNameRawData)
    HFOF.Advacam_Prepare_Raw_ASCII_List(FileList) # Sorts alphabetically file names and remove files that are not raw data files

    if (EndFileIndex < 0) | (EndFileIndex > len(FileList)): # Sets the index of last file if user entered -1 or the last index was larger than number of files in folder
        EndFileIndex = len(FileList)

    LoopTimer = time.perf_counter()
    TotalTimer = time.perf_counter()

    for FileInd in range(StartFileIndex, EndFileIndex):
        print('\n')
        # Converts raw data to photonos devided to frames with length TimeWindow
        SortedRawData, PhotonCounter, BegInd, EndInd, Filtered_Photons_per_Frame, Filtered_Frames_Indexes, Filtered_RawData, PixelEnergy, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Combined_Photons, Photons_Coordinates,Photons_XY, BegInd_Photons, EndInd_Photons = EFF.Timing_Of_Function(Load_and_Process_Single_File,'single raw file total analysis',True,FolderNameRawData,FileList[FileInd],CalibFolder,TimeWindow,Max_Number_of_Photons,Min_Number_of_Photons)
        StartTime = time.perf_counter()
        # Converting photons  data to 3D array that the function Filter_Pairs_Array_Numba expects
        # - Combined_Photons: a list of all photon's energies
        # - Photons_XY: a list of same length where for each energy written the corresponding X,Y coordinates of the photon
        # BegInd_Photons: an array of same length containing the raw indices for which a frame begins
        Data_in_3D_Array = Convert_2D_to_3D_array(Combined_Photons, Photons_XY, BegInd_Photons, EndInd_Photons)
        #First PDC filtration - energies and regions
        # for each frame Filter_Pairs_Array_Numba checks if there is at least one photon in the region of interest at the desired energy
        # Energy_and_Region_Filtration is a 3D array of photons that successfully passed the SPDC filtration
        Energy_and_Region_Filtration = HFF.Timing_Of_Function(HFF.Filter_Pairs_Array_Numba, 'energy and regions filtration', True, Data_in_3D_Array,EnergyMin1, EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2,YMin2, YMax2, False)[0]
        HSF.SaveDataToFile(Energy_and_Region_Filtration,{'Folder Name':FolderNameAnalysed, 'File Name':FileList[FileInd]+AnalysedFileSufix})
        print('Finished conversion to 3D and saving in {0:.2f} seconds'.format(time.perf_counter()-StartTime))

        # Each 10 seconds prints the number of files that were analyzed
        if LoopTimer + 10 < time.perf_counter():
            print('Finished loading {0:.2f}% of files.'.format(100 * (FileInd-StartFileIndex) / (EndFileIndex-StartFileIndex)))
            LoopTimer = time.perf_counter()

    print('Finished loading and analyzing {} files. It took {:0} seconds'.format(EndFileIndex-StartFileIndex, time.perf_counter() - TotalTimer))

def Load_and_Process_All_Raw_Data_with_Timings(FolderNameRawData, StartFileIndex, EndFileIndex,CalibFolder, TimeWindow, Max_Number_of_Photons, Min_Number_of_Photons, FolderNameAnalysed, AnalysedFileSufix,EnergyMin1, EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2,YMin2, YMax2, Pump_Energy, Tolerance):
    FileList = HFOF.ListFilesInFolder(FolderNameRawData)
    HFOF.Advacam_Prepare_Raw_ASCII_List(FileList)

    if (EndFileIndex < 0) | (EndFileIndex > len(FileList)):
        EndFileIndex = len(FileList)

    LoopTimer = time.perf_counter()
    TotalTimer = time.perf_counter()
    PDC_Events_TimeTagsALL = numpy.array([])
    for FileInd in range(StartFileIndex, EndFileIndex):
        print('\n')
        SortedRawData, PhotonCounter, BegInd, EndInd, Filtered_Photons_per_Frame, Filtered_Frames_Indexes, Filtered_RawData, PixelEnergy, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Combined_Photons, Photons_Coordinates,Photons_XY, BegInd_Photons, EndInd_Photons = EFF.Timing_Of_Function(Load_and_Process_Single_File,'single raw file total analysis',True,FolderNameRawData,FileList[FileInd],CalibFolder,TimeWindow,Max_Number_of_Photons,Min_Number_of_Photons)
        Photons_Time_Diff, Photons_Start_Time = Pixels_Time_Differences(Filtered_RawData, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter,Filtered_Photons_per_Frame)
        StartTime = time.perf_counter()
        Data_in_3D_Array = Convert_2D_to_3D_array(Combined_Photons, Photons_XY, BegInd_Photons, EndInd_Photons)
        #First PDC filtration - energies and regions
        Energy_and_Region_Filtration, Frame_Filter, Region1_Filter, Region2_Filter = HFF.Timing_Of_Function(EFF.Filter_Pairs_Array_Numba, 'energy and regions filtration', False, Data_in_3D_Array, EnergyMin1, EnergyMax1, XMin1, XMax1, YMin1, YMax1, EnergyMin2, EnergyMax2, XMin2, XMax2, YMin2, YMax2, False)
        Filtered_Photons_per_Frame, Filtered_BegInd_Photons, Filtered_EndInd_Photons, Filtered_Time_Difference, Filtered_Start_Time, Photon_Filter = Photon_Timing_Filtration(BegInd_Photons, EndInd_Photons, Photons_Time_Diff, Photons_Start_Time, Frame_Filter, Region1_Filter, Region2_Filter)
        Energy_Conserving_Frames, NumberOfPixels, FilterOfFramesWithPairs, EnergySum, EnergyConservingFilter = EFF.Generate_EnergyConserving_Array(Energy_and_Region_Filtration, Pump_Energy, Tolerance)
        # for each file in the folder we find the PDC events time tags. we have a vactor of the differenc between TOAs.
        PDC_Events_TimeTags = Prepare_PDC_Time_Histogram_loop(Combined_Photons, Photons_XY, Photon_Filter, Filtered_BegInd_Photons, Filtered_EndInd_Photons, Filtered_Time_Difference, Filtered_Start_Time, EnergyConservingFilter)
        # from all files we combine the TOA difference vectors into a one large vector
        PDC_Events_TimeTagsALL = numpy.append(PDC_Events_TimeTagsALL, PDC_Events_TimeTags)

        #HSF.SaveDataToFile(Energy_and_Region_Filtration,{'Folder Name':FolderNameAnalysed, 'File Name':FileList[FileInd]+AnalysedFileSufix})
        print('Finished conversion to 3D and saving in {0:.2f} seconds'.format(time.perf_counter()-StartTime))


        if LoopTimer + 10 < time.perf_counter():
            print('Finished loading {0:.2f}% of files.'.format(100 * (FileInd-StartFileIndex) / (EndFileIndex-StartFileIndex)))
            LoopTimer = time.perf_counter()

    print('Finished loading and analyzing {} files. It took {:0} seconds'.format(EndFileIndex-StartFileIndex, time.perf_counter() - TotalTimer))
    # Prep_For_Enrgy_Histogram creates the histogram of PDC_Events_TimeTagsALL
    Hist, Hist_Bins = HFF.Prep_For_Enrgy_Histogram(PDC_Events_TimeTagsALL, 1.5625, 0, 1000)
    # Custom_Plot_7 plots the histogram Hist
    SpectrumFig = HPF.Custom_Plot_7(Hist, Hist_Bins, XLabel='Time tag diffrence (ns)', YLabel='Number of pairs',Title='Time tag diffrences{}'.format(PDC_Events_TimeTagsALL.shape[0]))
    HSF.SaveDataToFile(PDC_Events_TimeTagsALL, {'Folder Name': FolderNameAnalysed, 'File Name': AnalysedFileSufix})
    return SpectrumFig, PDC_Events_TimeTagsALL

@jit(nopython=True, nogil=True, parallel=True)
def Photon_Timing_Filtration(BegInd_Photons, EndInd_Photons, Photons_Time_Diff, Photons_Start_Time, Frame_Filter, Region1_Filter, Region2_Filter):
    """
    A function that filters the timing data of photons that were selected by the Energy\Region filtration

    Args:
        BegInd_Photons: its size is the number of frames
        EndInd_Photons: its size is the number of frames
        Photons_Time_Diff: its size is the total number of photons
        Photons_Start_Time: its size is the total number of photons
        Frame_Filter: its size is the number of frames
        Region1_Filter: its size is the number of frames and the maximal number of photons in frame
        Region2_Filter: its size is the number of frames and the maximal number of photons in frame

    Returns:
        Filtered_Photons_per_Frame - Number of photons in each frame that passed the filtration. Its length is the number of frames that contain at least 2 photons in region1 and region2 within the correct energy range.
        Filtered_BegInd_Photons - Indexes of begging of frames in Filtered_Time_Difference and in Filtered_Start_Time. Its length is the number of frames after filtration (frames that contain at least 2 photons in region1 and region2 within the correct energy range).
        Filtered_EndInd_Photons - Indexes of end of frames in Filtered_Time_Difference and in Filtered_Start_Time. Its length is the number of frames after filtration (frames that contain at least 2 photons in region1 and region2 within the correct energy range).
        Filtered_Time_Difference - maximal time diffrences between pixels of each photon
        Filtered_Start_Time - time tag of first pixel of each photon

    """
    Filtered_Frame_Indexes_Array = numpy.nonzero(Frame_Filter)[0] # index from total number of frames
    Filtered_Photons_per_Frame = numpy.zeros(len(Filtered_Frame_Indexes_Array), dtype=numpy.uint8)
    Photon_Filter=numpy.full(Photons_Time_Diff.shape,False,dtype=numpy.bool_)

    for Filtered_Frame_For_Ind in prange(len(Filtered_Frame_Indexes_Array)):
        FrameInd = Filtered_Frame_Indexes_Array[Filtered_Frame_For_Ind] # index from total number of frames
        FrameLength = EndInd_Photons[FrameInd]-BegInd_Photons[FrameInd]
        Photon_Filter[BegInd_Photons[FrameInd]:EndInd_Photons[FrameInd]] = numpy.logical_or(Region1_Filter[FrameInd][:FrameLength],Region2_Filter[FrameInd][:FrameLength])
        # Filtered_Photons_per_Frame[Filtered_Frame_For_Ind] = Region1_Photons.shape[0]+Region2_Photons.shape[0]
        Filtered_Photons_per_Frame[Filtered_Frame_For_Ind] = numpy.count_nonzero(numpy.logical_or(Region1_Filter[FrameInd], Region2_Filter[FrameInd]))
        #for Photon_Ind in range(BegInd_Photons[FrameInd]:EndInd_Photons[FrameInd]): # loop over indexes from total photon length array
        # which photons to take from the frame
        # Photons_Start_Time[Photon_Ind]
        # Photons_Time_Diff[Photon_Ind]
    Filtered_Time_Difference = Photons_Time_Diff[Photon_Filter]
    Filtered_Start_Time = Photons_Start_Time[Photon_Filter]

    Filtered_BegInd_Photons = numpy.zeros(Filtered_Frame_Indexes_Array.shape[0],dtype=numpy.uint64)
    Filtered_BegInd_Photons[1:] = numpy.cumsum(Filtered_Photons_per_Frame[:-1].astype(numpy.uint64))
    Filtered_EndInd_Photons = numpy.cumsum(Filtered_Photons_per_Frame.astype(numpy.uint64))
    return Filtered_Photons_per_Frame, Filtered_BegInd_Photons, Filtered_EndInd_Photons, Filtered_Time_Difference, Filtered_Start_Time, Photon_Filter

@jit(nopython=True, nogil=True, parallel=True)
def Prepare_PDC_Time_Histogram_loop(Combined_Photons, Photons_XY, Photon_Filter, Filtered_BegInd_Photons, Filtered_EndInd_Photons, Filtered_Time_Difference, Filtered_Start_Time, EnergyConservingFilter):
    Frame_with_PDC = numpy.nonzero(EnergyConservingFilter)[0]
    PDC_Events_TimeTags = numpy.zeros(Frame_with_PDC.shape[0],dtype=numpy.float32)
    for PDC_Frame in prange(Frame_with_PDC.shape[0]):
        PDC_Photons_Start_Time = Prepare_PDC_Time_Histogram(Combined_Photons, Photons_XY, Photon_Filter, Filtered_BegInd_Photons, Filtered_EndInd_Photons, Filtered_Time_Difference, Filtered_Start_Time, Frame_with_PDC[PDC_Frame])[3]
        PDC_Events_TimeTags[PDC_Frame] = PDC_Photons_Start_Time
    return PDC_Events_TimeTags

@jit(nopython=True, nogil=True, parallel=False)
def Prepare_PDC_Time_Histogram(Combined_Photons, Photons_XY, Photon_Filter, Filtered_BegInd_Photons, Filtered_EndInd_Photons, Filtered_Time_Difference, Filtered_Start_Time, Frame_Ind):
    '''
    Function that returns photon data (energy, xy location, first pixel time tag, pixel time difference) for a required frame by the parameter Frame_Ind.
    Args:
        Combined_Photons: all photon energies before filtration
        Photons_XY: 2D photon location in X Y before filtration (related to Combined_Photons)
        Photon_Filter: boolian filter of photons from Combined_Photons that satisfy the filtration of regions and energies (not PDC).
        Filtered_BegInd_Photons: pointer to beginning of frames after applying Photon_Filter
        Filtered_EndInd_Photons: pointer to ending of frames after applying Photon_Filter
        Filtered_Time_Difference: Difference in time tag of pixels of each photon after applying Photon_Filter (charge sharing timer).
        Filtered_Start_Time: Time tag of first pixels of each photon after applying Photon_Filter
        Frame_Ind: Index of frame

    Returns:

    '''
    Filtered_Combined_Photons = Combined_Photons[Photon_Filter]
    Filtered_Photons_XY = Photons_XY[Photon_Filter]

    Frame_Photon_Energies = Filtered_Combined_Photons[Filtered_BegInd_Photons[Frame_Ind]:Filtered_EndInd_Photons[Frame_Ind]]
    Frame_Photon_XY = Filtered_Photons_XY[Filtered_BegInd_Photons[Frame_Ind]:Filtered_EndInd_Photons[Frame_Ind]]
    Frame_Time_Dif = Filtered_Time_Difference[Filtered_BegInd_Photons[Frame_Ind]:Filtered_EndInd_Photons[Frame_Ind]]
    Frame_Start_Time = Filtered_Start_Time[Filtered_BegInd_Photons[Frame_Ind]:Filtered_EndInd_Photons[Frame_Ind]] # start time tag of PDC photons
    return Frame_Photon_Energies, Frame_Photon_XY, Frame_Time_Dif, Frame_Start_Time[1]-Frame_Start_Time[0], Frame_Start_Time[1]-Frame_Start_Time[0]+Frame_Time_Dif[1]

def PDC_Ploting(Energy_Conservation,Binning,ROIX,ROIY):
    All_Pairs = numpy.vstack(Energy_Conservation)
    Image_Of_Pairs = HFF.Prepare_For_Plotting_Array_V2(All_Pairs, ImageShape={'Vertical Pixels':256,'Horizontal Pixels':256*2})
    BinnedImage = HFF.BinFrame(Image_Of_Pairs,{'vertical binning':Binning,'horizontal binning':Binning})
    Fig1 = HPF.Custom_Plot_8(Image_Of_Pairs,CLimMax=5)
    Fig2 = HPF.Custom_Plot_8(BinnedImage,CLimMax=20)

    return Fig1, Fig2

@jit(nopython=True, parallel=False)
def FrameDivLoop_Numba(BegInd,EndInd,RawData):
    FrameList = list()
    FrameLength = 0
    # for Ind,OneFrame in enumerate(zip(BegInd,EndInd)):
    # for Ind in prange(EndInd.shape[0]):
    # It is important to take EndInd because it could be shorter by one line if a time window ends at the end of the file
    for Ind in range(EndInd.shape[0]):
        OneFrame = [BegInd[Ind],EndInd[Ind]]
        FrameLength = OneFrame[1] - OneFrame[0]
        FrameTemp = numpy.zeros((FrameLength,3),dtype=numpy.int32)
        # Save energy (ToT) and time (ToA)
        FrameTemp[:, 1] = RawData[OneFrame[0]:OneFrame[1], 2]
        FrameTemp[:, 2] = RawData[OneFrame[0]:OneFrame[1], 1]
        FrameTemp[:, 0] = RawData[OneFrame[0]:OneFrame[1], 0]
        FrameList.append(FrameTemp)
    return FrameList

def Frame_Div_Loop_Array(BegInd,EndInd,LargestFrame,RawData):
    FrameList = numpy.zeros((EndInd.shape[0],LargestFrame,3),dtype=numpy.int32)
    
    for Ind in range(EndInd.shape[0]):
        FrameSize = EndInd[Ind]-BegInd[Ind]        
        FrameList[Ind,:FrameSize,0] = RawData[BegInd[Ind]:EndInd[Ind],0]
        FrameList[Ind,:FrameSize,1] = RawData[BegInd[Ind]:EndInd[Ind],2]
        FrameList[Ind,:FrameSize,2] = RawData[BegInd[Ind]:EndInd[Ind],1]       

    return FrameList

@jit(nopython=True, parallel=True)
def Frame_Div_Loop_Array_Numba(BegInd,EndInd,LargestFrame,RawData):
    
    FrameList = numpy.zeros((EndInd.shape[0],LargestFrame,3),dtype=numpy.int32)
    
    #for Ind in range(EndInd.shape[0]):
    for Ind in prange(EndInd.shape[0]):
        FrameSize = EndInd[Ind]-BegInd[Ind]        
        FrameList[Ind,:FrameSize,0] = RawData[BegInd[Ind]:EndInd[Ind],0]
        FrameList[Ind,:FrameSize,1] = RawData[BegInd[Ind]:EndInd[Ind],2]
        FrameList[Ind,:FrameSize,2] = RawData[BegInd[Ind]:EndInd[Ind],1]       

    return FrameList

def Index_To_Pixels(MatIndex,Pixels_Per_Chip_V = 256, Pixels_Per_Chip_H = 256, NumberOfChips = 2):
    PixelArray = numpy.zeros((len(MatIndex),2))
    for ChipNum in range(NumberOfChips):
        #Find all the pixels in one of the chips
        tempInd = (MatIndex < (Pixels_Per_Chip_V * Pixels_Per_Chip_H)*(ChipNum+1)) & (MatIndex >= (Pixels_Per_Chip_V * Pixels_Per_Chip_H)*ChipNum)
        #Shift the indexes of chip number ChipNum to values of single chip
        NewMatInd = (MatIndex[tempInd] - Pixels_Per_Chip_H*Pixels_Per_Chip_V*ChipNum)
        #Convert Matrix Index to (x,y) by using numpy.unravel_index
        (PixelArray[tempInd,0],PixelArray[tempInd,1]) = numpy.unravel_index(NewMatInd,(Pixels_Per_Chip_V,Pixels_Per_Chip_H))
        #Move the pixels horizontaly to the correct chip
        PixelArray[tempInd,1] += Pixels_Per_Chip_H*ChipNum
    return PixelArray.astype(numpy.uint16)

def Find_Frame_Indexes(RawData, TimeWindow):
    """
    A function that finds indexes of beginning and end of frames with length TimeWindow in RawData.
    TimeWindow is in nano-seconds
    """
    # ********** Important ****************
    # This function drops last frame if it ends at the end of the raw data array (its index is the last element)
    # ********** Important ****************
    # The value in nano seconds of time stamps
    One_Clock_Pulse = 25
    # Calculate subtraction between adjacent rows in RawData. Could be negative values therefore int and not uint
    PixTimeDiffrences = numpy.diff(RawData[:, 1].astype(numpy.int32))
    # ****** Should understand why negative!
    # Take absolute value
    PixTimeDiffrences=numpy.fabs(PixTimeDiffrences).astype(numpy.int32)
    # Make mask of raws where difference is less than the time window
    PulsesTimeWindow = int(TimeWindow/One_Clock_Pulse)
    FrameFilter = (PixTimeDiffrences < PulsesTimeWindow)
    # This mask doesn't show windows with single pixel. Separate_Single_Pixels function finds them.
    # Insert one False value at the beggining of FrameFilter array.
    # I need this for correct diff value for the first pair in FrameFilter
    FrameFilter = numpy.insert(FrameFilter,0,False)
    # Calculate diff of the mask array to find beginig and end of a frame
    FrameBeginEnd = numpy.diff(FrameFilter.astype(numpy.int16))
    # Beggining of frame is when FrameFilter changes from False to True
    BegInd = numpy.nonzero(FrameBeginEnd > 0)
    BegInd = BegInd[0]
    # End of frame is when FrameFilter changes from True to False
    EndInd = numpy.nonzero(FrameBeginEnd < 0)
    # Need to add 1 to the index
    EndInd = EndInd[0] + 1

    return BegInd, EndInd

def Devide_To_Frames_Array(RawData, TimeWindow):
    """
    Divide numpy array of pixels (raw data) to 3D array of frames of time window TimeWindow.
    """
    
    StartTime = time.time()
    BegInd,EndInd = Find_Frame_Indexes(RawData, TimeWindow)
    print('Found indexes after {0:.2f} seconds'.format(time.time()-StartTime))
    
    StartTime = time.time()
    print('Start dividing to 3D numpy array of frames with time window {0:} ns...'.format(TimeWindow))
    # Find size of largest frame.
    # BegInd could be larger than EndInd therefore I take the kength of EndInd
    # BegInd could be larger because a frame could be at the end of RawData
    LargestFrame = (EndInd-BegInd[:EndInd.shape[0]]).max()
    FrameArray = Frame_Div_Loop_Array(BegInd, EndInd, LargestFrame, RawData)
    print('Finished dividing to frames after {0:.2f} seconds'.format(time.time()-StartTime))
    return FrameArray
def Devide_To_Frames_Array_Numba(RawData, TimeWindow):
    """
    Divide numpy array of pixels (raw data) to 3D array of frames of time window TimeWindow.
    """
    
    StartTime = time.time()
    BegInd,EndInd = Find_Frame_Indexes(RawData, TimeWindow)
    print('Found indexes after {0:.2f} seconds'.format(time.time()-StartTime))
    
    StartTime = time.time()
    print('Start dividing to 3D numpy array of frames with time window {0:} ns...'.format(TimeWindow))
    # Find size of largest frame.
    # BegInd could be larger than EndInd therefore I take the kength of EndInd
    # BegInd could be larger because a frame could be at the end of RawData
    LargestFrame = (EndInd-BegInd[:EndInd.shape[0]]).max()
    numba.set_num_threads(30)
    FrameArray = Frame_Div_Loop_Array_Numba(BegInd, EndInd, LargestFrame, RawData)
    print('Finished dividing to frames after {0:.2f} seconds'.format(time.time()-StartTime))
    return FrameArray

def Devide_To_Frames_List(RawData, TimeWindow):
    StartTime = time.time()
    """
    A function that divides pixels in RawData to frames of time length TimeWindow.
    TimeWindow is in nano-seconds
    """
    # The value in seconds of time stamps
    One_Clock_Pulse = 25
    # Calculate substruction between adjacent raws in RawData. Could be negative values therefore int and not uint
    PixTimeDiffrences = numpy.diff(RawData[:, 1].astype(numpy.int32))
    # ****** Should understand why negative!
    # Take absolute value
    PixTimeDiffrences=numpy.fabs(PixTimeDiffrences).astype(numpy.int32)
    # Make mask of raws where difference is less than the time window
    PulsesTimeWindow = int(TimeWindow/One_Clock_Pulse)
    FrameFilter = (PixTimeDiffrences < PulsesTimeWindow)
    # This mask doesn't show windows with single pixel. Separate_Single_Pixels function finds them.
    # Insert one False value at the beggining of FrameFilter array.
    # I need this for correct diff value for the first pair in FrameFilter
    FrameFilter = numpy.insert(FrameFilter,0,False)
    # Calculate diff of the mask array to find beginig and end of a frame
    FrameBeginEnd = numpy.diff(FrameFilter.astype(numpy.int16))
    # Beggining of frame is when FrameFilter changes from False to True
    BegInd = numpy.nonzero(FrameBeginEnd > 0)
    BegInd = BegInd[0]
    # End of frame is when FrameFilter changes from True to False
    EndInd = numpy.nonzero(FrameBeginEnd < 0)
    # Need to add 1 to the index
    EndInd = EndInd[0] + 1

    FrameList = list()
    FrameLength = int()
    for Ind,OneFrame in enumerate(zip(BegInd,EndInd)):
        FrameLength = OneFrame[1] - OneFrame[0]
        FrameTemp = numpy.zeros((FrameLength,4),dtype=numpy.int32)
        # Save energy (ToT) and time (ToA)
        FrameTemp[:,(2,3)] = RawData[OneFrame[0]:OneFrame[1], (2, 1)]
        FrameTemp[:, :2] = Index_To_Pixels(RawData[OneFrame[0]:OneFrame[1],0])
        FrameList.append(FrameTemp)
    print('Finished time division after {} seconds'.format(time.time() - StartTime))
    return FrameList

def Devide_To_Frames_List_Faster(RawData, TimeWindow):    
    """
    A function that divides pixels in RawData to frames of time length TimeWindow.
    TimeWindow is in nano-seconds
    """
    StartTime = time.time()
    # The value in seconds of time stamps
    One_Clock_Pulse = 25
    # Calculate substruction between adjacent raws in RawData. Could be negative values therefore int and not uint
    PixTimeDiffrences = numpy.diff(RawData[:, 1].astype(numpy.int32))
    # ****** Should understand why negative!
    # Take absolute value
    PixTimeDiffrences=numpy.fabs(PixTimeDiffrences).astype(numpy.int32)
    # Make mask of raws where difference is less than the time window
    PulsesTimeWindow = int(TimeWindow/One_Clock_Pulse)
    FrameFilter = (PixTimeDiffrences < PulsesTimeWindow)
    # This mask doesn't show windows with single pixel. Separate_Single_Pixels function finds them.
    # Insert one False value at the beggining of FrameFilter array.
    # I need this for correct diff value for the first pair in FrameFilter
    FrameFilter = numpy.insert(FrameFilter,0,False)
    # Calculate diff of the mask array to find beginig and end of a frame
    FrameBeginEnd = numpy.diff(FrameFilter.astype(numpy.int16))
    # Beggining of frame is when FrameFilter changes from False to True
    BegInd = numpy.nonzero(FrameBeginEnd > 0)
    BegInd = BegInd[0]
    # End of frame is when FrameFilter changes from True to False
    EndInd = numpy.nonzero(FrameBeginEnd < 0)
    # Need to add 1 to the index
    EndInd = EndInd[0] + 1
    print('Found indexes after {0:.2f} seconds'.format(time.time()-StartTime))
    
    print('Start dividing to list of numpy frames with time window {0:} ns...'.format(TimeWindow))
    StartTime = time.time()
    FrameList = list()
    FrameLength = int()
    for Ind,OneFrame in enumerate(zip(BegInd,EndInd)):
        FrameLength = OneFrame[1] - OneFrame[0]
        FrameTemp = numpy.zeros((FrameLength,3),dtype=numpy.int32)
        # Save energy (ToT) and time (ToA)
        FrameTemp[:,(1,2)] = RawData[OneFrame[0]:OneFrame[1], (2, 1)]
        FrameTemp[:, 0] = RawData[OneFrame[0]:OneFrame[1],0]
        FrameList.append(FrameTemp)
    print('Finished frame division after {} seconds'.format(time.time()-StartTime))
    return FrameList

def Devide_To_Frames_List_Numba(RawData, TimeWindow):    
    """
    A function that divides pixels in RawData to frames of time length TimeWindow.
    TimeWindow is in nano-seconds
    """
    StartTime = time.time()
    # The value in seconds of time stamps
    One_Clock_Pulse = 25
    # Calculate substruction between adjacent raws in RawData.
    # In older version used int and not uint because could be negative values. Now data is sorted and should not be negative
    PixTimeDiffrences = numpy.diff(RawData[:, 1].astype(numpy.uint32))

    # Take absolute value. This is old part when data was not sorted. Now I sort it and no need in abs
    # PixTimeDiffrences=numpy.fabs(PixTimeDiffrences).astype(numpy.int32)

    # Make mask of raws where difference is less than the time window
    PulsesTimeWindow = int(TimeWindow/One_Clock_Pulse)
    FrameFilter = (PixTimeDiffrences < PulsesTimeWindow)
    # This mask doesn't show windows with single pixel. Separate_Single_Pixels function finds them.
    # Insert one False value at the beggining of FrameFilter array.
    # I need this for correct diff value for the first pair in FrameFilter
    FrameFilter = numpy.insert(FrameFilter,0,False)
    # Calculate diff of the mask array to find beginig and end of a frame
    FrameBeginEnd = numpy.diff(FrameFilter.astype(numpy.int16))
    # Beggining of frame is when FrameFilter changes from False to True
    BegInd = numpy.nonzero(FrameBeginEnd > 0)
    # numpy.nonzero() returns a tuple, therefore taking its firs [0] part
    BegInd = BegInd[0]
    # End of frame is when FrameFilter changes from True to False
    EndInd = numpy.nonzero(FrameBeginEnd < 0)
    # Need to add 1 to the index
    EndInd = EndInd[0] + 1
    print('Found indexes after {0:.2f} seconds'.format(time.time()-StartTime))
    print('Start dividing to list of numpy frames with NUMBA(!!!) with time window {0:} ns...'.format(TimeWindow))
    StartTime = time.time()
    FrameList = FrameDivLoop_Numba(BegInd,EndInd,RawData)
    print('Finished time division after {} seconds'.format(time.time()-StartTime))
    return FrameList


def Analyze_Pixels_and_Frames_Numba_New(RawData, TimeWindow, Min_Number_of_Photons, Max_Number_of_Photons):
    """
    A function that finds indexes of pixels in RawData that arrived within time length TimeWindow.
    It is improved function to Devide_To_Frames_List_Numba.
    Instead of list or 3D array, use 2D arrays with index vectors. Use more parallelizem with Numba and dived procedure to more functions
    TimeWindow is in nano-seconds
    """

    BegInd, EndInd = HFF.Timing_Of_Function(Find_BegInd_EndInd,'finding begining and end indexes of frames\events',True,RawData, TimeWindow)
    PhotonCounter = HFF.Timing_Of_Function(Count_Photons_In_Frame, 'counting photons in frame\event', True,RawData[:,0].astype(numpy.int32), BegInd,EndInd)
    # Count photon per each frame
    Photons_per_Frame = numpy.zeros(EndInd.shape[0],dtype=numpy.uint8)
    if BegInd.shape[0] > EndInd.shape[0]:
        HFF.Timing_Of_Function(Number_of_Photons_per_Frame_GPU,'counting number of photons per frame',True,PhotonCounter, BegInd[:-1], EndInd, Photons_per_Frame)
        Filtered_Pixel_Number, Filtered_Pixel_Number_per_Frame, Filtered_Frames_Indexes = HFF.Timing_Of_Function(Prepare_For_RawData_Filtering,'preparing for raw data filtration',True,Photons_per_Frame, BegInd[:-1], EndInd, Min_Number_of_Photons, Max_Number_of_Photons)
        Filtered_RawData, Filtered_BegInd, Filtered_EndInd = HFF.Timing_Of_Function(FilterRawData_Numba,'filtering raw data',True,RawData, BegInd[:-1], EndInd, Filtered_Frames_Indexes, Filtered_Pixel_Number, Filtered_Pixel_Number_per_Frame)
    else:
        HFF.Timing_Of_Function(Number_of_Photons_per_Frame_GPU,'counting number of photons per frame',True, PhotonCounter,BegInd,EndInd,Photons_per_Frame)
        Filtered_Pixel_Number, Filtered_Pixel_Number_per_Frame, Filtered_Frames_Indexes = HFF.Timing_Of_Function(Prepare_For_RawData_Filtering,'preparing for raw data filtration',True,Photons_per_Frame, BegInd, EndInd, Min_Number_of_Photons, Max_Number_of_Photons)
        Filtered_RawData, Filtered_BegInd, Filtered_EndInd = HFF.Timing_Of_Function(FilterRawData_Numba,'filtering raw data',True,RawData, BegInd, EndInd, Filtered_Frames_Indexes, Filtered_Pixel_Number, Filtered_Pixel_Number_per_Frame)

    Filtered_PhotonCounter = HFF.Timing_Of_Function(Count_Photons_In_Frame, 'counting photons in frame\event', True,Filtered_RawData[:, 0].astype(numpy.int32), Filtered_BegInd, Filtered_EndInd)
    Filtered_Photons_per_Frame = numpy.zeros(Filtered_EndInd.shape[0], dtype=numpy.uint8)
    HFF.Timing_Of_Function(Number_of_Photons_per_Frame_GPU, 'counting number of photons per frame', True, Filtered_PhotonCounter,Filtered_BegInd, Filtered_EndInd, Filtered_Photons_per_Frame)
    return BegInd,EndInd,PhotonCounter, Photons_per_Frame, Filtered_Frames_Indexes, Filtered_RawData, Filtered_BegInd, Filtered_EndInd, Filtered_PhotonCounter, Filtered_Photons_per_Frame

def Prepare_For_RawData_Filtering(Photons_per_Frame, BegInd, EndInd, Min_Number_of_Photons, Max_Number_of_Photons):
    Filtered_Frames_Indexes = numpy.nonzero((Photons_per_Frame>=Min_Number_of_Photons) & (Photons_per_Frame<=Max_Number_of_Photons))[0]
    Filtered_Pixel_Number_per_Frame = EndInd[Filtered_Frames_Indexes] - BegInd[Filtered_Frames_Indexes]
    Filtered_Pixel_Number = Filtered_Pixel_Number_per_Frame.sum()
    return Filtered_Pixel_Number, Filtered_Pixel_Number_per_Frame, Filtered_Frames_Indexes

@jit(nopython=True, nogil=True, parallel=True)
def FilterRawData_Numba(RawData, BegInd, EndInd, Filtered_Frames_Indexes, Filtered_Pixel_Number, Filtered_Pixel_Number_per_Frame):
    """
    Parameters:
        RawData - 5 columns, Matrix Index, ToA, ToT etc.
        BegInd, EndInd - begining and ending indexes of frames in raw data.
        Filtered_Frames_Indexes - indexes of frames that should keep. Indexes are from BegInd\EndInd
        Filtered_Pixel_Number - total number of rows in new raw data

    """
    Filtered_RawData = numpy.zeros((Filtered_Pixel_Number,5),dtype=numpy.uint32)
    Filtered_BegInd = numpy.zeros(Filtered_Frames_Indexes.shape[0],dtype=numpy.int64)
    Filtered_EndInd = numpy.zeros(Filtered_Frames_Indexes.shape[0],dtype=numpy.int64)

    Filtered_BegInd[1:] = numpy.cumsum(Filtered_Pixel_Number_per_Frame[:-1])
    Filtered_EndInd = numpy.cumsum(Filtered_Pixel_Number_per_Frame)
    for FrameInd in prange(Filtered_Frames_Indexes.shape[0]):
        #ShiftedIndex = FrameInd + 1
        #Filtered_BegInd[ShiftedIndex] = Filtered_EndInd[FrameInd]
        #Filtered_EndInd[ShiftedIndex] = Filtered_BegInd[ShiftedIndex] + Filtered_Pixel_Number_per_Frame[FrameInd]
        #print(str(Filtered_BegInd[ShiftedIndex])+','+str(Filtered_EndInd[ShiftedIndex]))
        Filtered_RawData[Filtered_BegInd[FrameInd]:Filtered_EndInd[FrameInd],:] = RawData[BegInd[Filtered_Frames_Indexes[FrameInd]]:EndInd[Filtered_Frames_Indexes[FrameInd]],:]
    return Filtered_RawData, Filtered_BegInd, Filtered_EndInd

def Find_BegInd_EndInd(RawData, TimeWindow):
    """
    A function that find begging and end indexes of pixels within a time window.
    If frame ends at the end of raw data and a new one does not appear, both returned vectors will have same size. If frame begins and end was not found, beginning indexes vector will be longer by 1
    Parameters:
        RawData - the raw data - 2D numpy array
        TimeWindow - time window in nano-seconds
    Returns:
        Two numpy 1D arrays of beginning and end indexes
    """
    # The value in nano-seconds of time stamps
    One_Clock_Pulse = 25
    # Calculate substruction between adjacent raws in RawData.
    # In older version used int and not uint because could be negative values. Now data is sorted and should not be negative
    PixTimeDiffrences = numpy.diff(RawData[:, 1].astype(numpy.uint32))

    # Make mask of raws where difference is less than the time window
    PulsesTimeWindow = int(TimeWindow / One_Clock_Pulse)
    FrameFilter = (PixTimeDiffrences < PulsesTimeWindow)
    # This mask doesn't show windows with single pixel. Separate_Single_Pixels function finds them.
    # Insert one False value at the beggining of FrameFilter array.
    # I need this for correct diff value for the first pair in FrameFilter
    FrameFilter = numpy.insert(FrameFilter, 0, False)
    # Calculate diff of the mask array to find beginig and end of a frame
    FrameBeginEnd = numpy.diff(FrameFilter.astype(numpy.int16))
    # Beggining of frame is when FrameFilter changes from False to True
    BegInd = numpy.nonzero(FrameBeginEnd > 0)
    # numpy.nonzero() returns a tuple, therefore taking its firs [0] part
    BegInd = BegInd[0]
    # End of frame is when FrameFilter changes from True to False
    EndInd = numpy.nonzero(FrameBeginEnd < 0)
    # Need to add 1 to the index
    EndInd = EndInd[0] + 1
    return BegInd,EndInd

@jit(nopython=True, nogil=True, parallel=True)
def Count_Photons_In_Frame(RawData,BegInd,EndInd):
    """
    A function that gives the same number to all the pixels in a frame that belong to the same photon.
    Determins pixels from the same photon by checking the Matrix Index differences for pixels in the same frame
    Parameters:
        RawData - a 1D numpy array with MatrixIndex from the Advapix detector
        BegInd - a 1D numpy array with indexes of start of frames\events in raw data
        EndInd - a 1D numpy array with indexes of end of frames\events in raw data

        The EndInd could be shorter than BegInd, if the frame\event did not end within raw data
    Returns:
        PhotonCount - 1D array with length of RawData. Each lines tells to which photon this pixel belongs
    """
    NumberOfFrames = EndInd.shape[0]
    PhotonCount = numpy.zeros(RawData.shape[0],dtype=numpy.uint8)
    for Ind in prange(NumberOfFrames):
        Photon_Counter(RawData,PhotonCount,BegInd,EndInd,Ind)
    return PhotonCount

@jit(nopython=True, nogil=True, parallel=False)
def Photon_Counter(RawData,PhotonCount,BegInd,EndInd,Ind):
    """
    The function counts how many photon are in a frame. It gives to each pixels the number of photon to which it belongs
    Parameters:
        RawData - a 1D numpy array with MatrixIndex from the Advapix detector
        PhotonCount - an array that saves to which photon each pixel belongs.
        Ind - index of frame in RawData.
        The frame in RawData starts at BegInd[Ind] and ends at EndInd[Ind]-1
    """
    Frame_Length = EndInd[Ind] - BegInd[Ind]
    Frame_Photon_Count = 0
    # Loop over all the pixels in a frame (time window)
    for PixInd in range(Frame_Length):
        RawData_Index = BegInd[Ind] + PixInd
        # Check if this pixel was already counted
        if PhotonCount[RawData_Index] == 0:
            Frame_Photon_Count += 1
            # print('Frame Ind is:' + str(Ind)+ ', Pix_ind is:' + str(PixInd)+', Raw data index:' + str(RawData_Index)+', Photon count is: ' + str(Frame_Photon_Count))
            PhotonCount[RawData_Index] = Frame_Photon_Count

            # Check neighbour pixels:
            # Loop over all the pixels that come ofter pixel PixInd
            for SecondaryInd in range(Frame_Length-PixInd - 1):
                Secondary_RawData_Ind = RawData_Index + SecondaryInd + 1

                Diff_Check = numpy.abs(RawData[RawData_Index] - RawData[Secondary_RawData_Ind])
                if (Diff_Check == 1) | (Diff_Check == 256) | (Diff_Check == 255) | (Diff_Check == 257):
                    PhotonCount[Secondary_RawData_Ind] = PhotonCount[RawData_Index]
                #if Ind==1:
                    #print('Diff_Check: '+str(Diff_Check)+', Secondary Ind: '+str(SecondaryInd)+ ', RawData_Index:'+str(RawData_Index) + ', Secondary_RawData_Ind: ' + str(Secondary_RawData_Ind) + ', Photon count: '+str(PhotonCount[Secondary_RawData_Ind]))

@jit(nopython=True, nogil=True, parallel=False)
def Pixels_Time_Differences(RawData,BegInd,EndInd,PhotonCounter,Photons_per_Frame):
    Pixel_Time_Diffs = numpy.zeros(numpy.sum(Photons_per_Frame.astype(numpy.uint32)),dtype=numpy.float32)
    PhotonTimeDiffIndexes = numpy.zeros(Photons_per_Frame.shape[0],dtype=numpy.uint32)
    PhotonTimeDiffIndexes[1:] = numpy.cumsum(Photons_per_Frame[:-1].astype(numpy.uint32)) # Frame beginning index
    Photon_First_Time_Tag = numpy.zeros(Photons_per_Frame.sum(),dtype=numpy.float64)
    for FrameInd in prange(EndInd.shape[0]):
        for PhotonInd in range(1,Photons_per_Frame[FrameInd]+1):
            MinTimeTag = 0
            MaxTimeTag = 0
            for RawInd in range(BegInd[FrameInd],EndInd[FrameInd]):
                if PhotonCounter[RawInd] == PhotonInd:
                    TimeTag = RawData[RawInd,1]*25 - RawData[RawInd,3]*(25/16)
                    if MinTimeTag == 0:
                        MinTimeTag = TimeTag
                    if TimeTag > MaxTimeTag:
                        MaxTimeTag = TimeTag
                    if TimeTag < MinTimeTag:
                        MinTimeTag = TimeTag

            MainPhotonIndex = PhotonTimeDiffIndexes[FrameInd]+PhotonInd-1
            Pixel_Time_Diffs[MainPhotonIndex] = numpy.float32(MaxTimeTag-MinTimeTag)
            Photon_First_Time_Tag[MainPhotonIndex] = MinTimeTag
    return Pixel_Time_Diffs, Photon_First_Time_Tag

@jit(nopython=True, nogil=True, parallel=True)
def Pixels_Time_Differences_Haims(RawData,BegInd,EndInd,PhotonCounter,Photons_per_Frame):
    """
    A function the calculates time differences between pixels of each photons and between first and last photon arrival in a frame
    Parameters:
        RawData - numpy 2D array with 5 columns. Length as number of pixels. Can't be sliced because of the indexes inside BegInd,EndInd
        BegInd, EndInd - indexes inside RawData. Length as number of frames.
        PhotonCounter - counts to which photon each pixel belongs. Its length is as the length of RawData and couldn't be sliced because of the BegInd,EndInd indexes.
        Photons_per_Frame - number of photons in each frame. Length as number of frames.

    """
    Pixel_Time_Diffs = numpy.zeros(numpy.sum(Photons_per_Frame.astype(numpy.uint32)),dtype=numpy.float32)
    Photons_Arrival_Time_Diff = numpy.zeros(EndInd.shape[0],dtype=numpy.float32)
    PhotonTimeDiffIndexes = numpy.zeros(Photons_per_Frame.shape[0],dtype=numpy.uint32)
    PhotonTimeDiffIndexes[1:] = numpy.cumsum(Photons_per_Frame[:-1].astype(numpy.uint32))

    for FrameInd in prange(EndInd.shape[0]):
        First_Photon_Time_of_Arrival = 0
        Last_Photon_Time_of_Arrival = 0

        for PhotonInd in range(1,Photons_per_Frame[FrameInd]+1):
            MinTimeTag = 0
            MaxTimeTag = 0
            # Find first and last pixels times for each photon
            for RawInd in range(BegInd[FrameInd],EndInd[FrameInd]):
                if PhotonCounter[RawInd] == PhotonInd:
                    TimeTag = RawData[RawInd,1]*25 - RawData[RawInd,3]*(25/16)
                    if MinTimeTag == 0:
                        MinTimeTag = TimeTag
                    if TimeTag > MaxTimeTag:
                        MaxTimeTag = TimeTag
                    if TimeTag < MinTimeTag:
                        MinTimeTag = TimeTag
            Pixel_Time_Diffs[PhotonTimeDiffIndexes[FrameInd]+PhotonInd-1] = numpy.float32(MaxTimeTag-MinTimeTag)
            if PhotonInd == 1:
                First_Photon_Time_of_Arrival = MinTimeTag
                Last_Photon_Time_of_Arrival = First_Photon_Time_of_Arrival
            if PhotonInd>1:
                if Last_Photon_Time_of_Arrival < MinTimeTag:
                    Last_Photon_Time_of_Arrival = MinTimeTag
                if First_Photon_Time_of_Arrival > MinTimeTag:
                    First_Photon_Time_of_Arrival = MinTimeTag
        Photons_Arrival_Time_Diff[FrameInd] = Last_Photon_Time_of_Arrival - First_Photon_Time_of_Arrival
    return Pixel_Time_Diffs, Photons_Arrival_Time_Diff

# ***** NEW FUNCTION


# ***** NEW FUNCTION

@guvectorize([(uint8[:],int64[:],int64[:],uint8[:])],'(a),(),()->()',nopython=True, target='parallel')
def Number_of_Photons_per_Frame_Numba(PhotonCount,BegInd,EndInd,Photons_per_Frame):
    #Photons_per_Frame[0] = 1
    Photons_per_Frame[0] = PhotonCount[BegInd[0]:EndInd[0]].max()
    #for Ind in range(BegInd[0],EndInd[0]):
    #    if PhotonCount[Ind] > Photons_per_Frame[0]:
    #        Photons_per_Frame[0] = PhotonCount[Ind]

# Need to switch back to 'cuda'
@guvectorize([(uint8[:],int64[:],int64[:],uint8[:])],'(a),(),()->()',nopython=True, target='parallel')
def Number_of_Photons_per_Frame_GPU(PhotonCount,BegInd,EndInd,Photons_per_Frame):
    """
    A function that counts total number of photons in each frame.
    It finds the maximal value for PhotonCount for each frame.
    Alternative function to Number_of_Photons_per_Frame_Numba which uses numpy.max() to find total number of photons in each frame
    Parameters:
        PhotonCount - 1D array with length of raw data array. It marks for each pixel to which photon it belongs.
        BegInd, EndInd - 1D array with length of number of frames\events. Stors the Begining\End indexes of each frame\event
    """
    Photons_per_Frame[0] = 1
    #Photons_per_Frame[0] = PhotonCount[BegInd[0]:EndInd[0]].max()
    for Ind in range(BegInd[0],EndInd[0]):
        if PhotonCount[Ind] > Photons_per_Frame[0]:
            Photons_per_Frame[0] = PhotonCount[Ind]


'''
@guvectorize([(uint8[:],float32[:],int64[:],int64[:],uint8[:],float32[:],int64[:])],'(a),(a),(),(),(),(b)->()',nopython=True, target='cuda')
def Combine_SinglePhoton_InFrame_GPU(PhotonCount,PixelEnergy,BegInd,EndInd, Photons_per_Frame, PhotonsEnergies, PhotonsList,New_FrameInd):
    for Photon_Number in range(1,Photons_per_Frame[0]+1):
        Tot_Photon_Energy = 0
        Max_Pixel_Energy = 0

        for RawInd in (BegInd[0],EndInd[0]):

            if PhotonCount[RawInd] == Photon_Number:
                Tot_Photon_Energy += PixelEnergy[RawInd]
                if PixelEnergy[RawInd] > Max_Pixel_Energy:
                    Max_Pixel_Energy = PixelEnergy[RawInd]
                    Max_Photon_Index = RawInd
        PhotonsEnergies[Max_Photon_Index] = Tot_Photon_Energy
        
        # decide what to do with PhotonsEnergies
    New_FrameInd[0, 1] = New_FrameInd[0,0]+Photons_per_Frame[0]
'''

@jit(nopython=True, nogil=True, parallel=True)
def Combine_SinglePhoton_InFrame_Numba(RawData, PhotonCount, PixelEnergy, BegInd, EndInd, Photons_per_Frame):
    """
    A function that
    Parameters:
        RawData - 5 columns - Matrix Index, ToA, ToT etc.
        PhotonCount - to which photon each pixel belongs. The photons are numbered
        PixelEnergy - energy in keV of each pixel
        Photons_per_Frame - number of photons in each frame. Length is number of frames (EndInd.shape[0])
    """
    # Define new arrays
    Combined_Photons = numpy.zeros(Photons_per_Frame.sum(), dtype=numpy.float32)
    Photons_Coordinates = numpy.zeros(Combined_Photons.shape[0], dtype=numpy.uint32)
    # The indexes arrays are larger by 1 to compensate for first loop iteration
    BegInd_Photons = numpy.zeros(EndInd.shape[0],dtype=numpy.int64)
    EndInd_Photons = numpy.zeros(EndInd.shape[0],dtype=numpy.int64)

    BegInd_Photons[1:] = numpy.cumsum(Photons_per_Frame[:-1].astype(numpy.int64))
    EndInd_Photons = numpy.cumsum(Photons_per_Frame.astype(numpy.int64))
    # Loop over number of frames
    for FrameInd in prange(EndInd.shape[0]):
        # Fill arrays of indexes of frames in new photon array.
        # The first frame always starts with index zero. In order not to add if condition, took index array larger by one
        #New_Ind_Shift = FrameInd+1
        #BegInd_Photons[New_Ind_Shift] = EndInd_Photons[FrameInd]
        #EndInd_Photons[New_Ind_Shift] = BegInd_Photons[New_Ind_Shift] + Photons_per_Frame[FrameInd]
        # Loop over the number of photons that were found in each frame
        for Photon_Number in range(1, Photons_per_Frame[FrameInd] + 1):
            Tot_Photon_Energy = 0
            Max_Pixel_Energy = 0
            # Loop over all the pixels in the frame for single photon
            for RawInd in range(BegInd[FrameInd], EndInd[FrameInd]):
                # Collect all the pixels in the frame the belong to a single photon and sum their energies
                # Save the index of the pixel with the largest energy
                if PhotonCount[RawInd] == Photon_Number:
                    Tot_Photon_Energy += PixelEnergy[RawInd]
                    if PixelEnergy[RawInd] > Max_Pixel_Energy:
                        Max_Pixel_Energy = PixelEnergy[RawInd]
                        Max_Photon_Index = RawInd

            New_Photon_Ind = BegInd_Photons[FrameInd] + Photon_Number - 1
            Combined_Photons[New_Photon_Ind] = Tot_Photon_Energy
            Photons_Coordinates[New_Photon_Ind] = RawData[Max_Photon_Index,0]

    return Combined_Photons, Photons_Coordinates, BegInd_Photons, EndInd_Photons

@jit(nopython=True, nogil=True, parallel=True)
def Find_Frames_From_ToA(RawData, ToA, BegInd, EndInd,PhotonCounter,PhotonEnergy):
    #PhotonCounter = numpy.zeros(RawData.shape[0],dtype=numpy.uint8)
    #PhotonEnergies = numpy.zeros(RawData.shape[0],dtype=numpy.uint8)
    NumberOfFrames = EndInd.shape[0]
    Test = numpy.zeros(NumberOfFrames,dtype=numpy.uint8)
    #Test=0
    for FrameInd in prange(NumberOfFrames):
        Temp_Filter = RawData[BegInd[FrameInd]:EndInd[FrameInd],1] == ToA
        if Temp_Filter.any():
            Test[FrameInd]=1
            #Test=FrameInd
            #return RawData[BegInd[FrameInd]:EndInd[FrameInd],:], PhotonCounter[BegInd[FrameInd]:EndInd[FrameInd]],PhotonEnergy[BegInd[FrameInd]:EndInd[FrameInd]]

def Print_Photons_from_Frame_Ind(BegInd, EndInd, FrameInd, *DataArray):
    print('Index: {}'.format(FrameInd))
    for LineIndex in range(BegInd[FrameInd],EndInd[FrameInd]):
        for DataInd in range(len(DataArray)):
            print('{}'.format(DataArray[DataInd][LineIndex]),',\t',end='')
        print()
    #return numpy.nonzero(Test)[0]


def Separate_Single_Pixels(RawData, TimeWindow):
    """
    Function that gets numpy array of time stamps.
    RawData is an array with one column
    Returns indexes of single pixels
    """
    # The value in seconds of time stamps
    One_Clock_Pulse = 25
    PixTimeDiffrences = numpy.diff(RawData.astype(numpy.int32))
    # Take absolute value
    PixTimeDiffrences=numpy.fabs(PixTimeDiffrences).astype(numpy.int32)
    # Make mask of raws where difference is less than the time window
    PulsesTimeWindow = int(TimeWindow/One_Clock_Pulse)
    FrameFilter = (PixTimeDiffrences < PulsesTimeWindow)
    # Insert one False value at the beggining of FrameFilter array.
    # I need this for correct diff value for the first pair in FrameFilter
    FrameFilter = numpy.insert(FrameFilter,0,False)

    # Adding filter of adjacent values
    TempFilter = FrameFilter[1:]
    TempFilter = numpy.append(TempFilter,False)
    FrameFilter += TempFilter

    return numpy.nonzero(~FrameFilter)[0]

def Load_Calibration_Coef(FolderName):
    """
    Function that loads calibration matrixes to a dictionary CalibData and returns it
    """
    CalibData = dict()
    CalibData['L_a'] = numpy.loadtxt(FolderName + 'L09-W0096_a.txt')
    CalibData['L_b'] = numpy.loadtxt(FolderName + 'L09-W0096_b.txt')
    CalibData['L_c'] = numpy.loadtxt(FolderName + 'L09-W0096_c.txt')
    CalibData['L_t'] = numpy.loadtxt(FolderName + 'L09-W0096_t.txt')

    CalibData['K_a'] = numpy.loadtxt(FolderName + 'K09-W0096_a.txt')
    CalibData['K_b'] = numpy.loadtxt(FolderName + 'K09-W0096_b.txt')
    CalibData['K_c'] = numpy.loadtxt(FolderName + 'K09-W0096_c.txt')
    CalibData['K_t'] = numpy.loadtxt(FolderName + 'K09-W0096_t.txt')

    return CalibData

def Pixel_To_Energy(Pix_Ind, ToT, CalibData):
    """
    A function that converts for single pixel, all the measured values of ToT to energy.
    ToT (time over threshold) is given in the raw data.
    Pix_Ind is the index of pixel for wich the energy should be calculated.
    CalibData is a dict() with calibration values from Advacam
    """
    if Pix_Ind < (256*256):
        ShortAxis = int(Pix_Ind/256) # Row
        LongAxis= Pix_Ind % 256 # Column
        a = CalibData['L_a'][ShortAxis, LongAxis]
        b = CalibData['L_b'][ShortAxis, LongAxis]
        c = CalibData['L_c'][ShortAxis, LongAxis]
        t = CalibData['L_t'][ShortAxis, LongAxis]

        xind = LongAxis # until 511
        yind = ShortAxis # Until 255
    else:
        Second_Chip_Offset = 256*256
        Pix_Ind_Temp = Pix_Ind - Second_Chip_Offset
        ShortAxis = int(Pix_Ind_Temp/256) # Row
        LongAxis= (Pix_Ind_Temp % 256) # Column
        a = CalibData['K_a'][ShortAxis, LongAxis]
        b = CalibData['K_b'][ShortAxis, LongAxis]
        c = CalibData['K_c'][ShortAxis, LongAxis]
        t = CalibData['K_t'][ShortAxis, LongAxis]

        xind = LongAxis + 256 # until 511
        yind = ShortAxis # Until 255

    f = ToT
    PixEnrg = (-(b - f - a * t) + ((b - f - a * t) ** 2 - 4 * a * (f * t - b * t - c)) ** (0.5)) / (2 * a)

    return PixEnrg, xind, yind, (a,b,c,t)

def Convert_to_Energy(RawData, Indexes, Folder):
    """
    A function that converts the raw data ToT to energy in keV.
    It loads calibration files and sends them, an array with the ToT and an array with Matrix Index to a function that calculates energy.
    Here I use parallel threads calculation by Numba. I build the calculation function in a form of Numba guvectorize function.
    Parameters:
        RawData - a 2D array with 5 columns. First important columns are - Matrix Index, ToA, ToT
        Indexes - not in use
        Folder - folder of 8 calibration files. 4 for each chip in the detector
    Return:
        Energies_Vec - an array with energies in keV
    """
    CalibMatrix = Load_Calibration_Coef(Folder)
    Energies_Vec = numpy.zeros(RawData.shape[0],dtype=numpy.float32)
    #Pixel_To_Energy_Numba(RawData[Indexes,0],RawData[Indexes,2],CalibMatrix['L_a'],CalibMatrix['L_b'],CalibMatrix['L_c'],CalibMatrix['L_t'], CalibMatrix['K_a'],CalibMatrix['K_b'],CalibMatrix['K_c'],CalibMatrix['K_t'],Energies_Vec[Indexes])
    Pixel_To_Energy_Numba(RawData[:, 0], RawData[:, 2], CalibMatrix['L_a'], CalibMatrix['L_b'],
                          CalibMatrix['L_c'], CalibMatrix['L_t'], CalibMatrix['K_a'], CalibMatrix['K_b'],
                          CalibMatrix['K_c'], CalibMatrix['K_t'], Energies_Vec[:])
    return Energies_Vec

def Convert_to_Energy_GPU(RawData, Indexes, Folder):
    """
    A function that converts the raw data ToT to energy in keV.
    It loads calibration files and sends them, an array with the ToT and an array with Matrix Index to a function that calculates energy.
    Here I use GPU and the main difference is to convert the arrays that I send to it to Numpy contiguousarray.
    Parameters:
        RawData - a 2D array with 5 columns. First important columns are - Matrix Index, ToA, ToT
        Indexes - not in use
        Folder - folder of 8 calibration files. 4 for each chip in the detector
    Return:
        Energies_Vec - an array with energies in keV
    """
    CalibMatrix = Load_Calibration_Coef(Folder)
    Energies_Vec = numpy.zeros(RawData.shape[0],dtype=numpy.float32)
    #Pixel_To_Energy_Numba(RawData[Indexes,0],RawData[Indexes,2],CalibMatrix['L_a'],CalibMatrix['L_b'],CalibMatrix['L_c'],CalibMatrix['L_t'], CalibMatrix['K_a'],CalibMatrix['K_b'],CalibMatrix['K_c'],CalibMatrix['K_t'],Energies_Vec[Indexes])
    Pixel_To_Energy_GPU(numpy.ascontiguousarray(RawData[:, 0]), numpy.ascontiguousarray(RawData[:, 2]), CalibMatrix['L_a'], CalibMatrix['L_b'],
                          CalibMatrix['L_c'], CalibMatrix['L_t'], CalibMatrix['K_a'], CalibMatrix['K_b'],
                          CalibMatrix['K_c'], CalibMatrix['K_t'], Energies_Vec[:])
    return Energies_Vec

@jit(nopython=True, nogil=True, parallel=True)
def Matrix_Index_to_XY_Numba(Matrix_Ind):
    """
    A function that converts for single pixel its matrix index to (X,Y) coordinates.
    Pix_Ind is the index of pixel for which the XY value should be calculated.
    """
    Numpber_of_Pixels = Matrix_Ind.shape[0]
    XY = numpy.zeros((Numpber_of_Pixels,2),dtype=numpy.uint16)

    for PixInd in prange(Numpber_of_Pixels):
        if Matrix_Ind[PixInd] < (256*256):
            XY[PixInd,1] = int(Matrix_Ind[PixInd]/256) # Row, until 255
            XY[PixInd,0] = Matrix_Ind[PixInd] % 256 # Column, until 511

        else:
            Second_Chip_Offset = 256*256
            Pix_Ind_Temp = Matrix_Ind[PixInd] - Second_Chip_Offset
            XY[PixInd,1] = int(Pix_Ind_Temp/256) # Row, until 255
            XY[PixInd,0] = (Pix_Ind_Temp % 256) + 256# Column, until 511

    return XY

@guvectorize([(uint32[:],uint32[:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float32[:])],'(),(),(j,j),(j,j),(j,j),(j,j),(j,j),(j,j),(j,j),(j,j)->()',nopython=True, target='parallel')
def Pixel_To_Energy_Numba(Pix_Ind, ToT, La, Lb, Lc, Lt, Ka, Kb, Kc, Kt, Result):
    """
    A function that converts for single pixel, all the measured values of ToT to energy.
    ToT (time over threshold) is given in the raw data.
    Pix_Ind is the index of pixel for wich the energy should be calculated.
    CalibData is a dict() with calibration values from Advacam
    """
    if Pix_Ind[0] < (256*256):
        ShortAxis = int(Pix_Ind[0]/256) # Row
        LongAxis= Pix_Ind[0] % 256 # Column
        a = La[ShortAxis, LongAxis]
        b = Lb[ShortAxis, LongAxis]
        c = Lc[ShortAxis, LongAxis]
        t = Lt[ShortAxis, LongAxis]

        #xind = LongAxis # until 511
        #yind = ShortAxis # Until 255
    else:
        Second_Chip_Offset = 256*256
        Pix_Ind_Temp = Pix_Ind[0] - Second_Chip_Offset
        ShortAxis = int(Pix_Ind_Temp/256) # Row
        LongAxis= (Pix_Ind_Temp % 256) # Column
        a = Ka[ShortAxis, LongAxis]
        b = Kb[ShortAxis, LongAxis]
        c = Kc[ShortAxis, LongAxis]
        t = Kt[ShortAxis, LongAxis]

        #xind = LongAxis + 256 # until 511
        #yind = ShortAxis # Until 255

    f = ToT[0]
    Result[0] = (-(b - f - a * t) + ((b - f - a * t) ** 2 - 4 * a * (f * t - b * t - c)) ** (0.5)) / (2 * a)

#Need to shitch back to 'cuda'
@guvectorize([(uint32[:],uint32[:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float32[:])],'(),(),(j,j),(j,j),(j,j),(j,j),(j,j),(j,j),(j,j),(j,j)->()',nopython=True, target='parallel')
def Pixel_To_Energy_GPU(Pix_Ind, ToT, La, Lb, Lc, Lt, Ka, Kb, Kc, Kt, Result):
    """
    A function that converts for single pixel, all the measured values of ToT to energy.
    ToT (time over threshold) is given in the raw data.
    Pix_Ind is the index of pixel for wich the energy should be calculated.
    CalibData is a dict() with calibration values from Advacam
    """
    if Pix_Ind[0] < (256*256):
        ShortAxis = int(Pix_Ind[0]/256) # Row
        LongAxis= Pix_Ind[0] % 256 # Column
        a = La[ShortAxis, LongAxis]
        b = Lb[ShortAxis, LongAxis]
        c = Lc[ShortAxis, LongAxis]
        t = Lt[ShortAxis, LongAxis]

        #xind = LongAxis # until 511
        #yind = ShortAxis # Until 255
    else:
        Second_Chip_Offset = 256*256
        Pix_Ind_Temp = Pix_Ind[0] - Second_Chip_Offset
        ShortAxis = int(Pix_Ind_Temp/256) # Row
        LongAxis= (Pix_Ind_Temp % 256) # Column
        a = Ka[ShortAxis, LongAxis]
        b = Kb[ShortAxis, LongAxis]
        c = Kc[ShortAxis, LongAxis]
        t = Kt[ShortAxis, LongAxis]

        #xind = LongAxis + 256 # until 511
        #yind = ShortAxis # Until 255

    f = ToT[0]
    Result[0] = (-(b - f - a * t) + ((b - f - a * t) ** 2 - 4 * a * (f * t - b * t - c)) ** (0.5)) / (2 * a)


def Pixel_XY_To_Energy(yind,xind, ToT, CalibData):
    """
    A function that converts for single pixel, all the measured values of ToT to energy.
    ToT (time over threshold) is given in the raw data.
    xind is the x (column, long axis until 511)
    yind is the y (row, short axis until 255)
    xind and yind are the coordinates of a pixel for which the energy should be calculated.
    CalibData is a dict() with calibration values from Advacam
    """
    if xind < 256:
        a = CalibData['L_a'][yind, xind]
        b = CalibData['L_b'][yind, xind]
        c = CalibData['L_c'][yind, xind]
        t = CalibData['L_t'][yind, xind]

    else:

        xind_Temp = xind-256
        a = CalibData['K_a'][yind, xind_Temp]
        b = CalibData['K_b'][yind, xind_Temp]
        c = CalibData['K_c'][yind, xind_Temp]
        t = CalibData['K_t'][yind, xind_Temp]

    f = ToT
    PixEnrg = (-(b - f - a * t) + ((b - f - a * t) ** 2 - 4 * a * (f * t - b * t - c)) ** (0.5)) / (2 * a)

    return PixEnrg, (a,b,c,t)

def Pixel_Spectrum(RawData, PixelIndex, EnergyRange, TimeWindow, CalibCoeff):
    """
    Function that draws a spectrum in a single pixel.
    The raw Advapix data is given by RawData (data with 5 columns)
    The pixel index is given by PixelIndex
    The histogram range is given by EnergyRange
    TimeWindow is in nano-seconds and defines what to consider as single pixel
    CalibCoeff is a dict with calibration coefficients that is produced by Load_Calibration_Coef function
    """
    ChoosenPixelInd = RawData[:, 0] == PixelIndex
    SingletFilter = numpy.full(ChoosenPixelInd.shape,False)
    SingletInd = Separate_Single_Pixels(RawData[:,1],TimeWindow)
    SingletFilter[SingletInd] = True
    ChoosenPixelData = RawData[ChoosenPixelInd & SingletFilter,2]


    PixelCoordinates = Index_To_Pixels(numpy.array([PixelIndex,]))
    #PixelPlots = HPF.Plotting_Results_Class(AxTitle='Single pixel (x: {:0}, y: {:1}) spectrum'.format(PixelCoordinates[0,0],PixelCoordinates[0,1]))
    #PixelPlots.FigList[-1].axes[0].hist(ChoosenPixelData, list(range(EnergyRange)))

    # Load calibration data
    '''
    Coeff_A = numpy.loadtxt('/home/Haim/Desktop/MountPoint-NetworkDrive/Haim/Research/Equipment/2D_Detectors/Advacam/Calibration/L09-W0096_a.txt')
    Coeff_B = numpy.loadtxt('/home/Haim/Desktop/MountPoint-NetworkDrive/Haim/Research/Equipment/2D_Detectors/Advacam/Calibration/L09-W0096_b.txt')
    Coeff_C = numpy.loadtxt('/home/Haim/Desktop/MountPoint-NetworkDrive/Haim/Research/Equipment/2D_Detectors/Advacam/Calibration/L09-W0096_c.txt')
    Coeff_T = numpy.loadtxt('/home/Haim/Desktop/MountPoint-NetworkDrive/Haim/Research/Equipment/2D_Detectors/Advacam/Calibration/L09-W0096_t.txt')
    xind=PixelCoordinates[0,0]
    yind=PixelCoordinates[0,1]
    a = Coeff_A[xind, yind]
    b = Coeff_B[xind, yind]
    c = Coeff_C[xind, yind]
    t = Coeff_T[xind, yind]
    f=ChoosenPixelData
    CalibratedData = (-(b-f-a*t)+((b-f-a*t)**2-4*a*(f*t-b*t-c))**(0.5))/(2*a)
    '''
    CalibratedData = Pixel_To_Energy(PixelIndex,ChoosenPixelData, CalibCoeff)
    #PixelPlots.Add_Fig('Single pixel (Long axis: {:0}, Short axis: {:1}) calibrated spectrum'.format(CalibratedData[1],CalibratedData[2]))
    #PixelPlots.FigList[-1].axes[0].hist(CalibratedData[0], numpy.arange(0,int(CalibratedData[0].max()),0.5))
    #return PixelPlots
    return CalibratedData

def Region_Spectrum_New(RawData, LongAx_Ind, ShortAx_Ind, LongAx_Width, ShortAx_Width, TimeWindow,CalibCoeff):
    pass
# Old function - before charge sharing functions
def Region_Spectrum(RawData, LongAx_Ind, ShortAx_Ind, LongAx_Width, ShortAx_Width, TimeWindow,CalibCoeff):
    """
    A function the returns and plots spectrum of a region on the detector.
    The function receives raw data (5 columns), time window, region

    """

    # Find indexes of frames in the raw data according to a time window
    MultiPix_Events = Find_Frame_Indexes(RawData, TimeWindow)
    # Choose events with two pixels in the region in the time window
    DoublePix_Events = list()
    TotalPixelNum = list()
    TotalLoopLength = MultiPix_Events[0].shape[0]
    StartTimer = time.time()
    TotalTimer = StartTimer
    for LoopInd,Event_Ind in enumerate(zip(MultiPix_Events[0],MultiPix_Events[1])):
        # Take the x,y of only those pixels that arrive in the time window
        TempData = RawData[Event_Ind[0]:Event_Ind[1],:]
        ShortAx_Ind_Temp = (TempData[:,0]/256).astype(numpy.int16)
        LongAx_Ind_Temp = TempData[:,0]%256
        # Filter events with exactly two pixels in a region
        IndFilter = (ShortAx_Ind_Temp>ShortAx_Ind) & (ShortAx_Ind_Temp<ShortAx_Ind+ShortAx_Width) & (LongAx_Ind_Temp>LongAx_Ind) & (LongAx_Ind_Temp<LongAx_Ind+LongAx_Width)
        if numpy.count_nonzero(IndFilter) == 2:
            PixelDiff = numpy.abs( numpy.diff(TempData[IndFilter,0]) )
            # Check if pixels are adjacent and save charge sharing event
            if (PixelDiff == 1) | (PixelDiff == 256):
                DoublePix_Events.append(TempData[IndFilter])
            # Only for checking. Save all event with two pixels in the region but not only charge sharing
            TotalPixelNum.append(TempData)
        if time.time()-StartTimer > 10:
            print('Finished {0:.1f}% events'.format(100*(LoopInd+1)/TotalLoopLength))
            StartTimer = time.time()
    print('Finished selecting events with 2 photons. It took: {:0} seconds.'.format(time.time()-TotalTimer))


    # Energy histogram in region
    EnergyHist = numpy.zeros((len(DoublePix_Events),1),dtype=numpy.float16)
    for LoopInd,Event in enumerate(DoublePix_Events):
        E1 = Pixel_To_Energy(Event[0, 0], Event[0, 2], CalibCoeff)[0]
        E2 = Pixel_To_Energy(Event[1, 0], Event[1, 2], CalibCoeff)[0]
        EnergyHist[LoopInd] = E1+E2
    # Convert pixels to enrgies and sum

    return EnergyHist, MultiPix_Events, DoublePix_Events, TotalPixelNum

def Filter_Frames_Initial(FrameList, MinEnergy):
    '''
    A function that makes a list of frames that have at least one pixel at both halves with enegy below MinEnergy
    MinEnrgy in raw data units - number of clock pulses
    '''

    FilteredPixels = list()
    try:
        for SingleFrame in FrameList:
            if ((SingleFrame[:,0]<256*256) ).any() & ((SingleFrame[:,0]>256*256) ).any() :
                FilteredPixels.append(SingleFrame)
        return FilteredPixels
    except:
        print('Problem with initial filtration of frames. Check data structure, energy ranges etc.')

@jit(nopython=True, parallel=False)
def Filter_Frames_Initial_Numba(FrameList, MinEnergy):
    '''
    A function that makes a list of frames that have at least one pixel at both halves with enegy below MinEnergy
    MinEnrgy in raw data units - number of clock pulses
    '''

    FilteredPixels = list()
    try:
        for SingleFrame in FrameList:
            if ((SingleFrame[:,0]<256*256) ).any() & ((SingleFrame[:,0]>256*256) ).any() :
                FilteredPixels.append(SingleFrame)
        return FilteredPixels
    except:
        print('Problem with initial filtration of frames. Check data structure, energy ranges etc.')


def Filter_Frames_Initial_ROI(FrameList, ROI):
    '''
    A function that makes a list of frames that have at least one pixel at both halves with enegy below MinEnergy
    MinEnrgy in raw data units - number of clock pulses
    '''
    LStart = ROI['long start']
    LEnd = ROI['long end']
    SStart = ROI['short start']
    SEnd = ROI['short end']
    FilteredPixels = list()
    try:
        for SingleFrame in FrameList:
            TempFilter = (SingleFrame[:,1]<LEnd) & (SingleFrame[:,1]>LStart) & (SingleFrame[:,0]<SEnd) & (SingleFrame[:,0]>SStart)
            if (TempFilter).any():
                TempInd = numpy.nonzero(TempFilter)[0]
                FilteredPixels.append(SingleFrame[TempInd])
        return FilteredPixels
    except:
        print('Problem with initial filtration of frames. Check data structure, energy ranges etc.')

# Error with numba and cuda - should check why
'''
@guvectorize([(uint16[:,:],boolean[:],uint16,uint16,uint16,uint16,uint16,uint16,boolean[:])],'(raw,column),(raw),(),(),(),(),(),()->()',target='cuda')
def GPU_Filter_Frames(Frame,PixelsFilter,LeftStart,LeftEnd,RightStart,RightEnd,EnrgMin,EnrgMax,Result):

    for Ind in range(Frame.shape[0]):
        PixelsFilter[Ind] = (Frame[Ind,2]>=EnrgMin) & (Frame[Ind,2]<=EnrgMax)

    AnyLeft = False
    AnyRight = False

    for Ind in range(Frame.shape[0]):
        if PixelsFilter[Ind] == True:
            AnyLeft = AnyLeft | ((Frame[Ind,1]>=LeftStart) & (Frame[Ind,1]<=LeftEnd))
            AnyRight = AnyRight | ((Frame[Ind,1]>=RightStart) & (Frame[Ind,1]<=RightEnd))


    Result[0] = bool(AnyLeft & AnyRight)
'''
def OpenFrame(FrameParameters):
    FolderName = FrameParameters['Folder']
    FileName = FrameParameters['File']
    # return numpy.loadtxt('/home/Haim/Desktop/TEMP/Integrated_Frame_Event.txt')
    return numpy.loadtxt(FolderName+FileName)

def Find_Counter_Reset(RawData, CounterMaxSize):
    """
    Function that finds when the time counter goes to zero.
    RawData - 5 columns of raw data - pixel index, ToA, ToT, FToA, Overflow
    Returns the indexes in RawData
    CounterMaxSize - size in clock pulses of maximal counter value
    """
    RawDiff = numpy.diff(RawData[:,1].astype(numpy.int32))
    CounterIndexes = numpy.nonzero(RawDiff < int(-CounterMaxSize*0.5))[0]
    return CounterIndexes

def SortRawData(RawData,CounterResetIndex):
    """
    Sorts the raw data according to time stamps.
    The RawData is 5 column numpy array.
    There are events of time counter going to zero. The sorting is done separately for each section
    """
    BeginInd = numpy.insert(CounterResetIndex+1,0,0)
    EndInd = numpy.append(CounterResetIndex+1,RawData.shape[0])

    NewRawData = RawData.copy()

    for Ind in zip(BeginInd,EndInd):
        TempData = RawData[Ind[0]:Ind[1],:]
        SortedIndex = numpy.argsort(TempData[:,1])
        NewRawData[Ind[0]:Ind[1],:]=TempData[SortedIndex,:]

    return NewRawData

class AdvaOpener:
    def __init__(self):
        self.Pixels = None
        self.FrameListMain = None
        self.PixelsSorted = None
        self.FileList = list()
        self.FolderName = ''
        self.TestFigs = None
        self.TimeDiff = None
        self.SaveFolder = None
        self.SaveFile = None

    def LoadPixelData(self):
        #Read all the raw data files in self.FileList from folder self.FolderName
        # The file has 6 columns: Index, Matrix Index, ToA, ToT, FToA, Overflow
        DataList = list()
        LoopLength = len(self.FileList)
        LoopTimer = time.time()
        TotalTimer = time.time()
        for LoopInd,SingleFile in enumerate(self.FileList):
            if SingleFile.find('info')<0:
                DataList.append(numpy.loadtxt(self.FolderName + SingleFile,dtype=numpy.uint32,skiprows=1,usecols=(1,2,3,4,5)))
            if LoopTimer+10 < time.time():
                print('Finished loading {0:.2f}% of files.'.format(100*LoopInd/LoopLength))
                LoopTimer = time.time()
        self.Pixels = numpy.vstack(DataList)
        print('Finished loading files. It took {:0} seconds'.format(time.time()-TotalTimer))
        return self.Pixels

    def LoadPixelIndex(self):
        #Read all the first columns from files in self.FileList from folder self.FolderName
        #The file has 6 columns: Index, Matrix Index, ToA, ToT, FToA, Overflow
        DataList = list()
        for SingleFile in self.FileList:
            DataList.append(numpy.loadtxt(self.FolderName + SingleFile,dtype=numpy.uint32,skiprows=1,usecols=(0,)))
        return numpy.hstack(DataList)

    def TestTimeDiffRaw(self):
        """
        Plot histogram of time stamps differences and the time stampes
        """
        self.TimeDiff = numpy.diff(self.Pixels[:,1].astype(numpy.int32))
        if self.TestFigs == None:
            self.TestFigs = HPF.Plotting_Results_Class(AxTitle='Checking time differences')
        else:
            self.TestFigs.Add_Fig(AxTitle='Checking time differences')

        self.TestFigs.FigList[-1].axes[0].hist(self.TimeDiff*25,numpy.arange(-50000,50000,100))
        self.TestFigs.Add_Fig(AxTitle='Checking time')
        self.TestFigs.FigList[-1].axes[0].plot(self.Pixels[:,1]*25e-9)
        self.TestFigs.Show_All_Figs(False)

    def TestTimeDiffSorted(self):
        CounterSize = 2**28
        CounterIndexes = Find_Counter_Reset(self.Pixels,CounterSize)
        self.PixelsSorted = SortRawData(self.Pixels, CounterIndexes)
        self.TimeDiff = numpy.diff(self.PixelsSorted[:,1].astype(numpy.int32))
        if self.TestFigs == None:
            self.TestFigs = HPF.Plotting_Results_Class(AxTitle='Checking sorted time differences')
        else:
            self.TestFigs.Add_Fig(AxTitle='Checking sorted time differences')

        self.TestFigs.FigList[-1].axes[0].hist(self.TimeDiff*25,numpy.arange(-50000,50000,100))
        self.TestFigs.Add_Fig(AxTitle='Checking sorted time')
        self.TestFigs.FigList[-1].axes[0].plot(self.PixelsSorted[:,1]*25e-9)
        self.TestFigs.Show_All_Figs(False)

    def TestEnergyHist(self, TimeWindow):
        """
        TimeWindow in nanosec
        """
        EnergyRange = 200
        SinglePixelsIndex = Separate_Single_Pixels(self.Pixels[:,1],TimeWindow)

        if self.TestFigs == None:
            self.TestFigs = HPF.Plotting_Results_Class(AxTitle='Single pixels spectrum')
        else:
            self.TestFigs.Add_Fig(AxTitle='Single pixels spectrum')

        self.TestFigs.FigList[-1].axes[0].hist(self.Pixels[SinglePixelsIndex,2],list(range(EnergyRange)))

        ResetIndex = Find_Counter_Reset(self.Pixels, 2 ** 28)
        SortedRawData = SortRawData(self.Pixels, ResetIndex)
        self.TestFigs.Add_Fig(AxTitle='Single pixels sorted spectrum')
        SinglePixelsIndex = Separate_Single_Pixels(SortedRawData[:, 1], TimeWindow)
        self.TestFigs.FigList[-1].axes[0].hist(SortedRawData[SinglePixelsIndex, 2], list(range(EnergyRange)))

    def Proces_Raw_Data(self, TimeWindow, MinEnergy):
        """
        Method that loads raw data from FileList attribute, doing first filtration and saves in a new pikle file - SaveFile attribute
        Time window in nanosec
        """
        if (self.SaveFile == None) or (self.SaveFolder == None):
            print('Folder and file name for saving the data were not given')
            return
        try:
            FrameList = list()
            for Single_File in self.FileList:
                # Load numbers from text file Single_File into numpy array RawData
                StartTime = time.time()
                RawData = numpy.loadtxt(self.FolderName + Single_File, dtype=numpy.uint32, skiprows=1, usecols=(1, 2, 3, 4, 5))
                print('Loaded file {0:} after: {1:.1f} seconds.'.format(Single_File,time.time()-StartTime))

                ResetIndex = Find_Counter_Reset(RawData, 2 ** 28)
                SortedRawData = SortRawData(RawData, ResetIndex)

                # Use function Devide_To_Frames_List to separate pixels into list of frames(numpy arrays) with time window TimeWindow
                # The function Devide_To_Frames_List also:
                # - Takes only frames with at least one pixel in each of the detector.
                # - Does initial energy filtration (Compton and elastic)
                StartTime = time.time()
                TempFrameList = Devide_To_Frames_List(SortedRawData, TimeWindow)
                print('Divided {0:} pixels to {1:} frames in: {2:.1f} seconds.'.format(SortedRawData.shape[0], len(TempFrameList), time.time() - StartTime))
                # Collect all the frames from RawData into one large list of numpy arrays
                StartTime = time.time()
                Filtered_Frame_List = Filter_Frames_Initial(TempFrameList, MinEnergy)
                FrameList.extend(Filtered_Frame_List)
                print('Filtered from {0:} frames to {1:} frames in: {2:.1f} seconds.'.format(len(TempFrameList), len(Filtered_Frame_List),time.time() - StartTime))
            self.FrameListMain = FrameList
        except:
            print('Problem with loading the data. Check given folder and list of files.')

        try:
            # Save the list of frames into file with the name and folder in SaveFile, SaveFolder attributes
            DataSaverObj1 = HSF.DataSaver()
            DataSaverObj1.DestFile={'Folder Name':self.SaveFolder, 'File Name':self.SaveFile}
            DataSaverObj1.Save_Everything(FrameList)
        except:
            print('Problem with saving the data. Check the folder and file given for saving.')

    def Load_Proccesed_Data(self):
        # Load frame list from file SaveFile in folder SaveFolder
        DataOpenerObj = HFOF.DataLoader()
        DataOpenerObj.SourceFile = {'Folder Name':self.SaveFolder, 'File Name':self.SaveFile}
        return DataOpenerObj.LoadEverythingFromFile()