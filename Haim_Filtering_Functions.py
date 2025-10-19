import numpy
from numba import jit,njit,guvectorize,uint16,uint64,boolean, float64, prange
import itertools
from scipy.sparse import csr_array
import time
from joblib import Parallel, delayed
from multiprocessing import Pool
from math import sqrt,pow
import Haim_File_Opening_Functions as HFOF
import Haim_Saving_Functions as HSF
import Haim_Ploting_Functions as HPF

def Timing_Of_Function(InputFunc, Description,PrintOutFlag, *Params, **DicParams):
    """
    A function that measures execution time of other functions.
    Parameters:
        InputFunc - a function object
        Description - string with text that will be printed before and after execution of the function
        PrintOutFlag - True ot False if time measurement and printing are wanted
        *Params - all the parameters of the functions
        **DicParams - all the parameters in the dictionary fancy python way
    """
    if PrintOutFlag:
        print('Start ' + Description + '...:')
        StartTime = time.perf_counter()
        Out = InputFunc(*Params,**DicParams)
        EndTime = time.perf_counter()
        print('Finished ' + Description + ' after {0:.3f} seconds'.format(EndTime-StartTime))
    else:
        Out = InputFunc(*Params, **DicParams)
    return Out
def List_To_3Darray(FrameList):
    """
    Turn list of numpy arrays to 3D numpy array. Need to do this for using numba acceleration
    :param: FrameList
    """

    LongestFrame = 0
    for Frame in FrameList:
        if Frame.shape[0] > LongestFrame:
            LongestFrame = Frame.shape[0]
    New_3D_Array = numpy.zeros((len(FrameList),LongestFrame,3),dtype=numpy.uint16)

    for Ind in range(len(FrameList)):
        New_3D_Array[Ind,0:FrameList[Ind].shape[0],:] = FrameList[Ind]
    return New_3D_Array

###########
@guvectorize([(uint16[:,:],uint16,uint16,uint16,uint16,boolean[:])],'(raws,colum),(),(),(),()->(raws)',nopython=True,target='parallel')
def GPU_Find_Pairs(FrameArray,E_Min_1,E_Max_1,X_Min_1,X_Max_1,FilteredRows):
    for Ind in range(FrameArray.shape[0]):
        FilteredRows[Ind] = (FrameArray[Ind,2] >= E_Min_1) & (FrameArray[Ind,2] <= E_Max_1) & (FrameArray[Ind,0] >= X_Min_1) & (FrameArray[Ind,0] <= X_Max_1)

@guvectorize([(boolean[:],uint16[:])],'(frame)->()',nopython=True,target='parallel')
def Count_Pixels_GPU(FramePixels, PixelsCounter):
    """
    Counts number of pixels in a frame
    """
    PixelsCounter[0] = 0

    for Ind in range(FramePixels.shape[0]):
        if FramePixels[Ind]>0:
            PixelsCounter[0] += 1
#########

@jit(nopython=True, nogil=True, parallel=True)
def Filter_Array_Region_Numba(DataArr, EnrgMin, EnrgMax, XMin, XMax, YMin, YMax):
    """
    A Numba compiled function that generates boolean filter of pixels that potentially could be PDC pairs.
    Parameters:
        DataArr - a 3D array of the data. axis 0 is frames, axis 1 is pixels, axis 2 is x,y and energy.
        EnrgMax and EnrgMin - are the energy range of the potentially PDC pixels
        X and Y Min, Max - are the region of the potentially PDC pixels
    Returns:
         Boolean 2D array. axis 0 is frames, axis 1 is pixels in the frames that are potentially PDC.
    """
    FilterX = (DataArr[:,:,0] >= XMin) & (DataArr[:,:,0] <= XMax)
    FilterY = (DataArr[:, :, 1] >= YMin) & (DataArr[:, :, 1] <= YMax)
    FilterEnrg = (DataArr[:, :, 2] >= EnrgMin) & (DataArr[:, :, 2] <= EnrgMax)

    Filter = FilterX & FilterY & FilterEnrg
    return Filter
@jit(nopython=True, nogil=True, parallel=True)
def Generate_Frame_Filter_Numba(Filter1, Filter2):
    """
    Function that generates boolean filter of frames with potentially PDC pairs.
    A frame is chosen if it contains at least one pixel in two regions given by filters Filter1 and 2.
    Parameters:
        Filter1, Filter2 - boolean 2D arrays of frames and pixels in each frame that satisfy energy\position conditions
    Returns:
        Filter - boolean 1D vector with length as number of frames. True means there are potentially PDC pair.
    """
    NumberOfFrames = Filter1.shape[0]
    Filter = numpy.full(NumberOfFrames,False,dtype=numpy.bool_)
    for Ind in prange(NumberOfFrames):
        #Filter = Filter1.any(axis=1) & Filter2.any(axis=1)
        Filter[Ind] = Filter1[Ind,:].any() & Filter2[Ind,:].any()
    return Filter
@jit(nopython=True, nogil=True, parallel=True)
def Generate_Filtered_Array_Of_Frames_Numba_OLD(OldArray, NewArray, NumberOfFrames, FrameInd, Filter1, Filter2):
    """
    Old function.
    Function that generates new frames 3D array with only potentially PDC pixels given by Filter1 and 2.
    Parameters:
        OldArray - A 3D array of frames before filtration. axis 0 is frame, axis 1 is pixels inside frame and axis 2 are x,y and energy of pixel
        NumberOfFrames - number of frames from OldArray in which there are potentially PDC pairs.
        FrameInd - indexes of frames from OldArray in which there are potentially PDC pairs.
        NewArray - The new 3D array of frames. It will be filled with only potentially PDC pixels according to Filter1 and Filter2.
                    This array should be an array of zeros prepared in advance and given to this function as parameter
        Filter1 and 2 - boolean 2D arrays of filters of potentially PDC pixels. axis 0 is frame and axis 1 is pixel
    Returns:
        Doesn't return but writes into array NewArray

    """
    #def InnerFunc(NewAr, OldAr, FramLen, TempF):
    #    NewAr[:FramLen,:] = OldAr[TempF,:]

    for Ind in prange(NumberOfFrames):
        # Single_Frame_Ind - single index of frame from OldArray or Filter1,2
        Single_Frame_Ind =  FrameInd[Ind]
        # TempFilter is boolian 1D vector of all the pixels that should be saved from a single frame in OldArray
        TempFilter = Filter1[Single_Frame_Ind,:] | Filter2[Single_Frame_Ind,:]
        FrameLength = numpy.count_nonzero(TempFilter)
        #NewArray[Ind,:FrameLength, :] = OldArray[Single_Frame_Ind,TempFilter, :]

        Single_New_Frame = NewArray[Ind,:,:]
        Single_Old_Frame = OldArray[Single_Frame_Ind,:,:]
        Single_New_Frame[:FrameLength, :] = Single_Old_Frame[TempFilter, :]

@jit(nopython=True, nogil=True, parallel=True)
def Generate_Filtered_Array_Of_Frames_Numba(OldArray, NewArray, NumberOfFrames, FrameInd, Filter1, Filter2):
    """
    Function that generates new frames 3D array with only potentially PDC pixels given by Filter1 and 2.
    Parameters:
        OldArray - A 3D array of frames before filtration. axis 0 is frame, axis 1 is pixels inside frame and axis 2 are x,y and energy of pixel
        NumberOfFrames - number of frames from OldArray in which there are potentially PDC pairs.
        FrameInd - indexes of frames from OldArray in which there are potentially PDC pairs.
        NewArray - The new 3D array of frames. It will be filled with only potentially PDC pixels according to Filter1 and Filter2.
                    This array should be an array of zeros prepared in advance and given to this function as parameter
        Filter1 and 2 - boolean 2D arrays of filters of potentially PDC pixels. axis 0 is frame and axis 1 is pixel
    Returns:
        Doesn't return but writes into array NewArray
    """
    for Ind in prange(NumberOfFrames):
        # Single_Frame_Ind - single index of frame from OldArray or Filter1,2
        Single_Frame_Ind = FrameInd[Ind]
        # TempFilter is boolian 1D vector of all the pixels that should be saved from a single frame in OldArray
        TempFilter = Filter1[Single_Frame_Ind,:] | Filter2[Single_Frame_Ind,:]

        # New improvement:
        Single_Old_Frame = OldArray[Single_Frame_Ind]
        Single_Old_Frame = Single_Old_Frame[TempFilter,:]
        NewArray[Ind,:Single_Old_Frame.shape[0],:] = Single_Old_Frame

#@jit(nopython=True, nogil=True, parallel=True)
def Prepare_Parameters_For_New_Array(Filter, Filter1, Filter2, Numba_Flag=True, OutputFlag=True):
    """
    A function that prepares a new 3D array of frames that will be filled only with the potentially PDC pixels.
    Parameters:
        Filter - boolean 1D filter array of only the frames that contain potentially PDC pixels.
        Filter1, 2 - boolean 2D filter arrays with the pixels that potentially could be PDC pixels. axis 0 is the frame and axis 1 is the pixels in each frame.
    Returns:
        FrameInd - index of frames in which there are potentially PDC pair (given by Filter).
        NumberOfFrames - number of frames in which there are potentially PDC pairs (length of FrameInd)
        MaxFrameLength - the longest frame. The frames with the largest number of pixels given by Filter1 and 2.
        NewData - 3D array of zeros with the size of the new array that will contain only the pixels that potentially could by PDC pairs.
    """
    if OutputFlag:
        print('\nPreparing to create new array with frames of only pairs:\n')
    FrameInd = Timing_Of_Function(InputFunc=Filter.nonzero,Description='finding indexes of frames with at least one pair',PrintOutFlag=OutputFlag)
    FrameInd = FrameInd[0]

    if (FrameInd.shape[0] == 0):
        if OutputFlag:
            print('Didn\'t find pairs')
        return 0,0,0,0


    StartTime = time.perf_counter()
    if Numba_Flag:
        Filter1_Frames_Length = numpy.zeros(Filter1.shape[0], numpy.uint16)
        Filter2_Frames_Length = numpy.zeros(Filter2.shape[0], numpy.uint16)
        Count_Pixels_GPU(Filter1, Filter1_Frames_Length)
        Count_Pixels_GPU(Filter2, Filter2_Frames_Length)
        Filter1_Frames_Length = Filter1_Frames_Length[Filter]
        Filter2_Frames_Length = Filter2_Frames_Length[Filter]

    else:
        Filter1_Frames_Length = numpy.count_nonzero(Filter1[Filter],axis=1)
        Filter2_Frames_Length = numpy.count_nonzero(Filter2[Filter],axis=1)

    EndTime = time.perf_counter()
    if OutputFlag:
        print('Found length of frames in regions {0:.2f} seconds'.format(EndTime-StartTime))

    StartTime = time.perf_counter()
    Frames_Length = Filter1_Frames_Length + Filter2_Frames_Length
    EndTime = time.perf_counter()
    if OutputFlag:
        print('Found total length of frames {0:.2f} seconds'.format(EndTime - StartTime))

    MaxFrameLength = Timing_Of_Function(InputFunc=Frames_Length.max,Description='finding maximal frame length',PrintOutFlag=OutputFlag)

    #NumberOfFrames = numpy.count_nonzero(Filter)
    NumberOfFrames = FrameInd.shape[0]
    StartTime = time.perf_counter()
    NewData = numpy.zeros((NumberOfFrames,MaxFrameLength,3),dtype=numpy.uint32) #Changed to u32 instead of u16 because energy could be large
    EndTime = time.perf_counter()
    if OutputFlag:
        print('Prepared empty new array after {0:.2f} seconds'.format(EndTime-StartTime))

    return FrameInd, MaxFrameLength, NumberOfFrames, NewData

def Generate_EnergyConserving_Array(NewData, Pump_Energy, Tolerance):
    """
    A fuction that chooses only frames with exactly 2 pixels from the new filtered data array.
    It calculates the energy sum and compares with the pump energy and tolerance.
    Parameters:
    Returns:
        PDC_Pixels - 3D array of x,y of pixels with PDC. axis 0 is frame, axis 1 2 pixels, axis 2 are x and y of each pixel
        NumberOfPixels - number of pixels in each frame.
        FilterOfFramesWithPairs - boolean filter of frames with PDC (exactly 2 pixels that conserve energy).
        EnergySum - 1D array of length of number of frames in FilterOfFramesWithPairs with the energy sum of the 2 pixels in the frame.
        EnergyConservingFilter - boolean filter of frames that conserve energy.
    """
    print('Start find frames with only pairs')
    StartTime = time.perf_counter()
    NumberOfPixels = numpy.count_nonzero(NewData[:, :, 2], axis=1)
    FilterOfFramesWithPairs = NumberOfPixels == 2
    EndTime = time.perf_counter()
    print('Finished finding frames with only pairs after {0:.2f} seconds.'.format(EndTime - StartTime))

    print('Start summing energy of pairs...')
    StartTime = time.perf_counter()
    EnergySum = numpy.zeros((FilterOfFramesWithPairs.shape), dtype=numpy.uint32) # changed to int32 from uint16 since energy sum could be higher
    EnergySum[FilterOfFramesWithPairs] = numpy.sum(NewData[FilterOfFramesWithPairs, :, 2], axis=1)
    EndTime = time.perf_counter()
    print('Finished summing energy of pairs after {0:.2f} seconds'.format(EndTime - StartTime))
    print('')
    print('Start picking pixels that conserve energy...')
    StartTime = time.perf_counter()
    EnergyConservingFilter = (EnergySum >= Pump_Energy-Tolerance) & (EnergySum <= Pump_Energy+Tolerance)
    PDC_Pixels = NewData[EnergyConservingFilter, :2, :2]
    EndTime = time.perf_counter()
    print('Finished picking pixels that conserve energy after {0:.2f} seconds'.format(EndTime - StartTime))
    print('\nStatistics:')
    NumberOfFramesWithPairs = numpy.count_nonzero(FilterOfFramesWithPairs)
    NumberOfPDCPairs = numpy.count_nonzero(EnergyConservingFilter)
    if NumberOfFramesWithPairs == 0:
        return 0, NumberOfPixels, FilterOfFramesWithPairs
    print('Frames with exactly 2 pixels from all the data: {0:} from {1:} ({2:.2f}%)'.format(NumberOfFramesWithPairs,NewData.shape[0],NumberOfFramesWithPairs/NewData.shape[0]*100))
    print('Number of energy conserving frames {0:} from frames with pairs {1:} ({2:.2f}%)'.format(NumberOfPDCPairs,NumberOfFramesWithPairs,NumberOfPDCPairs/NumberOfFramesWithPairs*100))
    return PDC_Pixels, NumberOfPixels, FilterOfFramesWithPairs, EnergySum, EnergyConservingFilter

def Filter_Pairs_Array_Numba(DataArr, Enrg1Min, Enrg1Max, X1Min, X1Max, Y1Min, Y1Max,Enrg2Min, Enrg2Max, X2Min, X2Max, Y2Min, Y2Max, OutputFlag):
    """
    A function that takes 3D array of frames and generates a new smaller 3D array of frames.
    It chooses frames with at least one pixel in each region given by X,Y Min and Max and within energy range Enrg1Min, Enrg2Man etc.
    It saves into NewData only pixels and frames that satisfy all the condition.
    NewData has less frames and less pixels in each frame.

    """
    Filter1 = Timing_Of_Function(Filter_Array_Region_Numba,'generating boolean filter of first region',OutputFlag ,DataArr, Enrg1Min, Enrg1Max, X1Min, X1Max, Y1Min, Y1Max)
    Filter2 = Timing_Of_Function(Filter_Array_Region_Numba,'generating boolean filter of second region',OutputFlag ,DataArr, Enrg2Min, Enrg2Max, X2Min, X2Max, Y2Min, Y2Max)

    # Should do the following line in a separate function because Numba doesn't know any() with axis=1 argument
    Filter = Timing_Of_Function(Generate_Frame_Filter_Numba, 'generating filter of frames from data', OutputFlag, Filter1, Filter2)

    #print('Preparing for final filtration (finding parameters and creating matrix)...')
    #StartTime = time.perf_counter()
    #FrameInd, MaxFrameLength, NumberOfFrames, NewData = Prepare_Parameters_For_New_Array(Filter, Filter1, Filter2,OutputFlag=OutputFlag)
    FrameInd, MaxFrameLength, NumberOfFrames, NewData = Timing_Of_Function(Prepare_Parameters_For_New_Array,'preparing for final filtration (finding parameters and creating matrix)',OutputFlag,Filter, Filter1, Filter2,OutputFlag=OutputFlag)
    if type(NewData) == int:
        return 0,Filter,Filter1,Filter2
    #EndTime = time.perf_counter()
    #print('Finished preparation for final filtration after {0:.2f} seconds.'.format(EndTime-StartTime))

    Timing_Of_Function(Generate_Filtered_Array_Of_Frames_Numba, 'creating new 3D array with filtered data', OutputFlag, DataArr, NewData, NumberOfFrames, FrameInd, Filter1, Filter2)
    if OutputFlag:
        print('Statistics of initial filtration of frames:')
        print('Frames with potential PDC pairs {0:.2f}% ({1:} frames from {2:} frames)'.format(NewData.shape[0]/DataArr.shape[0]*100,NewData.shape[0], DataArr.shape[0]))

    return NewData, Filter, Filter1, Filter2

@jit(nopython=True, nogil=True, parallel=True)
def Remove_Fluorescence_Numba(DataArr, EnergyMin, EnergyMax):
    NumberOfFrames = DataArr.shape[0]
    for Ind in prange(NumberOfFrames):
        Filter = (DataArr[Ind,:,2] >= EnergyMin) & (DataArr[Ind, :,2] <= EnergyMax)
        for BoolInd in range(Filter.shape[0]):
            if Filter[BoolInd]:
                DataArr[Ind, BoolInd, 0] = 0
                DataArr[Ind, BoolInd, 1] = 0
                DataArr[Ind, BoolInd, 2] = 0



def FilterArray_Test_Timing(MainArr, FilterParam):
    MinEnergyL = FilterParam['Minimal Left Energy']
    MaxEnergyL = FilterParam['Maximal Left Energy']
    MinEnergyR = FilterParam['Minimal Right Energy']
    MaxEnergyR = FilterParam['Maximal Right Energy']
    MinLeft = FilterParam['Smallest Left Pixel']
    MaxLeft = FilterParam['Largest Left Pixel']
    MinRight = FilterParam['Smallest Right Pixel']
    MaxRight = FilterParam['Largest Right Pixel']
    BotLimL = FilterParam['Left Bottom Pixel']
    UpLimL = FilterParam['Left Top Pixel']
    BotLimR = FilterParam['Right Bottom Pixel']
    UpLimR = FilterParam['Right Top Pixel']

    FilteredPixelsList = []

    for FrameData in MainArr:
        EnergyFilterL = (FrameData[:,2] >= MinEnergyL) & (FrameData[:,2]<=MaxEnergyL)
        EnergyFilterR = (FrameData[:, 2] >= MinEnergyR) & (FrameData[:, 2] <= MaxEnergyR)
        PosFilterL = (FrameData[:,0] >= MinLeft) & (FrameData[:,0]<= MaxLeft)
        PosFilterR = (FrameData[:,0] >= MinRight) & (FrameData[:,0]<= MaxRight)
        PosFilterVerticalR = (FrameData[:,1] >= BotLimR) & (FrameData[:,1] <= UpLimR)
        PosFilterVerticalL = (FrameData[:, 1] >= BotLimL) & (FrameData[:, 1] <= UpLimL)

        LeftFilter = EnergyFilterL & PosFilterL & PosFilterVerticalL
        RightFilter = EnergyFilterR & PosFilterR & PosFilterVerticalR

        if LeftFilter.any() and RightFilter.any():
            ChoosenPixels = (LeftFilter) | (RightFilter)
            FilteredPixelsList.append(FrameData[ChoosenPixels,:])
    return FilteredPixelsList
def FilterArray(MainArr, FilterParam, DetectorParam={'Calibration':1}):
    CalibFactor=DetectorParam['Calibration']
    MinEnergyL = FilterParam['Minimal Left Energy']
    MaxEnergyL = FilterParam['Maximal Left Energy']
    MinEnergyR = FilterParam['Minimal Right Energy']
    MaxEnergyR = FilterParam['Maximal Right Energy']
    MinLeft = FilterParam['Smallest Left Pixel']
    MaxLeft = FilterParam['Largest Left Pixel']
    MinRight = FilterParam['Smallest Right Pixel']
    MaxRight = FilterParam['Largest Right Pixel']
    BotLimL = FilterParam['Left Bottom Pixel']
    UpLimL = FilterParam['Left Top Pixel']
    BotLimR = FilterParam['Right Bottom Pixel']
    UpLimR = FilterParam['Right Top Pixel']

    FilteredFrameList = []
    FilteredPixelsList = []
    StartTime = time.time()
    TotalLen = len(MainArr)

    for FrameInd,FrameData in enumerate(MainArr):
        EnergyFilterL = (FrameData[:,2] >= MinEnergyL) & (FrameData[:,2]<=MaxEnergyL)
        EnergyFilterR = (FrameData[:, 2] >= MinEnergyR) & (FrameData[:, 2] <= MaxEnergyR)
        PosFilterL = (FrameData[:,0] >= MinLeft) & (FrameData[:,0]<= MaxLeft)
        PosFilterR = (FrameData[:,0] >= MinRight) & (FrameData[:,0]<= MaxRight)
        PosFilterVerticalR = (FrameData[:,1] >= BotLimR) & (FrameData[:,1] <= UpLimR)
        PosFilterVerticalL = (FrameData[:, 1] >= BotLimL) & (FrameData[:, 1] <= UpLimL)

        LeftFilter = EnergyFilterL & PosFilterL & PosFilterVerticalL
        RightFilter = EnergyFilterR & PosFilterR & PosFilterVerticalR

        if LeftFilter.any() and RightFilter.any():
            FilteredFrameList.append(FrameInd)
            ChoosenPixels = (LeftFilter) | (RightFilter)
            FilteredPixelsList.append(FrameData[ChoosenPixels,:])
        #Show progres
        #if (time.time() > StartTime + 10) or (FrameInd < 1):
        if (time.time() > StartTime + 10):
            MyString = 'Filtered {:,} ({:.1%}) of {:,} frames'
            print(MyString.format(FrameInd,FrameInd / TotalLen, TotalLen))
            StartTime = time.time()

    return FilteredFrameList,FilteredPixelsList

def ArrayToSpars(PixelsArray,DetParameters={'Vertical Pixels':264, 'Horizontal Pixels':264}):
    '''
    PixelsArray - all pixels (x,y,energy) as one long numpy array
    ImageShape - shape of image
    Return: sparse row array
    '''
    YPix = DetParameters['Vertical Pixels']
    XPix = DetParameters['Horizontal Pixels']
    PixelsSparse=csr_array((numpy.ones(PixelsArray.shape[0],dtype=numpy.uint8),(PixelsArray[:,1],PixelsArray[:,0])),shape=(YPix,XPix))
    return PixelsSparse

def ListToArray(FilteredPixelsFrameList):
    FilteredPixels = numpy.vstack(FilteredPixelsFrameList)

    return FilteredPixels

def FilterRegion(FrameList, FilterParam={'Left':1,'Right':10,'Top':10,'Bottom':1,'Enrg Max':21000,'Enrg Min':0},DetectorParam={'Calibration':1}):
    '''
    FrameList - list of numpy arrays. Each numpy array is a frame.
    FilterParam - dict with keys: Left, Right, Top, Buttom, Enrg Max, Enrg Min
    Return list of frames(numpy arrays) with only pixels that satisfy FilterParam
    '''
    try:
        LPix=FilterParam['Left']
        RPix=FilterParam['Right']
        TPix=FilterParam['Top']
        BPix = FilterParam['Bottom']
        MinEnrg=FilterParam['Enrg Min']
        MaxEnrg = FilterParam['Enrg Max']

        FilteredRegion=[]
        StartTime = time.time()
        TotalLen = len(FrameList)

        for Ind, SingleFrame in enumerate(FrameList,start=1):
            EnergyFilter = (SingleFrame[:,2] >= MinEnrg) & (SingleFrame[:,2] <= MaxEnrg)
            HorizontalFilter = (SingleFrame[:,0] >= LPix) & (SingleFrame[:,0] <= RPix)
            VerticalFilter = (SingleFrame[:,1] >= BPix) & (SingleFrame[:,1] <= TPix)
            TotalFilter = EnergyFilter & HorizontalFilter & VerticalFilter
            FilteredRegion.append(SingleFrame[TotalFilter,:])
            if (time.time()>StartTime+10) or (Ind < 3):
                MyString='Filtered {:,} ({:.1%}) of {:,} frames'
                print(MyString.format(Ind, Ind/TotalLen, TotalLen))
                StartTime=time.time()

        return FilteredRegion
    except:
        print('Something is wrong with the parameters given to the function FilterRegion')


def PixelsToSquare(FrameList,Parameters):
    '''
    Reshape the readout from the pnCCD to square
    Should move this function to library Haim_pnCCD_RawData_Functions
    '''
    HorPixSize=Parameters['Horizontal Pixels']
    VerPixSize=Parameters['Vertical Pixels']
    for Frame in FrameList:
        #PixInd=Frame[:,0]>263
        PixInd = Frame[:, 0] > (HorPixSize-1)
        #Frame[PixInd,1]=263-Frame[PixInd,1]
        Frame[PixInd, 1] = (VerPixSize-1) - Frame[PixInd, 1]
        #Frame[PixInd,0]=527-Frame[PixInd,0]
        Frame[PixInd, 0] = (HorPixSize*2-1) - Frame[PixInd, 0]

def Prep_For_Enrgy_Histogram(Data, EnergyResolution, StartEnergy, StopEnergy):
    EnergyBins = numpy.arange(StartEnergy-int(EnergyResolution/2),StopEnergy+EnergyResolution,EnergyResolution)
    HistData, HistBins = numpy.histogram(Data, bins=EnergyBins,)
    return HistData, HistBins[:-1]

def Prepare_For_Plotting_Array_V2(FrameList,ImageShape = None):
    try:
        HorPixSize = ImageShape['Horizontal Pixels']
        VerPixSize = ImageShape['Vertical Pixels']
    except:
        print('Default image size: 132 rows, 264 columns.')
        HorPixSize = 264
        VerPixSize = 132

    try:
        PixelsArray = numpy.vstack(FrameList)
        # Should count only pixels with energy > 0. Otherwise count many (0,0)
        if PixelsArray.shape[1]==3:
            Filter = PixelsArray[:,2]>0
            # uint8 is not enough
            #PixelsCumulat = csr_array((numpy.ones(PixelsArray.shape[0], dtype=numpy.uint16), (PixelsArray[:, 1], PixelsArray[:, 0])),shape=(VerPixSize, HorPixSize))
            # By using histogram2d:
            PixelsCumulat = numpy.histogram2d(PixelsArray[Filter, 1], PixelsArray[Filter, 0], [numpy.arange(-0.5, VerPixSize + 0.5), numpy.arange(-0.5, HorPixSize + 0.5)])
        else:
            PixelsCumulat = numpy.histogram2d(PixelsArray[:, 1], PixelsArray[:, 0], [numpy.arange(-0.5, VerPixSize + 0.5), numpy.arange(-0.5, HorPixSize + 0.5)])

        return PixelsCumulat[0]
    except:
        print('Error in method Prepare_For_Plotting. Check the attribute FrameList. Should be list of numpy arrays.')

def FrameStatistics(FrameList,FilterParam = None):
    PixNum = 0
    MoreThanTwo = 0
    ExactTwo = 0
    if FilterParam == None:
        PixelArea = 0
        AreaIn_mm = 0
    else:
        PixelArea = FilterParam['PixelArea']
        AreaIn_mm = (FilterParam['Right']-FilterParam['Left']+1) * (FilterParam['Top']-FilterParam['Bottom']+1) * PixelArea

    StartTime=time.time()
    TotalLen=len(FrameList)
    for Ind, SingFrame in enumerate (FrameList,start=1):
        FramePixNum = SingFrame.shape[0]
        #Calculate the total number of photons in all the frames
        PixNum += FramePixNum
        if FramePixNum > 2:
            MoreThanTwo += 1
        elif FramePixNum ==2:
            ExactTwo += 1

        if time.time()>StartTime+10:
            MyString='Analyzed {:.1%} of {:,} frames'
            print(MyString.format(Ind/TotalLen,TotalLen))
            StartTime=time.time()
    if FilterParam != None:
        print('The total area is:{}'.format(AreaIn_mm))
    #return (PixNum/FrameList.__len__())/AreaIn_mm
    return PixNum, MoreThanTwo, AreaIn_mm, ExactTwo

def FrameStatistics2(FrameList,MinEnergy, MaxEnergy):
    #Energy conservation
    PairsNum=list()
    ConservingPairs=list()
    for FrameInd,SingFrame in enumerate(FrameList):
        if SingFrame.shape[0] == 2:
            PairsNum.append(FrameInd)

            EnergySum=SingFrame[:,2].sum()
            if EnergySum > MinEnergy and EnergySum < MaxEnergy:
                ConservingPairs.append(FrameInd)
    return PairsNum,ConservingPairs

def BadPixelsFromFile(FrameList,Parameters):
    #New function: Loading bad pixels from file and producing array (2D image)
    Param=dict()
    Param['Vertical Pixels'] = int(264 / 2)
    Param['Horizontal Pixels'] = 264

    FileName = '/home/Haim/Desktop/MountPoint-NetworkDrive/Haim/Research/Experiments/Desy_07_2022/pnCCD_CalibData/BadPixelMap_1000HzB.bpx'
    BadPix = numpy.loadtxt(FileName,dtype=numpy.uint16, usecols=(1, 2))
    PixelsToSquare([BadPix,],Param)
    BadPix_Spars = ArrayToSpars(BadPix)
    return BadPix_Spars.toarray()


def PairsScanStep(FrameList, FilterParameters, HStart,HStop,VStart,VStop):
    FilterParameters['Smallest Right Pixel'] = HStart
    FilterParameters['Largest Right Pixel'] = HStop
    FilterParameters['Right Bottom Pixel'] = VStart
    FilterParameters['Right Top Pixel'] = VStop

    FilteredFrameIndexesList, FilteredPixelsFrameList = FilterArray(FrameList, FilterParameters)
    PairsInd, ConservingPairsInd = FrameStatistics2(FilteredPixelsFrameList, 0, 100000)

    ScanStepResult=list()
    ScanStepResult.append(FilterParameters)
    ScanStepResult.append(len(ConservingPairsInd))
    ScanStepResult.append(list(map(FilteredPixelsFrameList.__getitem__, ConservingPairsInd)))

    print('Finished step with right region: {},{} - {},{}'.format(FilterParameters['Smallest Right Pixel'], FilterParameters['Right Bottom Pixel'], FilterParameters['Largest Right Pixel'], FilterParameters['Right Top Pixel']))
    # ScanStepResult is a list of 3  objects that consists of:filter parameters, number of pairs that was found, list of frames with the pairs
    return ScanStepResult

def PairsSearch(FrameList, FilterParameters, DetParameters):
    if (FilterParameters['Do Scan']):
        HStartVal = FilterParameters['Right Horizontal Scan Start']
        HEndVal = FilterParameters['Right Horizontal Scan End']
        HStep = FilterParameters['Right Horizontal Scan Step']
        VStartVal = FilterParameters['Right Vertical Scan Start']
        VEndVal = FilterParameters['Right Vertical Scan End']
        VStep = FilterParameters['Right Vertical Scan Step']

        HorizVector = range(HStartVal,HEndVal,HStep)
        HLen = len(HorizVector)
        VerticVector = range(VStartVal,VEndVal,VStep)
        VLen = len(VerticVector)

        ScanResultsList = Parallel(n_jobs=16)(delayed(PairsScanStep)(FrameList,FilterParameters,HorizVector[Hind],HorizVector[Hind+1],VerticVector[Vind],VerticVector[Vind+1]) for Hind in range(HLen-1) for Vind in range(VLen-1))

        return ScanResultsList

    else:
        FilteredFrameIndexesList, FilteredPixelsFrameList = FilterArray(FrameList, FilterParameters)

        Statistics = FrameStatistics(FilteredPixelsFrameList)
        PairsInd, ConservingPairsInd = FrameStatistics2(FilteredPixelsFrameList, 0, 100000)

        ListOfPairs = list(map(FilteredPixelsFrameList.__getitem__, PairsInd))
        ListOfConserving = list(map(FilteredPixelsFrameList.__getitem__, ConservingPairsInd))

        PairsArray = ListToArray(ListOfPairs)
        ConservingArray = ListToArray(ListOfConserving)

        PairsSparse = ArrayToSpars(PairsArray, DetParameters)
        ConservingSparse = ArrayToSpars(ConservingArray, DetParameters)

    return FilteredPixelsFrameList, ListOfPairs, ListOfConserving, PairsSparse, ConservingSparse, Statistics

def BinFrame(FrameData,BinParam = None):
    '''
    Bin the frame.
    FrameData - 2D frame. numpy array.
    BinParam - dict of the vertical and horizontal required binning.
    returns - the binned frame. numpy array.
    '''
    if BinParam == None:
        print('No binning parameters were given.\nReturning the given frame unchanged.')
        return FrameData
    else:
        vBin = BinParam['vertical binning']
        hBin = BinParam['horizontal binning']
        NewVertSize = int( FrameData.shape[0] / vBin )
        NewHorizSize = int( FrameData.shape[1] / hBin )
        BinnedFrame = FrameData.reshape(NewVertSize, vBin, NewHorizSize, hBin)
        BinnedFrame = BinnedFrame.sum(axis=(1, 3))
    return BinnedFrame

def UnBin(FrameData, BinParam = None):
    '''
    Unbin the frame.
    FrameData - 2D frame. numpy array.
    BinParam - dict of the vertical and horizontal required unbinning.
    returns - the unbinned frame. numpy array.
    '''
    if BinParam == None:
        print('No binning parameters were given.\nReturning the given frame unchanged.')
        return FrameData
    else:
        vBin = BinParam['vertical binning']
        hBin = BinParam['horizontal binning']
        MegaPixelMatrix = numpy.ones((vBin,hBin))
        UnBinnedFrame = numpy.kron(FrameData,MegaPixelMatrix)

    return UnBinnedFrame

def Pairs_List_to_Image(List_Of_Pairs,VBin=4, HBin=8):
    """
    A function that receives list of numpy arrays of pairs and returns binned image of them
    """

    if len(List_Of_Pairs) > 0:
        All_Pairs = numpy.vstack(List_Of_Pairs)
        Image_Of_Pairs = Prepare_For_Plotting_Array_V2(All_Pairs,ImageShape={'Horizontal Pixels':264,'Vertical Pixels':132})
        BinnedImage = BinFrame(Image_Of_Pairs,{'vertical binning':VBin, 'horizontal binning':HBin})
        return BinnedImage
    else:
        print('The list of 3D arrays with pairs is empty')

def ScanArea(DataArray, Energy_Region1, Energy_Region2, PumpEnergy, BinningSteps, Fixed_Area_Coordinates,Fixed_Area_Size, Scan_Rows_Binned, Scan_Columns_Binned, Scan_Steps,EnergyTolerance):
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
        EnergyTolerance - tolerance for energy conservation in eV
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
    Fixed_End_Row = Fixed_Begin_Row + Fixed_Area_Size[0] - 1 # Minus 1 at the end because the filtering function takes also the last row
    Fixed_Begin_Column = Fixed_Area_Coordinates[1] * BinningSteps[1]
    Fixed_End_Column = Fixed_Begin_Column + Fixed_Area_Size[1] - 1 # Minus 1 at the end because the filtering function takes also the last column
    ResultsList = list()
    print('Fixed region is:\n\tRows {}-{}\n\tColumns{}-{}\n'.format(Fixed_Begin_Row,Fixed_End_Row,Fixed_Begin_Column,Fixed_End_Column))
    for Current_Row in Rows_Scan:
        Current_Row_End = Current_Row + Scan_Row_Step - 1 # Minus 1 at the end because the filtering function takes also the last row
        print('Scanning rows {}-{} (inclusive)'.format(Current_Row,Current_Row_End))
        for Current_Column in Columns_Scan:
            Current_Column_End = Current_Column + Scan_Column_Step - 1 # Minus 1 at the end because the filtering function takes also the last column
            print('Scanning columns {}-{} (inclusive)'.format(Current_Column,Current_Column_End),end='\r')
            FilteredArray_First = Filter_Pairs_Array_Numba(DataArray, Energy_Region1[0], Energy_Region1[1], Fixed_Begin_Column, Fixed_End_Column, Fixed_Begin_Row, Fixed_End_Row,Energy_Region2[0], Energy_Region2[1], Current_Column, Current_Column_End, Current_Row, Current_Row_End, False)[0]
            if type(FilteredArray_First) != int:
                FilteredArray_Second = Generate_EnergyConserving_Array(FilteredArray_First, PumpEnergy, EnergyTolerance)[0]
                if type(FilteredArray_Second) != int:
                    ResultsList.append(FilteredArray_Second)
        print('')

            # Testing
            # DataArray[Current_Row,Current_Column] = int(str(Current_Row)+str(Current_Column))
    return ResultsList

def SubtractFromFrame(Frame,Const):
    NewFrame = numpy.zeros(Frame.shape, numpy.int32)
    NewFrame[:] = Frame[:] - Const
    NewFrame[NewFrame<0] = 0
    return NewFrame

def SubtractFromAreas(List_Of_Pairs,Const):
    # Hard coded for 33 over 33 frame
    """
    A function that returns 2D numpy array. It is the binned image of pairs in List_Of_Pairs while reducing the correct number of background.
    List_Of_Pairs is list of lists. Each item in the list is list of squares with pairs for a single fixed square. From each such list the background value is subtracted.
    """
    if len(List_Of_Pairs) > 0:
        Noise_Subtracted_Frame = numpy.zeros((33,33),numpy.int32)
        for Ind in range(len(List_Of_Pairs)):
            Small_Area_Frame = Prepare_For_Plotting_Array_V2(List_Of_Pairs[Ind])
            Binned_Small_Area = BinFrame(Small_Area_Frame, {'vertical binning': 4, 'horizontal binning': 8})
            Binned_Small_Area[Binned_Small_Area>0] = List_Of_Pairs[Ind].shape[0] - numpy.int32(Const)
            Noise_Subtracted_Frame += Binned_Small_Area.astype(numpy.int32)
        return Noise_Subtracted_Frame
    else:
        print('The list of 3D arrays with pairs is empty')
        return None

def Calc_Center_Mass(BinnedImage, Center):
    """
    A function that calculates center of intensity in two halves of an image
    The image should have small illuminated regions on the right and the left
    The center is calaculated by weighted averages
    Args:
        BinnedImage: 2D numpy array with image. Built for regions of PDC correlations
        Center: the x (column) of the center of the image. It is required to devide to two groups of pixels that will be averaged

    Returns:
        x and y of the left and right centers
    """
    Coordinates = numpy.nonzero(BinnedImage)
    X_val = Coordinates[1]
    Y_val = Coordinates[0]
    Intens_val = BinnedImage[Y_val, X_val]

    Right_Pixels = X_val > Center

    #Right_X = numpy.sum(X_val[Right_Pixels]) / Right_Pixels_Length
    #Right_Y = numpy.sum(t2[Right_Pixels,1]) / Right_Pixels_Length
    Right_X = numpy.average(a=X_val[Right_Pixels],weights=Intens_val[Right_Pixels])
    Right_Y = numpy.average(a=Y_val[Right_Pixels],weights=Intens_val[Right_Pixels])

    Left_Pixels = ~Right_Pixels

    Left_X = numpy.average(a=X_val[Left_Pixels],weights=Intens_val[Left_Pixels])
    Left_Y = numpy.average(a=Y_val[Left_Pixels], weights=Intens_val[Left_Pixels])

    return Right_X, Right_Y, Left_X, Left_Y


def Choose_from_Scan(FileParam,Rows,Columns, Background = 0):
    """
    A function that receives a list of 3D numpy arrays and ranges of columns and rows.
    Each 3D numpy array is all the pairs (all the frames) in two small areas.
    It checks all the 3D numpy arrays to find 3D arrays with pixels in the range.
    It returns a list of filtered 3D numpy arrays
    Rows and Columns are NOT binned
    """
    SelectedAreas = list()

    for SingleFile in FileParam['File Name']:
        PDC_Result = HFOF.LoadDataFromFile({'Folder Name':FileParam['Folder Name'],'File Name':SingleFile})
        if isinstance(PDC_Result['Results'][0],list) == False:
            PDC_Result['Results'] = [PDC_Result['Results']] # turn single fixed square to list

        for Main_Ind in range(len(PDC_Result['Results'])):
            for Sub_Ind in range(len(PDC_Result['Results'][Main_Ind])):
                if ((PDC_Result['Results'][Main_Ind][Sub_Ind][:,:,0] >= Columns[0]) & (PDC_Result['Results'][Main_Ind][Sub_Ind][:,:,0]<=Columns[1] ) & (PDC_Result['Results'][Main_Ind][Sub_Ind][:,:,1]>=Rows[0]) & (PDC_Result['Results'][Main_Ind][Sub_Ind][:,:,1]<=Rows[1])).any():
                    if PDC_Result['Results'][Main_Ind][Sub_Ind].shape[0] > Background:
                        SelectedAreas.append(PDC_Result['Results'][Main_Ind][Sub_Ind])
    return SelectedAreas

def Choose_from_Scan_Area_List(Area_List,Rows,Columns, Background = 0):
    """
    A function that receives a list of 3D numpy arrays and ranges of columns and rows.
    Each 3D numpy array is all the pairs (all the frames) in two small areas.
    It checks all the 3D numpy arrays to find 3D arrays with pixels in the range.
    It returns a list of filtered 3D numpy arrays
    Rows and Columns are NOT binned
    """

    All_Areas_of_Fixed = list()
    Total_Selected_per_Fixed = list()
    All_Selected_Areas = list()
    if isinstance(Area_List[0],list) == False:
        Area_List = [Area_List] # turn single fixed square to list


    for Main_Ind in range(len(Area_List)):
        SelectedAreas = list()
        for Sub_Ind in range(len(Area_List[Main_Ind])):
            if ((Area_List[Main_Ind][Sub_Ind][:,:,0] >= Columns[0]) & (Area_List[Main_Ind][Sub_Ind][:,:,0]<=Columns[1] ) & (Area_List[Main_Ind][Sub_Ind][:,:,1]>=Rows[0]) & (Area_List[Main_Ind][Sub_Ind][:,:,1]<=Rows[1])).any():

                for Sub_Ind_2 in range(len(Area_List[Main_Ind])):
                    if Area_List[Main_Ind][Sub_Ind_2].shape[0] > Background:
                        SelectedAreas.append(Area_List[Main_Ind][Sub_Ind_2])
                        All_Selected_Areas.append(Area_List[Main_Ind][Sub_Ind_2])
                        All_Areas_of_Fixed = Area_List[Main_Ind]
                        print('Fixed area index is: {}'.format(Main_Ind))
                Total_Selected_per_Fixed.append(SelectedAreas)
                break
                #return SelectedAreas, All_Areas_of_Fixed
    return All_Selected_Areas, All_Areas_of_Fixed, Total_Selected_per_Fixed


def Load_Scanned_Areas_from_List_of_Files(FileParam):
    Total_List_of_Areas = list()

    if isinstance(FileParam['File Name'],list):
        for SingleFile in FileParam['File Name']:
            PDC_Result = HFOF.LoadDataFromFile({'Folder Name':FileParam['Folder Name'],'File Name':SingleFile})
            if isinstance(PDC_Result['Results'][0],list) == False:
                PDC_Result['Results'] = [PDC_Result['Results']] # turn single fixed square to list
            Total_List_of_Areas.extend(PDC_Result['Results'])
    else:
        print('ERROR: given parameter should be dict with list of file names in key "File Name"')
    return Total_List_of_Areas

def Collect_Ring_Parts_from_List(Area_List, Threshold=3):
    List_of_Scannig_Area = list()
    for Main_Ind in range(len(Area_List)):
        for Sub_Ind in range(len(Area_List[Main_Ind])):
            if Area_List[Main_Ind][Sub_Ind].shape[0] > Threshold:
                List_of_Scannig_Area.append(Area_List[Main_Ind][Sub_Ind])
    return List_of_Scannig_Area

def Collect_Ring_Parts(FileParam, Background = 0):
    """
    A function the makes a long list of all the 3D numpy arrays of areas in which the number of pairs is above the threshold 'Background'
    FileParam is a dict with 'Folder Name' a string with folder name, and 'File Name' a list of file names of saved results of script 'Haim_pnCCD_Scan_Area.py'
    """

    RingParts = list()

    for SingleFile in FileParam['File Name']:
        PDC_Result = HFOF.LoadDataFromFile({'Folder Name':FileParam['Folder Name'],'File Name':SingleFile})
        if isinstance(PDC_Result['Results'][0],list) == False:
            PDC_Result['Results'] = [PDC_Result['Results']] # turn single fixed square to list

        for Main_Ind in range(len(PDC_Result['Results'])):
            for Sub_Ind in range(len(PDC_Result['Results'][Main_Ind])):
                if PDC_Result['Results'][Main_Ind][Sub_Ind].shape[0] > Background:
                    RingParts.append(PDC_Result['Results'][Main_Ind][Sub_Ind])

    return RingParts

def Calc_PDC_Statistics(PDC_Areas_List, Threshold = 3):
    """
    A function that receives a list of lists of 3D numpy arrays with x,y of pairs in each area and counts how many pairs above and below the threshold.
    The main list is of fixed areas, while the inner is of "scanning" areas
    """
    List_of_Fixed = list()
    Fixed_Area_Counts = numpy.zeros(len(PDC_Areas_List),dtype = numpy.uint32)
    Fixed_Area_Counts_Subtracted = numpy.zeros(Fixed_Area_Counts.shape[0], dtype = numpy.uint32)
    #Total_Count_Fixed
    #Total_Count_Scanning = # it is for checking
    if isinstance(PDC_Areas_List[0],list) == False:
        PDC_Areas_List=[PDC_Areas_List]
        print('The input to function Calc_PDC_Statistics was NOT list of lists. Converted it to list of lists')

    for Main_Ind in range(len(PDC_Areas_List)):
        List_of_Scannig_Area = list()
        for Sub_Ind in range(len(PDC_Areas_List[Main_Ind])):
            if PDC_Areas_List[Main_Ind][Sub_Ind].shape[0] > Threshold:
                List_of_Scannig_Area.append(PDC_Areas_List[Main_Ind][Sub_Ind]) # List of numpy 3D arrays
                Fixed_Area_Counts[Main_Ind] += PDC_Areas_List[Main_Ind][Sub_Ind].shape[0]
                Fixed_Area_Counts_Subtracted[Main_Ind] += PDC_Areas_List[Main_Ind][Sub_Ind].shape[0] - Threshold
        if len(List_of_Scannig_Area) != 0:
            List_of_Fixed.append(List_of_Scannig_Area)
    #ImageForSelectingRegion = Pairs_List_to_Image(Ring_Parts, VBin=VBin, HBin=HBin)

    StatisticsAboveThresh = list()
    StatisticsBelowThresh = list()

    Areas_List = Collect_Ring_Parts_from_List(PDC_Areas_List, Threshold=0)
    for Ind in range(len(Areas_List)):
        if Areas_List[Ind].shape[0] > Threshold:
            StatisticsAboveThresh.append(Areas_List[Ind].shape[0])
        else:
            StatisticsBelowThresh.append(Areas_List[Ind].shape[0])
    # should return total PDC pairs counts,
    # total counts after threshold subtraction,
    # total fixed areas above threshold
    # The total area should be calculated after converting data to 2D image - should count non zero area
    return numpy.array(StatisticsAboveThresh), numpy.array(StatisticsBelowThresh), List_of_Fixed, Fixed_Area_Counts, Fixed_Area_Counts_Subtracted

def Calc_Background(Pairs_List):
    """
    A function that creates numpy array with number of pairs in each 3D numpy array in the list Pairs_List.

    Parameters:
        Pairs_List - list of 3D numpy arrays of different regions. Number of pairs in each region is saved and return by the function.
    """
    SNR_Plot = numpy.zeros((len(Pairs_List)), dtype=numpy.uint32)

    for Ind in range(SNR_Plot.shape[0]):
        SNR_Plot[Ind] = Pairs_List[Ind].shape[0]

    return SNR_Plot

class MyLine:
    def __init__(self,x1,y1,x2,y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2

        self.m=(y2-y1)/(x2-x1)
        self.b = y2-x2*self.m
    def Calc_y_from_x(self,x):
        y = x*self.m+self.b
        return y

class LinesContainerObject():
    def __init__(self):
        self.Lines = list()
        self.Radii_Left = list()
        self.Radii_Right = list()

    def AddLine(self,LineObj):
        self.Lines.append(LineObj)

    def CalcCoefMat(self):
        NumberOfLines = len(self.Lines)
        NewMat = numpy.zeros((NumberOfLines,2),dtype=numpy.float64)
        NewMat[:,0] = 1
        for LineInd in range(NumberOfLines):
            NewMat[LineInd,1] = -self.Lines[LineInd].m
        return NewMat

    def Calc_b_Vector(self):
        NumberOfLines = len(self.Lines)
        b_vector = numpy.zeros(NumberOfLines,dtype=numpy.float64)
        for LineInd in range(NumberOfLines):
            b_vector[LineInd] = self.Lines[LineInd].b
        return b_vector

    def Intersection_Point(self):
        Solution_Numeric = numpy.linalg.lstsq(self.CalcCoefMat(), self.Calc_b_Vector(), None)
        return Solution_Numeric[0]

    def Radii_Calc_Horizontal_separation(self):
        """
        A function (method) that calculates all the radii of lines stored in the object
        The radii are the distances between the center of intersection of all the lines (stored in the object) and the two edges of each line.

        """
        NumberOfLines = len(self.Lines)
        Cent = self.Intersection_Point()
        Xcent = Cent[1]
        Ycent = Cent[0]
        RadiudRight = numpy.zeros(NumberOfLines,dtype=numpy.float64)
        RadiusLeft = numpy.zeros(NumberOfLines, dtype=numpy.float64)
        # I check left and right separately because they have different energies
        for LineInd in range(NumberOfLines):
            if self.Lines[LineInd].x1 > Xcent:
                RadiudRight[LineInd] = numpy.sqrt( (self.Lines[LineInd].x1-Xcent)**2 + (self.Lines[LineInd].y1-Ycent)**2)
                RadiusLeft[LineInd] = numpy.sqrt((Xcent - self.Lines[LineInd].x2) ** 2 + (self.Lines[LineInd].y2 - Ycent) ** 2)
            elif self.Lines[LineInd].x2 > Xcent:
                RadiudRight[LineInd] = numpy.sqrt( (self.Lines[LineInd].x2-Xcent)**2 + (self.Lines[LineInd].y2-Ycent)**2)
                RadiusLeft[LineInd] = numpy.sqrt((Xcent - self.Lines[LineInd].x1) ** 2 + (self.Lines[LineInd].y1 - Ycent) ** 2)

        self.Radii_Right = RadiudRight
        self.Radii_Left = RadiusLeft
        return numpy.average(RadiusLeft), numpy.average(RadiudRight), numpy.std(RadiusLeft),  numpy.std(RadiudRight), RadiudRight, RadiusLeft


    def Plot_Lines(self, Old_Ax=None):
        NumberOfLines = len(self.Lines)
        #Fig1 = HPF.plt.figure()
        #Ax1 = Fig1.add_subplot()
        if isinstance(Old_Ax,type(None)):
            Ax1 = HPF.Plot_Line_Between_Points(self.Lines[0].x1,self.Lines[0].y1, self.Lines[0].x2,self.Lines[0].y2,PlotSize=[0.2, 0.15, 0.7, 0.8])[1]
            for LineInd in range(1,NumberOfLines):
                Fig1, Ax1 = HPF.Plot_Line_Between_Points(self.Lines[LineInd].x1, self.Lines[LineInd].y1,
                                                         self.Lines[LineInd].x2, self.Lines[LineInd].y2, Ax1)
        else:
            for LineInd in range(NumberOfLines):
                Fig1, Ax1 = HPF.Plot_Line_Between_Points(self.Lines[LineInd].x1,self.Lines[LineInd].y1, self.Lines[LineInd].x2,self.Lines[LineInd].y2,Old_Axis=Old_Ax)

        return Fig1, Ax1

    def Plot_Circles(self, Old_Axes = None):
        # Draw circles with calculated radii
        RL,RR = self.Radii_Calc_Horizontal_separation()[:2]
        Cent = self.Intersection_Point()
        Xcent = Cent[1]
        Ycent = Cent[0]

        Fig1, Ax1 = HPF.Plor_Circle_from_Radius(X=Xcent,Y=Ycent,Radius = RL,Old_Axis=Old_Axes,Line_Width=10,Line_Color='r')
        Fig1, Ax1 = HPF.Plor_Circle_from_Radius(X=Xcent, Y=Ycent, Radius = RR, Old_Axis=Ax1,Line_Width=10,Line_Color='b')
        return Fig1, Ax1

# Lines:

def Calc_PDC_Coordinates(FileParam, LeftCentralEnergy, RighCentralEnergy, Background = 0, Center = 16, DoSave = False, SaveLocation = None, VBin=4, HBin=8):

    """
    A function that finds and saves x,y coordinates of correlated areas (areas above the threshold)
    It collects results of small square scans and calculates centers of intensity for squares pairs that are above the background\threshold
    The files that are loaded are the results of the Haim_pnCCD_Scan_Area script.
    The files contain dictionary of keys Results and Scan Info
    Args:
        FileParam:
        Background:
        Center:

    Returns:
        A dictionary with coordinates of squares centers above threshold and object of type LinesContainerObject with all the lines
    """
    LinesContainer = LinesContainerObject()
    CircleLinesData = dict()
    CircleLinesData['File Name list'] = []
    CircleLinesData['File Data list'] = []
    CircleLinesData['Data Lines list'] = []
    CircleLinesData['Scan time'] = []
    CircleLinesData['Left Energy'] = LeftCentralEnergy
    CircleLinesData['Right Energy'] = RighCentralEnergy

    for SingleFile in FileParam['File Name']:
        LinesList = []

        PDC_Result = HFOF.LoadDataFromFile({'Folder Name':FileParam['Folder Name'],'File Name':SingleFile})
        CircleLinesData['File Name list'].append(SingleFile)
        if 'Scan Info' in PDC_Result:
            CircleLinesData['File Data list'].append(PDC_Result['Scan Info'])
        else:
            CircleLinesData['File Data list'].append([])
        if 'Time' in PDC_Result:
            CircleLinesData['Scan time'].append(PDC_Result['Time'])
        else:
            CircleLinesData['Scan time'].append([])
        if isinstance(PDC_Result['Results'][0],list) == False:
            PDC_Result['Results'] = [PDC_Result['Results']] # turn single fixed square to list
        for Main_Ind in range(len(PDC_Result['Results'])): # Loop over fixed area
            DataDic = dict()  # check how is used
            DataDic = {'X1': None, 'Y1': None, 'X2': None, 'Y2': None}

            RingParts = [] # check how is used
            for Sub_Ind in range(len(PDC_Result['Results'][Main_Ind])): # Loop over scanning area
                if PDC_Result['Results'][Main_Ind][Sub_Ind].shape[0] > Background:
                    RingParts.append(PDC_Result['Results'][Main_Ind][Sub_Ind])
            if len(RingParts)>0:
                BinnedImage = Pairs_List_to_Image(RingParts, VBin=VBin, HBin=HBin)
                X1, Y1, X2, Y2 = Calc_Center_Mass(BinnedImage, Center=Center) # Suitable for images with two small correlation areas. Therefore calculate it for each fixed area
                DataDic['X1'] = X1
                DataDic['Y1'] = Y1
                DataDic['X2'] = X2
                DataDic['Y2'] = Y2
                LinesObj = MyLine(X1, Y1, X2, Y2)
                LinesContainer.AddLine(LinesObj)
            LinesList.append(DataDic)
        CircleLinesData['Data Lines list'].append(LinesList)

    if DoSave:
        Folder = SaveLocation['Folder Name']
        File = SaveLocation['Base Name'] + '_{}_{}.pkl'.format(LeftCentralEnergy, RighCentralEnergy)
        HSF.SaveDataToFile(CircleLinesData, {'Folder Name': Folder, 'File Name':File})
    return CircleLinesData, LinesContainer

def Plot_Energy_vs_Radii(Folder, FileList):
    """
    A function that loads x,y coordinates of correlated areas, and calculates average radius and radius STD for each file.
    The files are the result of function: Calc_PDC_Radii
    Each file should have specific energy range
    """
    EnergiesArray = numpy.zeros(len(FileList)*2) # Each file contains two energies (signal and idler)
    RadiiArray = numpy.zeros(len(FileList)*2) # Each file contains two energies (signal and idler)
    RadiiSTDArray = numpy.zeros(len(FileList) * 2)  # Each file contains two energies (signal and idler)
    for FileInd in range(len(FileList)): # Loop over files with different energies
        LinesContainer = LinesContainerObject()
        LinesCoordinates = HFOF.LoadDataFromFile({'Folder Name': Folder, 'File Name': FileList[FileInd]})
        for PrevDataFileInd in range(len(LinesCoordinates['File Name list'])):
            for Ind in range(len(LinesCoordinates['Data Lines list'][PrevDataFileInd])):
                if not isinstance(LinesCoordinates['Data Lines list'][PrevDataFileInd][Ind]['X1'], type(None)):
                    X1 = LinesCoordinates['Data Lines list'][PrevDataFileInd][Ind]['X1']
                    Y1 = LinesCoordinates['Data Lines list'][PrevDataFileInd][Ind]['Y1']
                    X2 = LinesCoordinates['Data Lines list'][PrevDataFileInd][Ind]['X2']
                    Y2 = LinesCoordinates['Data Lines list'][PrevDataFileInd][Ind]['Y2']
                    LinesObj = MyLine(X1, Y1, X2, Y2)
                    LinesContainer.AddLine(LinesObj)
        EnergiesArray[FileInd * 2] = LinesCoordinates['Left Energy']
        EnergiesArray[FileInd * 2 + 1] = LinesCoordinates['Right Energy']
        RadiiResult = LinesContainer.Radii_Calc_Horizontal_separation()[:4]
        RadiiArray[FileInd * 2], RadiiArray[FileInd * 2 + 1], RadiiSTDArray[FileInd * 2], RadiiSTDArray[FileInd * 2 + 1] = RadiiResult
    return EnergiesArray, RadiiArray, RadiiSTDArray

def Radii_Line_Fitting (Xi,Yi):
    # Solving: a * vector = b -> finding vector that solves equation. a is matrix with Xi, b is vector with Yi.
    # vector is m and n coefecents of a line: mx+n
    # Thoretically mX1+n=Y1, but there is noise, threfore fing m and n that fits best
    a = numpy.ones((Xi.shape[0],2))
    a[:,0] = Xi
    [m,n], LinError = numpy.linalg.lstsq(a,Yi,rcond=None)[:2]
    return m, n, LinError

class Data_Handler:
    '''
    Data analysis object.
    InputData - list of numpy arrays of frames.
    ImageShape - a dictionary with keys:  Horizontal Pixels, Vertical Pixels
    ScanParam - a dictionary with all the scan parameters for the filtering methods.
    '''
    def __init__(self,InputData = None, ImageShape = None, ScanParam = None):
        self.FrameList = list()
        try:
            HorPixSize = ImageShape['Horizontal Pixels']
            VerPixSize = ImageShape['Vertical Pixels']
        except:
            print('Default image size: 132 rows, 264 columns.')
            HorPixSize = 264
            VerPixSize = 132

        self.SingleFrame = numpy.ones((VerPixSize,HorPixSize))
        self.ScanRegions = numpy.zeros((VerPixSize,HorPixSize))
        self.BinParametersList = list()
        self.BinValuesList = list()

        if type(ScanParam) == dict:
            self.Scan_Parameters = ScanParam
            self.Parameters_For_Plotting = ScanParam
        else:
            self.Scan_Parameters = dict()
            self.Parameters_For_Plotting = dict()

        if InputData != None:
            if type(InputData) == list:
                if type(InputData[0]) == numpy.ndarray:
                    self.FrameList = InputData
                else:
                    print('Given incorrect data. Should be list of numpy arrays.')
            else:
                print('Given incorrect data. Should be list of numpy arrays.')
    def Scan_With_Regions(self):

        HStartVal = self.Scan_Parameters['Right Horizontal Scan Start']
        HEndVal = self.Scan_Parameters['Right Horizontal Scan End']
        HStep = self.Scan_Parameters['Right Horizontal Scan Step']
        VStartVal = self.Scan_Parameters['Right Vertical Scan Start']
        VEndVal = self.Scan_Parameters['Right Vertical Scan End']
        VStep = self.Scan_Parameters['Right Vertical Scan Step']

        HorizVector = range(HStartVal,HEndVal,HStep)
        VerticVector = range(VStartVal,VEndVal,VStep)

        #ScanResultsList = Parallel(n_jobs=16)(delayed(PairsScanStep)(FrameList,FilterParameters,HorizVector[Hind],HorizVector[Hind+1],VerticVector[Vind],VerticVector[Vind+1]) for Hind in range(HLen-1) for Vind in range(VLen-1))
        #ScanResultsList = Parallel(n_jobs=16)(delayed(PairsScanStep)(self.FrameList,self.Scan_Parameters,XVal,XVal+HStep-1,YVal,YVal+VStep-1) for XVal in HorizVector for YVal in VerticVector)

        # Last version:
        #ScanResultsList = Parallel(n_jobs=4,backend='multiprocessing')(delayed(PairsScanStep)(self.FrameList, self.Scan_Parameters, XVal, XVal + HStep - 1, YVal, YVal + VStep - 1)for XVal in HorizVector for YVal in VerticVector)
        ScanResultsList = list()
        StartTime = time.time()
        for XVal in HorizVector:
            #I defined this function for the map method of the multiprocessing
            #def FuncPrepForMultiproc(IterParam):
            #    NewF = PairsScanStep(self.FrameList, self.Scan_Parameters, XVal, XVal + HStep - 1, IterParam, IterParam + VStep - 1)
            #   return NewF
            for YVal in VerticVector:
                # ScanResultsList is a list of results list of each step. Each step is list with: parameters dict, number of pairs, list of frames with pixels
                ScanResultsList.append(PairsScanStep(self.FrameList,self.Scan_Parameters,XVal, XVal + HStep - 1, YVal, YVal + VStep - 1))
            #Test to work with Multiprocessing. There are problems: Cant pickle local object
            #with Pool(8) as ParLoop:
            #    ParLoop.map(FuncPrepForMultiproc,VerticVector)
            print('Finished vertical scan after {} seconds'.format(time.time()-StartTime))
            StartTime = time.time()
        '''
        # Trying to work with multiproccessor - didn't finish:
        ScanResultsList = list()
        for XVal in HorizVector:
            with Pool(5) as MultiPro:
                TempIter=itertools.repeat((self.FrameList,self.Scan_Parameters,XVal,XVal + HStep - 1),len(VerticVector))
                TempParam = zip(PairsScanStep(,, , ,YVal,YVal+VStep-1))
                StepRes = list(MultiPro.starmap( , TempIter))
            ScanResultsList.extend(StepRes)
        '''
        # The ScanResultsList consists of a list the length of scan steps and contains the result of each step
        return ScanResultsList

    def Pick_Regions(self,ScanResults):
        ScanStepPairs = list()
        ParametersList = list()
        PairsNum = numpy.zeros((2, len(ScanResults)))
        for Ind, StepResults in enumerate(ScanResults):
            # ScanedFrame[BinResults[2]:BinResults[3], BinResults[0]:BinResults[1]] = BinResults[4]
            PairsNum[0, Ind] = StepResults[1]
            PairsNum[1, Ind] = Ind

        #PairsNumStd = numpy.std(PairsNum[0, :])
        #PDC_Regions = (PairsNum[0, :] - PairsNum[0, :].mean()) > PairsNumStd

        ShotNoise = numpy.sqrt(PairsNum[0, :])
        #ShotNoise = numpy.std(PairsNum[0, :])
        Background = PairsNum[0, :].mean()
        PDC_Regions = (PairsNum[0, :] - Background) > ShotNoise
        '''
        # Give one bin above noise
        if numpy.count_nonzero(PDC_Regions) > 0:
            MaxInd = PairsNum[0,:].argmax()
            ScanStepPairs.extend(ScanResults[MaxInd][2])
            ParametersList.append(ScanResults[MaxInd][0])
        '''
        # Gives list of bins above noise
        for Ind in PairsNum[1, PDC_Regions].astype(numpy.uint16):
            ScanStepPairs.extend(ScanResults[Ind][2])

        for Ind in list(numpy.flatnonzero(PDC_Regions)):
            # List of dictionaries of parameters of PDC regions
            ParametersList.append(ScanResults[Ind][0])

        return ScanStepPairs, ParametersList

    def Check_Scan_Bounderies(self):
        '''
        Return frame ready for imaging of bounderies of scanning regions
        '''
        try:
            HStartVal = self.Parameters_For_Plotting['Right Horizontal Scan Start']
            HEndVal = self.Parameters_For_Plotting['Right Horizontal Scan End']
            HStep = self.Parameters_For_Plotting['Right Horizontal Scan Step']
            VStartVal = self.Parameters_For_Plotting['Right Vertical Scan Start']
            VEndVal = self.Parameters_For_Plotting['Right Vertical Scan End']
            VStep = self.Parameters_For_Plotting['Right Vertical Scan Step']

            HorizVector = range(HStartVal, HEndVal, HStep)
            VerticVector = range(VStartVal, VEndVal, VStep)

            Scan_Regions_Frame = numpy.zeros(self.SingleFrame.shape)
            #Scan_Regions_Frame = self.ScanRegions
            #print('Corners of scan regions:')
            for XVal in HorizVector:
                for YVal in VerticVector:
                    Scan_Regions_Frame[YVal:(YVal+VStep), XVal:(XVal+HStep)] += 1
                    #print('{},{} - {},{}'.format(XVal,YVal,XVal+HStep-1,YVal+VStep-1))
            HStartVal = self.Parameters_For_Plotting['Smallest Left Pixel']
            HEndVal = self.Parameters_For_Plotting['Largest Left Pixel']
            VStartVal = self.Parameters_For_Plotting['Left Bottom Pixel']
            VEndVal = self.Parameters_For_Plotting['Left Top Pixel']
            Scan_Regions_Frame[VStartVal:VEndVal+1,HStartVal:HEndVal+1] += 1

            return Scan_Regions_Frame
        except:
            print('Error in method Check_Scan_Bounderies - something is wrong with scan parameters. Check the dictionary that was given to the method!')
            return

    def Prepare_For_Plotting(self,FrameList,ImageShape = None):
        try:
            HorPixSize = ImageShape['Horizontal Pixels']
            VerPixSize = ImageShape['Vertical Pixels']
        except:
            print('Default image size: 132 rows, 264 columns.')
            HorPixSize = 264
            VerPixSize = 132

        if type(FrameList) == list:
            if type(FrameList[0]) == numpy.ndarray:
                try:
                    PixelsArray = numpy.vstack(FrameList)
                    # uint8 is not enough
                    #PixelsSparse = csr_array((numpy.ones(PixelsArray.shape[0], dtype=numpy.uint8), (PixelsArray[:, 1], PixelsArray[:, 0])),shape=(VerPixSize, HorPixSize))
                    PixelsSparse = csr_array((numpy.ones(PixelsArray.shape[0], dtype=numpy.uint16), (PixelsArray[:, 1], PixelsArray[:, 0])),shape=(VerPixSize, HorPixSize))
                    self.SingleFrame = PixelsSparse.toarray()
                except:
                    print('Error in method Prepare_For_Plotting. Check the attribute FrameList. Should be list of numpy arrays.')
            else:
                print('Error in method Prepare_For_Plotting. The data structure is incorrect. Check attribute FrameList. Should be list of numpy arrays.')
        else:
            print('Error in method Prepare_For_Plotting. The data structure is incorrect. Check attribute FrameList. Should be list of numpy arrays.')
    def Prepare_For_Plotting_Array(self,FrameList,ImageShape = None):
        try:
            HorPixSize = ImageShape['Horizontal Pixels']
            VerPixSize = ImageShape['Vertical Pixels']
        except:
            print('Default image size: 132 rows, 264 columns.')
            HorPixSize = 264
            VerPixSize = 132

        try:
            PixelsArray = numpy.vstack(FrameList)
            # uint8 is not enough
            #PixelsSparse = csr_array((numpy.ones(PixelsArray.shape[0], dtype=numpy.uint8), (PixelsArray[:, 1], PixelsArray[:, 0])),shape=(VerPixSize, HorPixSize))
            PixelsSparse = csr_array((numpy.ones(PixelsArray.shape[0], dtype=numpy.uint16), (PixelsArray[:, 1], PixelsArray[:, 0])),shape=(VerPixSize, HorPixSize))
            self.SingleFrame = PixelsSparse.toarray()
        except:
            print('Error in method Prepare_For_Plotting. Check the attribute FrameList. Should be list of numpy arrays.')

    def Combine_Scan_Regions_For_Plotting(self):
        self.ScanRegions =numpy.zeros(self.SingleFrame.shape)

        for CurrentParameters in self.BinParametersList:
            self.Parameters_For_Plotting = CurrentParameters
            self.ScanRegions += self.Check_Scan_Bounderies()

    def Bins_OverLap_Image(self):
        # Show how many times each region was scanned. The region on the left is sacnned more.
        # For the PDC result, it shows if region on the right was selected more than one time
        try:
            self.SingleFrame = numpy.zeros(self.SingleFrame.shape)
            for CurrentParameters in self.BinParametersList:

                HStartVal = CurrentParameters['Smallest Right Pixel']
                HEndVal = CurrentParameters['Largest Right Pixel']
                VStartVal = CurrentParameters['Right Bottom Pixel']
                VEndVal = CurrentParameters['Right Top Pixel']
                self.SingleFrame[VStartVal:VEndVal+1,HStartVal:HEndVal+1] += 1

                HStartVal = CurrentParameters['Smallest Left Pixel']
                HEndVal = CurrentParameters['Largest Left Pixel']
                VStartVal = CurrentParameters['Left Bottom Pixel']
                VEndVal = CurrentParameters['Left Top Pixel']
                self.SingleFrame[VStartVal:VEndVal+1,HStartVal:HEndVal+1] += 1

        except:
            print('Error in methos Bins_OverLap_Image')

    def Bins_Image(self):
        # Shows the total counts in the regions (binned image)
        try:
            self.SingleFrame = numpy.zeros(self.SingleFrame.shape)
            for CurrentValue, CurrentParameters in zip(self.BinValuesList, self.BinParametersList):
                HStartVal = CurrentParameters['Smallest Right Pixel']
                HEndVal = CurrentParameters['Largest Right Pixel']
                VStartVal = CurrentParameters['Right Bottom Pixel']
                VEndVal = CurrentParameters['Right Top Pixel']
                self.SingleFrame[VStartVal:VEndVal + 1, HStartVal:HEndVal + 1] += CurrentValue

                HStartVal = CurrentParameters['Smallest Left Pixel']
                HEndVal = CurrentParameters['Largest Left Pixel']
                VStartVal = CurrentParameters['Left Bottom Pixel']
                VEndVal = CurrentParameters['Left Top Pixel']
                self.SingleFrame[VStartVal:VEndVal + 1, HStartVal:HEndVal + 1] += CurrentValue

        except:
            print('Error in methos Bins_Image')

    def Values_Statistics(self):
        Values = numpy.array(self.BinValuesList)
        Statistics = dict()
        Statistics['Mean'] = Values.mean()
        Statistics['Std'] = Values.std()
        Statistics['Sqrt'] = numpy.sqrt(Values)
        Statistics['Selected Values Indexes Method 1'] = numpy.flatnonzero(( Values - Statistics['Mean'] ) > Statistics['Std'])
        Statistics['Selected Values Method 1'] = Values[Statistics['Selected Values Indexes Method 1']]
        Statistics['Selected Values Indexes Method 2'] = numpy.flatnonzero( (Values - Statistics['Mean']) > Statistics['Sqrt'] )
        Statistics['Selected Values Method 2'] = Values[Statistics['Selected Values Indexes Method 2']]
        Statistics['Selected Values Indexes Method 3'] = numpy.argmax(Values * (( Values - Statistics['Mean'] ) > Statistics['Std'] ))
        Statistics['Selected Values Method 3'] = Values[Statistics['Selected Values Indexes Method 3']]

        return Statistics