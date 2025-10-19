import pickle
import os
import os.path as OP
import scipy.io
import mmap as FModules
import numpy
import time
import io

def LastFileLines(FileNamesDic):
    ReadFName=FileNamesDic['FileNameToRead']
    WriteFName=FileNamesDic['FileNameToWrite']
    LinesNum=FileNamesDic['NumberOfLines']

    with open(file=ReadFName, mode="r") as OpenFileObj:
        with FModules.mmap(OpenFileObj.fileno(), length=0, access=FModules.ACCESS_READ, offset=0) as OpenFileMap:
            EndLineInd=OpenFileMap.rfind(b'\n')
            for TempIndex in range(LinesNum):
                print(TempIndex)
                EndLineInd=OpenFileMap.rfind(b'\n',0,EndLineInd)
                if EndLineInd==-1:
                    print('There are {} lines in the file:\n{}\nThe size of the file in bytes is:{}'.format(TempIndex+1,ReadFName,OpenFileMap.size()))
                    break
            if EndLineInd == -1:
                OpenFileMap.seek(0)
            else:
                OpenFileMap.seek(EndLineInd)
            LastLines=OpenFileMap.read().decode('utf-8')
    return LastLines

def LoadIndsFromMatFile(InputParameters):
    '''
    Give full path to *.mat file with indexes of begining of frames
    :return:
    Numpy array of indexes
    '''
    FileName = InputParameters['Folder'] + InputParameters['File Name']
    IndexesFromMAT=scipy.io.loadmat(FileName)
    return IndexesFromMAT['MyPythonIndexes'][0]

def OpenFrames(IndexData, Parameters):
    FirstFrame=Parameters['FirstFrame']
    NumberOfFrames=Parameters['NumberOfFrames']
    FileName=Parameters['Folder'] + Parameters['File Name']
    EmptyFrames = list()
    FrameList=list()
    TotalNumOfFrames=IndexData.__len__()

    with open(file=FileName, mode="r") as RawDataFile:
        with FModules.mmap(RawDataFile.fileno(), length=0, access=FModules.ACCESS_READ, offset=0) as RawDataFileMap:
            if FirstFrame>TotalNumOfFrames:
                print('First frame is larger than the length of the raw data.')
            else:
                if (FirstFrame-1+NumberOfFrames)>TotalNumOfFrames:
                    NumberOfFrames=TotalNumOfFrames-FirstFrame+1
                StartTime=time.time()
                for FrameInd,FramePos in enumerate(IndexData[FirstFrame-1:FirstFrame-1+NumberOfFrames],start=FirstFrame-1):
                    if time.time() > StartTime + 10:
                        print('Loaded {:.1f}% frames'.format((FrameInd-FirstFrame)/NumberOfFrames*100))
                        StartTime = time.time()
                    #print('Frame index is: {}'.format(FrameInd))
                    #print('Frame position in the file is: {}'.format(FramePos))
                    if FrameInd==TotalNumOfFrames-1:
                        # !!!!!!! - Crush in the case of reading number of frames from a file. Last index may not be the last frame in a file- !!!!!!
                        print('Reached last frame in the file.')
                        print('The FramePos is: {} and size of fie is: {}'.format(FramePos,RawDataFileMap.size()))
                        FrameTextData = io.StringIO(RawDataFileMap[FramePos:-1].decode('utf-8'))
                        LoadedData = numpy.loadtxt(FrameTextData, dtype=numpy.int16, usecols=(1, 2, 3), skiprows=1)
                        if LoadedData.shape[0] == 0:
                            EmptyFrames.append(FrameInd)
                        else:
                            FrameList.append(LoadedData)
                    else:
                        FrameTextData= io.StringIO(RawDataFileMap[FramePos:IndexData[FrameInd+1]].decode('utf-8'))
                        LoadedData = numpy.loadtxt(FrameTextData, dtype=numpy.int16, usecols=(1, 2, 3), skiprows=1)
                        if LoadedData.shape[0] == 0:
                            EmptyFrames.append(FrameInd)
                        else:
                            FrameList.append(LoadedData)
                print('Loaded {} frames.'.format(len(FrameList)))
                if len(EmptyFrames) > 0:
                    print('There were {} empty frames. Indexes of 4 frames: {}, {},... {},{}'.format(len(EmptyFrames),EmptyFrames[0],EmptyFrames[1],EmptyFrames[-2],EmptyFrames[-1]))

    return FrameList

def OpenNumberOfFrames(Parameters):
    NumberOfFrames=Parameters['NumberOfFrames']
    FileName = Parameters['Folder Name'] + Parameters['File Name']
    FrameList=list()
    EmptyFrames = list()

    with open(file=FileName, mode="r") as RawDataFile:
        with FModules.mmap(RawDataFile.fileno(), length=0, access=FModules.ACCESS_READ, offset=0) as RawDataFileMap:
            FrameStartInd = RawDataFileMap.find(b'SF',0)
            FrameStopInd = RawDataFileMap.find(b'SF',FrameStartInd+2)
            if FrameStartInd>0:
                StartTimer=time.time()
                for ForIndex in range(NumberOfFrames):
                    if FrameStopInd>0:
                        FrameTextData = io.StringIO(RawDataFileMap[FrameStartInd:FrameStopInd].decode('utf-8'))
                        LoadedData = numpy.loadtxt(FrameTextData, dtype=numpy.int16, usecols=(1, 2, 3), skiprows=1)
                        if LoadedData.shape[0] == 0:
                            EmptyFrames.append(ForIndex)
                        else:
                            FrameList.append(LoadedData)
                        if time.time()>StartTimer+10:
                            print('Loaded {:.1f}% frames'.format(ForIndex/NumberOfFrames*100))
                            StartTimer=time.time()
                    else:
                        FrameTextData = io.StringIO(RawDataFileMap[FrameStartInd:].decode('utf-8'))
                        LoadedData = numpy.loadtxt(FrameTextData, dtype=numpy.int16, usecols=(1, 2, 3), skiprows=1)
                        if LoadedData.shape[0] == 0:
                            EmptyFrames.append(ForIndex)
                        else:
                            FrameList.append(LoadedData)
                        print('Reached last frame in file: {}'.format(FileName))
                        break

                    FrameStartInd=FrameStopInd
                    FrameStopInd=RawDataFileMap.find(b'SF',FrameStartInd+2)
                print('Loaded {} frames.'.format(len(FrameList)))
                if len(EmptyFrames) > 0:
                    print('There were {} empty frames. Indexes of 4 frames: {}, {},... {},{}'.format(len(EmptyFrames),EmptyFrames[0],EmptyFrames[1],EmptyFrames[-2],EmptyFrames[-1]))

            else:
                print('Did not find frames in the file: {}'.format(FileName))

    return FrameList

def OpenTextBetweenIndexes(IndexData, Parameters):
    """
    A function that reads a text from a large file.
    The text is between given frame indexes
    """
    NumberOfFrames=Parameters['NumberOfFrames']
    FileName = Parameters['Folder Name'] + Parameters['File Name']
    FirstFrame = Parameters['FirstFrame'] # Index of frame in IndexData
    TextData = ''

    TotalNumOfFrames=IndexData.__len__()

    with open(file=FileName, mode="r") as RawDataFile:
        with FModules.mmap(RawDataFile.fileno(), length=0, access=FModules.ACCESS_READ, offset=0) as RawDataFileMap:
            if FirstFrame>TotalNumOfFrames:
                print('First frame is larger than the length of the raw data.')
            else:
                if (FirstFrame-1+NumberOfFrames)>TotalNumOfFrames:
                    # If the requested number of frames is larger than the total number of frames in the file, fix the number of frames to a correct number according to the real number of frames in the file
                    print('Reading till the end of the file')
                    NumberOfFrames=TotalNumOfFrames-FirstFrame+1
                #print('Frame index is: {}'.format(FrameInd))
                #print('Frame position in the file is: {}'.format(FramePos))
                if FirstFrame==TotalNumOfFrames:
                    # !!!!!!! - I think it will crush in the case of reading exact number of frames till the end!!!!!!
                    print('The first requested frame is the last frame in the file.')
                    print('The position in the file is: {} and size of fie is: {}'.format(IndexData[FirstFrame-1],RawDataFileMap.size()))
                    TextData = RawDataFileMap[IndexData[FirstFrame-1]:-1].decode('utf-8')
                else:
                    TextData= RawDataFileMap[IndexData[FirstFrame-1]:IndexData[FirstFrame-1+NumberOfFrames]].decode('utf-8')

    return TextData

def LoadDataFromFile(Param):
    """
    Load all the data from a pickel file.

    Argument:
    Param -- a dict with keys: Folder Name, File Name

    Return:
    The loaded data.
    """
    FileName=Param['Folder Name'] + Param['File Name']
    with open(FileName,'rb') as FileObj:
        Data=pickle.load(FileObj)
    return Data

def DataListsCombining(MyData1, ListOfFiles, FileName):
    '''
    MyData1 - the list to which to add the loaded lists.
    ListOfFiles - list of strings. Names of files inside the folder that is given in FileName dict.
    FileName - dict with folder name in key 'Folder Name' and file name in key 'File Name'.
    Combining the loaded lists from different files to one large list.
    This function receives the list and extends the given list MyData1.
    The function works directly on the list and doesn't return a value.
    '''
    try:
        if type(ListOfFiles) == list:
            for FName in ListOfFiles:
                FileName['File Name'] = FName
                DataFromFile = LoadDataFromFile(FileName)
                ListOfFarmesToInt16(DataFromFile)
                MyData1.extend(DataFromFile)
        else:
            FileName['File Name'] = ListOfFiles
            DataFromFile = LoadDataFromFile(FileName)
            ListOfFarmesToInt16(DataFromFile)
            MyData1.extend(DataFromFile)
    except:
        print('Error in DataListsCombining!!!\nSomething is wrong with loading the frames list (list of numpy arrays).')
        print('Check: folder name, file names and that the file contain correct data (python pickel file).')
def ListOfFarmesToInt16(ListOfFrames):
    '''
    Receives a list of numpy arrays and converts all the arrays in the list to uint16.
    The function works directly on the list and doesn't return a value.
    '''
    for Ind, SingFrame in enumerate(ListOfFrames):
        if SingFrame.dtype != numpy.uint16:
            ListOfFrames[Ind] = SingFrame.astype(numpy.uint16)

def ListFilesInFolder(FolderName):
    try:
        FilesList = list()
        for FolderContent in os.listdir(FolderName):
            if OP.isfile(FolderName + FolderContent):
                FilesList.append(FolderContent)
    except:
        print('Error in method ListFilesInFolder: could not get list of files in folder - please check the path.')
        return
    return FilesList

def Advacam_Prepare_Raw_ASCII_List(FileList):
    """
    A function that take a list of file names, sorts them and deletes file names that are without t3pa extantion
    """
    FileList.sort()
    # Didn't work well (skeeps files since remove method makes list of files shorter)
    FileNames = FileList.copy()
    for FileName in FileNames:
        if (FileName.rsplit('.',maxsplit=1)[1]!='t3pa') & (FileName.rsplit('.',maxsplit=1)[1]!='t3p') | (FileName[0]=='.'):
            FileList.remove(FileName)

class DataLoader:
    def __init__(self, DataLocation = None):
        self.BinsParametersList = list()
        self.BinsValuesList = list()
        self.NewData = list()
        self.NewDataSingleFile = list()

        if DataLocation == None:
            self.SourceFile = dict()
        else:
            self.SourceFile = DataLocation

    def LoadNewData_FromFolder(self, DataLocation = None):

        try:
            # Get from user directory name:
            if DataLocation != None:
                self.SourceFile = DataLocation
            # Check that there is folder name in given dictionary:
            if self.SourceFile.keys().__contains__('Folder Name') == False:
                print('****** No folder name for loading files was provided!')
                return
        except:
            print('Error in LoadNewData_FromFolder method: something wrong with the attribute SourceFile')
            return

        # Get list of files in the given folder:
        ListOfFiles = self.ListFilesInFolder()
        if ListOfFiles == None:
            return
        self.NewData = list()
        # Colect all the lists of frames from all the files.
        # Assume the file contains a list and the frame list is the first object of the list
        for FName in ListOfFiles:
            self.SourceFile['File Name'] = FName
            DataFromFile = self.LoadNewData_FromFile()
            # ListOfFarmesToInt16(DataFromFile)
            if type(DataFromFile) == list and len(DataFromFile) > 0:
                self.NewData.extend(DataFromFile)

    def ListFilesInFolder(self):
        try:
            FilesList = list()
            FolderName = self.SourceFile['Folder Name']
            for FolderContent in os.listdir(FolderName):
                if OP.isfile(FolderName + FolderContent):
                    FilesList.append(FolderContent)
        except:
            print('Error in method ListFilesInFolder: could not get list of files in folder - please check the path.')
            return
        return FilesList

    def LoadNewData_FromFile(self):
        try:
            FileName = self.SourceFile['Folder Name'] + self.SourceFile['File Name']
            print('Loading list of frames from: {}'.format(FileName))
            with open(FileName, 'rb') as FileObj:
                Data = pickle.load(FileObj)
        except:
            print('Error in method LoadNewData_FromFile - while opening file. Check attribute SourceFile')
            return
        if type(Data) != list:
            print('Problem - The content of the file doesn\'t have the expected structure!')
            return
        if type(Data[0]) == numpy.ndarray:
            ListOfFarmesToInt16(Data)
            self.NewDataSingleFile = Data
            return Data
        if (type(Data[0]) == list) and len(Data[0]) > 0:
            if type(Data[0][0]) == numpy.ndarray:
                ListOfFarmesToInt16(Data[0])
                self.NewDataSingleFile = Data[0]
                return Data[0]
            else:
                print('Problem - The content of the file doesn\'t have the expected structure!')
                return
        else:
            print('Empty file or wrong data structure in: {}'.format(FileName))

    def LoadBinsFromFile(self):
        self.BinsParametersList = list()
        self.BinsValuesList = list()
        DataFromFile = self.LoadEverythingFromFile()

        if (type(DataFromFile) == list) and len(DataFromFile[0]) > 0:
            if type(DataFromFile[0][0]) == dict:
                # This is file of bins
                for Ind in range(len(DataFromFile)):
                    self.BinsParametersList.append(DataFromFile[Ind][0])
                    self.BinsValuesList.append(DataFromFile[Ind][1])
            elif type(DataFromFile[1][0]) == dict:
                # This is file of selected PDC regions
                self.BinsParametersList.extend(DataFromFile[1])
            else:
                print('Wrong data structure in file: {}'.format(self.SourceFile))
        else:
            print('Wrong data structure in file: {}'.format(self.SourceFile))

    def LoadEverythingFromFile(self):
        try:
            FileName = self.SourceFile['Folder Name'] + self.SourceFile['File Name']
            with open(FileName, 'rb') as FileObj:
                Data = pickle.load(FileObj)
        except:
            print('Error in method LoadEverythingFromFile - while opening file. Check attribute SourceFile')
            return

        return Data

    def AddtoData_FromFile(self):
        pass

    def AddtoData_FromFolder(self):
        pass

    def Load_Binnig_Data_FromFolder(self, FolderName = None):
        try:
            if FolderName != None:
                self.SourceFile['Folder Name'] = FolderName['Folder Name']
            print('Loading bins parameters from all file in folder:\n{}'.format(self.SourceFile['Folder Name']))

            # Get list of files in the given folder:
            ListOfFiles = self.ListFilesInFolder()
            if ListOfFiles == None:
                return

            self.BinsParametersList = list()
            self.BinsValuesList = list()

            # Colect all the parameters dictionaries from all the files.
            # Assume the file contains a dictionary in some list:
            for FName in ListOfFiles:
                self.SourceFile['File Name'] = FName
                DataFromFile = self.LoadEverythingFromFile()

                if (type(DataFromFile) == list) and len(DataFromFile[0]) > 0:
                    if type(DataFromFile[0][0]) == dict:
                        # This is file of bins
                        for Ind in range(len(DataFromFile)):
                            self.BinsParametersList.append(DataFromFile[Ind][0])
                            self.BinsValuesList.append(DataFromFile[Ind][1])
                    elif type(DataFromFile[1][0]) == dict:
                        # This is file of selected PDC regions
                        self.BinsParametersList.extend(DataFromFile[1])
                    else:
                        print('Wrong data structure in file: {}'.format(self.SourceFile))
                else:
                    print('Empty file or wrong data structure in file: {}'.format(self.SourceFile))

        except:
            print('Error in method Load_Binnig_Parameters_FromFolder - check SourceFile attribute and content of folder')