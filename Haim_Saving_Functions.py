import numpy
import mmap as FModules
import pickle
import os.path as OP



def SaveNumberOfFrames(Parameters):
    NumberOfFrames=Parameters['NumberOfFrames']
    RawFileName=Parameters['FileName']
    OutFileName=Parameters['Output File Name']

    with open(file=RawFileName, mode="r") as RawDataFile:
        with FModules.mmap(RawDataFile.fileno(), length=0, access=FModules.ACCESS_READ, offset=0) as RawDataFileMap:
            StopInd = RawDataFileMap.find(b'SF',0)

            if StopInd>0:
                for ForIndex in range(NumberOfFrames):
                    StopInd=RawDataFileMap.find(b'SF',StopInd+2)
                    if StopInd==-1:
                        print('You asked for {} frames.'.format(NumberOfFrames))
                        print('Found only {} frames in the file.'.format(ForIndex+1))
                        StopInd=RawDataFileMap.size()
                        break

                with open(file=OutFileName + '_first_{}_frames.txt'.format(ForIndex + 1), mode='w') as OutputFile:
                    OutputFile.write(RawDataFileMap[0:StopInd].decode('utf-8'))

            elif RawDataFileMap.size()>1000:
                print('Did not find any frames in the file.')
                print('Saving first 1000 bytes.')
                with open(file=OutFileName + '_first_1000_bytes.txt', mode='w') as OutputFile:
                    OutputFile.write(RawDataFileMap[0:1000].decode('utf-8'))
            else:
                print('Did not find any frames in the file.')
                print('Saving the entire file.\nThe file size is: {} bytes'.format(RawDataFileMap.size()))
                with open(file=OutFileName + '_{}_bytes.txt'.format(RawDataFileMap.size()), mode='w') as OutputFile:
                    OutputFile.write(RawDataFileMap[0:].decode('utf-8'))

def SaveDataToFile(Data, Param, Override=False):
    FileName=Param['Folder Name'] + Param['File Name']
    if (OP.exists(FileName)) & (Override==False):
        print('\nThe file already exists!!!')
        UserAppruval = input('Do you want to replace it??? Press y if yes')
        if (UserAppruval == 'y'):
            with open(FileName, 'wb') as FileObj:
                pickle.dump(Data, FileObj, protocol=5)
    else:
        with open(FileName,'wb') as FileObj:
            pickle.dump(Data,FileObj,protocol=5)

class DataSaver:
    def __init__(self):
        self.SourceFile = dict()
        self.DestFile = dict()
        self.SufixString = ''
        self.FilterParam = dict()
        self.Data = list()

    def SaveMyData(self, NewDestFile = None, AskUser = True):
        if NewDestFile != None:
            try:
                self.DestFile['Folder Name'] = NewDestFile['Folder Name']
                self.DestFile['File Name'] = NewDestFile['File Name']
            except:
                print('Error in SaveMyData: Folder or file name that were given are incorrect!')
                return
        else:
            try:
                # ****** Should correct add checking that the folder exists
                if self.DestFile['Folder Name'] == '':
                    print('No folder name. Add dictionary to atrtibute DestFile')
                    return
            except:
                print('Error in SaveMyData: No information about where to save the data. Please give values to attribute DestFile.')
                return
            # Split file extention:
            New_File_Name = self.DestFile['File Name']
            New_File_Name = New_File_Name.split('.',1)[0] + self.SufixString + '.pkl'

            self.DestFile['File Name'] = New_File_Name

        try:
            Things_To_Dave = list()
            Things_To_Dave.append(self.Data)
            Things_To_Dave.append(self.FilterParam)
        except:
            print('Error in SaveMyData: check the Data and FilterParam attributes of the object.')
            return

        # print to user
        PrintString = self.DestFile['Folder Name'] + self.DestFile['File Name']
        print('Going to save to:\n{}'.format(PrintString))
        if AskUser:
            UserAppruval = input('Agree??? Press y if yes')
            if UserAppruval != 'y':
                return
        # check that exist and abort if doesn't
        if OP.exists(self.DestFile['Folder Name'] + self.DestFile['File Name']):
            print('The file already exists!!!')
        else:
            SaveDataToFile(Things_To_Dave, self.DestFile)
    def Save_Everything(self, Data_To_Save):
        try:
            # print to user
            PrintString = self.DestFile['Folder Name'] + self.DestFile['File Name']
            print('Going to save to:\n{}'.format(PrintString))

            if OP.exists(self.DestFile['Folder Name'] + self.DestFile['File Name']):
                print('The file already exists!!!')
            else:
                SaveDataToFile(Data_To_Save, self.DestFile)

        except:
            print('Error in method Save_Everything - check destination and data')