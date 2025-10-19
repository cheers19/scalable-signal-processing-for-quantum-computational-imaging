import matplotlib
matplotlib.use('Qt5Agg') # To install this backend: python -m pip install PyQt5
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as Patch
import SimulationPlot

def My_Colorbar(Axis, Tick_Label_Size=20):
    ImgR1_Norm = Axis.images[0].norm
    ImgR1_CM = Axis.images[0].get_cmap()
    SM_Obj = matplotlib.cm.ScalarMappable(cmap=ImgR1_CM, norm=ImgR1_Norm)
    CB_1 = Axis.figure.colorbar(SM_Obj, ax=Axis,location='right',orientation='vertical')
    Axis.figure.axes[-1].tick_params(labelsize=Tick_Label_Size)

def Custom_Plot_1(Data1, Data2, Data3):
    SubPltFig = plt.figure()
    # Grid_SubPltFig = SubPltFig.add_gridspec(3,4)

    ImgR1 = SubPltFig.add_subplot(3, 4, 1)
    Grid_SubPltFig = ImgR1.get_subplotspec()
    Grid_SubPltFig = Grid_SubPltFig.get_gridspec()
    Grid_SubPltFig.update(top=0.95, bottom=0.05, left=0, right=1, wspace=0.1, hspace=0.1)

    ImgR2 = SubPltFig.add_subplot(3, 4, 5)
    ImgR3 = SubPltFig.add_subplot(3, 4, 9)
    LineR1 = SubPltFig.add_subplot(3, 4, 2)
    LineR2 = SubPltFig.add_subplot(3, 4, 6)
    LineR3 = SubPltFig.add_subplot(3, 4, 10)
    ContR1 = SubPltFig.add_subplot(3, 4, 3)
    ContR2 = SubPltFig.add_subplot(3, 4, 7)
    ContR3 = SubPltFig.add_subplot(3, 4, 11)
    ContR12 = SubPltFig.add_subplot(3, 4, 4)
    ContR22 = SubPltFig.add_subplot(3, 4, 8)
    ContR32 = SubPltFig.add_subplot(3, 4, 12)

    ImgR1.imshow(Data1)
    ImgR2.imshow(Data2)
    ImgR3.imshow(Data3)
    ImgR1.yaxis.set_inverted(0)
    ImgR2.yaxis.set_inverted(0)
    ImgR3.yaxis.set_inverted(0)
    ImgR1.images[0].set_cmap('inferno')
    ImgR2.images[0].set_cmap('inferno')
    ImgR3.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=65,vmax=100)
    ImgR2.images[0].set_clim(vmin=70, vmax=100)
    ImgR3.images[0].set_clim(vmin=140, vmax=170)

    Yvals = Data1[:, 12]
    ErrorBars = numpy.sqrt(Yvals)
    LineR1.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars, linestyle='',marker='p')
    Yvals = Data2[:, 12]
    ErrorBars = numpy.sqrt(Yvals)
    LineR2.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars,linestyle='',marker='p')
    Yvals = Data3[:, 12]
    ErrorBars = numpy.sqrt(Yvals)
    LineR3.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars,linestyle='',marker='p')
    # LineR1.set_aspect('equal')
    # LineR1.set_box_aspect(1)
    # Changing axes size:
    # TempPosition=LineR1.get_position().bounds
    # LineR1.set_position([TempPosition[0],TempPosition[1],TempPosition[2]*2, TempPosition[3]*2])

    ContR1.contour(Data1, numpy.arange(65, 110, 15), cmap='inferno')
    ContR1.set_aspect('equal')
    ContR12.contourf(Data1, numpy.arange(65, 110, 15), cmap='inferno')
    ContR12.set_aspect('equal')

    ContR2.contour(Data2, numpy.arange(70, 100, 10), cmap='inferno')
    ContR2.set_aspect('equal')
    ContR22.contourf(Data2, numpy.arange(70, 100, 10), cmap='inferno')
    ContR22.set_aspect('equal')

    ContR3.contour(Data3, numpy.arange(130, 170, 10), cmap='inferno')
    ContR3.set_aspect('equal')
    ContR32.contourf(Data3, numpy.arange(130, 170, 10), cmap='inferno')
    ContR32.set_aspect('equal')

    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig,Grid_SubPltFig, ImgR1,ImgR2,ImgR3,LineR1,LineR2,LineR3,ContR1,ContR2,ContR3,ContR12,ContR22, ContR32

def Custom_Plot_2(Data1, Data2, Data3):
    SubPltFig = plt.figure()
    Rows = 3
    Columns = 2
    # Grid_SubPltFig = SubPltFig.add_gridspec(3,4)

    ImgR1 = SubPltFig.add_subplot(Rows, Columns, 1)
    Grid_SubPltFig = ImgR1.get_subplotspec()
    Grid_SubPltFig = Grid_SubPltFig.get_gridspec()
    Grid_SubPltFig.update(top=1-0.01, bottom=0.05, left=0, right=1, wspace=0.1, hspace=0.2)

    ImgR2 = SubPltFig.add_subplot(Rows, Columns, 3)
    ImgR3 = SubPltFig.add_subplot(Rows, Columns, 5)
    ContR1 = SubPltFig.add_subplot(Rows, Columns, 2)
    ContR2 = SubPltFig.add_subplot(Rows, Columns, 4)
    ContR3 = SubPltFig.add_subplot(Rows, Columns, 6)

    ImgR1.imshow(Data1)
    ImgR2.imshow(Data2)
    ImgR3.imshow(Data3)
    ImgR1.yaxis.set_inverted(0)
    ImgR2.yaxis.set_inverted(0)
    ImgR3.yaxis.set_inverted(0)
    ImgR1.images[0].set_cmap('inferno')
    ImgR2.images[0].set_cmap('inferno')
    ImgR3.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=110,vmax=150)
    ImgR2.images[0].set_clim(vmin=110, vmax=140)
    ImgR3.images[0].set_clim(vmin=250, vmax=300)
    ImgR1.tick_params(axis='both',labelsize=20)
    ImgR2.tick_params(axis='both', labelsize=20)
    ImgR3.tick_params(axis='both', labelsize=20)
    # LineR1.set_aspect('equal')
    # LineR1.set_box_aspect(1)
    # Changing axes size:
    # TempPosition=LineR1.get_position().bounds
    # LineR1.set_position([TempPosition[0],TempPosition[1],TempPosition[2]*2, TempPosition[3]*2])

    ContR1.contourf(Data1, numpy.arange(80, 150, 25), cmap='inferno')
    ContR1.set_aspect('equal')

    ContR2.contourf(Data2, numpy.arange(80, 140, 20), cmap='inferno')
    ContR2.set_aspect('equal')

    ContR3.contourf(Data3, numpy.arange(230, 330, 20), cmap='inferno')
    ContR3.set_aspect('equal')

    ContR1.tick_params(axis='both',labelsize=20)
    ContR2.tick_params(axis='both', labelsize=20)
    ContR3.tick_params(axis='both', labelsize=20)

    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig,Grid_SubPltFig, ImgR1,ImgR2,ImgR3,ContR1,ContR2,ContR3

def Custom_Plot_3(Data1, Data2, Data3):
    SubPltFig = plt.figure()
    Rows = 4
    Columns = 3
    # Grid_SubPltFig = SubPltFig.add_gridspec(3,4)


    #Grid_SubPltFig = ImgR1.get_subplotspec()
    #Grid_SubPltFig = Grid_SubPltFig.get_gridspec()
    Grid_SubPltFig = matplotlib.gridspec.GridSpec(Rows, Columns)
    Grid_SubPltFig.update(top=1 - 0.01, bottom=0.05, left=0.06, right=1 - 0.01, wspace=0.1, hspace=0.2)

    ImgR1 = SubPltFig.add_subplot(Grid_SubPltFig[0,0])
    ImgR2 = SubPltFig.add_subplot(Grid_SubPltFig[0,1])
    ImgR3 = SubPltFig.add_subplot(Grid_SubPltFig[0,2])
    LineR1 = SubPltFig.add_subplot(Grid_SubPltFig[1,0:2])
    LineR2 = SubPltFig.add_subplot(Grid_SubPltFig[2, 0:2])


    ImgR1.imshow(Data1)
    ImgR2.imshow(Data2)
    ImgR3.imshow(Data3)
    ImgR1.yaxis.set_inverted(0)
    ImgR2.yaxis.set_inverted(0)
    ImgR3.yaxis.set_inverted(0)
    ImgR1.images[0].set_cmap('inferno')
    ImgR2.images[0].set_cmap('inferno')
    ImgR3.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=110,vmax=150)
    ImgR2.images[0].set_clim(vmin=110, vmax=140)
    ImgR3.images[0].set_clim(vmin=250, vmax=300)
    ImgR1.tick_params(axis='both',labelsize=20)
    ImgR2.tick_params(axis='both', labelsize=20)
    ImgR3.tick_params(axis='both', labelsize=20)
    ImgR1.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    ImgR2.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    ImgR3.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    ImgR1.set_ylabel('2D detector - \nvertical axis (pixels)',fontsize=20)

    Yvals = Data1[1, :]
    ErrorBars = numpy.sqrt(Yvals)
    LineR1.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars, linestyle='',marker='p')
    Yvals = Data1[2, :]
    ErrorBars = numpy.sqrt(Yvals)
    LineR1.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars, linestyle='',marker='p')
    Yvals = Data1[3, :]
    ErrorBars = numpy.sqrt(Yvals)
    LineR1.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars, linestyle='',marker='p')

    Yvals = Data1[10, :]
    ErrorBars = numpy.sqrt(Yvals)
    LineR2.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars, linestyle='',marker='p')
    Yvals = Data1[12, :]
    ErrorBars = numpy.sqrt(Yvals)
    LineR2.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars, linestyle='',marker='p')
    Yvals = Data1[22, :]
    ErrorBars = numpy.sqrt(Yvals)
    LineR2.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars, linestyle='',marker='p')


    #LineR1.set_aspect('equal')
    #LineR1.set_box_aspect(1)
    # Changing axes size:
    # TempPosition=LineR1.get_position().bounds
    # LineR1.set_position([TempPosition[0],TempPosition[1],TempPosition[2]*2, TempPosition[3]*2])
    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig,Grid_SubPltFig, ImgR1,ImgR2,ImgR3

def Custom_Plot_4(Data1, Data2, Data3):
    SubPltFig = plt.figure()
    Rows = 1
    Columns = 3
    # Grid_SubPltFig = SubPltFig.add_gridspec(3,4)

    ContR1 = SubPltFig.add_subplot(Rows, Columns, 1)
    Grid_SubPltFig = ContR1.get_subplotspec()
    Grid_SubPltFig = Grid_SubPltFig.get_gridspec()
    Grid_SubPltFig.update(top=1-0.01, bottom=0.05, left=0.06, right=1-0.01, wspace=0.1, hspace=0.2)

    ContR2 = SubPltFig.add_subplot(Rows, Columns, 2)
    ContR3 = SubPltFig.add_subplot(Rows, Columns, 3)

    ContR1.contourf(Data1, numpy.arange(95, 150, 18), cmap='inferno')
    ContR1.set_aspect('equal')

    ContR2.contourf(Data2, numpy.arange(95, 150, 18), cmap='inferno')
    ContR2.set_aspect('equal')

    ContR3.contourf(Data3, numpy.arange(235, 320, 18), cmap='inferno')
    ContR3.set_aspect('equal')

    ContR1.tick_params(axis='both',labelsize=20)
    ContR2.tick_params(axis='both', labelsize=20)
    ContR3.tick_params(axis='both', labelsize=20)
    ContR1.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    ContR2.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    ContR3.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    ContR1.set_ylabel('2D detector - \nvertical axis (pixels)',fontsize=20)

    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig,Grid_SubPltFig,ContR1,ContR2,ContR3

def Custom_Plot_5(Data1, XLabel='2D detector - \nhorizontal axis (pixels)', YLabel='2D detector - \nvertical axis (pixels)',CLimMin=0,CLimMax=100,PlotSize=[0.2,0.15,0.7,0.8], Title = ''):
    """
    A function that plot image in parameter Data1. It plots a single axis in a figure
    Parameters:
        Data1 - the image as a 2D numpy array.
        XLabel - x axis label
        YLabel - y axis label
        CLimMax and CLimMin - color map limits
        PlotSize - size of axes in the figure. list of 4 numbers - [left, bottom, width, height]

    """
    SubPltFig = plt.figure()

    ImgR1 = SubPltFig.add_axes(PlotSize)

    ImgR1.imshow(Data1)
    ImgR1.yaxis.set_inverted(0)
    ImgR1.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR1.tick_params(axis='both',labelsize=30)
    ImgR1.set_xlabel(XLabel,fontsize=20)
    ImgR1.set_ylabel(YLabel,fontsize=20)
    ImgR1.set_title(Title)

    #Yvals = Data1[1, :]
    #ErrorBars = numpy.sqrt(Yvals)
    #LineR1.errorbar(numpy.arange(33), Yvals, yerr=ErrorBars, linestyle='',marker='p')


    #LineR1.set_aspect('equal')
    #LineR1.set_box_aspect(1)
    # Changing axes size:
    # TempPosition=LineR1.get_position().bounds
    # LineR1.set_position([TempPosition[0],TempPosition[1],TempPosition[2]*2, TempPosition[3]*2])
    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig, ImgR1

def Custom_Plot_6(Data1, XData = None, PixLine=None,XLabel='2D detector - horizontal axis (pixels)',YLabel='Counts',PlotSize=[0.06,0.15,0.9,0.9], YError = None):
    """
    A function that plots a line of pixels from parameter Data1 and row PixLine. It plots a single axis in a figure
    PlotSize is given to a new axes rect = 'left, bottom, width, height'
    """
    SubPltFig = plt.figure()

    LineR1 = SubPltFig.add_axes(PlotSize)

    if PixLine != None:
        Yvals = Data1[PixLine, :]
    else:
        Yvals = Data1

    if not isinstance(YError, type(None)):
        ErrorBars = YError
    else:
        ErrorBars = numpy.sqrt(Yvals)

    if type(XData) == type(None):
        LineR1.errorbar(numpy.arange(Yvals.shape[0]), Yvals, yerr=ErrorBars, linestyle='',marker='p')
    else:
        LineR1.errorbar(XData, Yvals, yerr=ErrorBars, linestyle='', marker='p')

    LineR1.yaxis.set_inverted(0)
    LineR1.tick_params(axis='both',labelsize=20)
    LineR1.set_xlabel(XLabel,fontsize=20)
    LineR1.set_ylabel(YLabel,fontsize=20)



    #LineR1.set_aspect('equal')
    #LineR1.set_box_aspect(1)
    # Changing axes size:
    # TempPosition=LineR1.get_position().bounds
    # LineR1.set_position([TempPosition[0],TempPosition[1],TempPosition[2]*2, TempPosition[3]*2])
    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig, LineR1

def Custom_Plot_7(Data1, XData = None, XLabel='Energy (eV)',YLabel='Counts',PlotSize=[0.08,0.15,0.85,0.8], Title=''):
    """
    A function that plots a histogram from Data1 and bin's left edges XData. It plots a single axis in a figure
    PlotSize is given to a new axes rect = 'left, bottom, width, height'
    """
    SubPltFig = plt.figure()

    HistAx = SubPltFig.add_axes(PlotSize)

    if type(XData) == type(None):
        HistObj = HistAx.bar(range(len(Data1)),Data1,width=1)

    else:
        # I played with align and width to get it correct.
        HistObj = HistAx.bar(XData,Data1,align='edge',width=XData[1]-XData[0])


    #LineR1.yaxis.set_inverted(0)
    HistAx.tick_params(axis='both',labelsize=15)
    HistAx.set_xlabel(XLabel,fontsize=20)
    HistAx.set_ylabel(YLabel,fontsize=20)
    HistAx.set_title(Title)



    #LineR1.set_aspect('equal')
    #LineR1.set_box_aspect(1)
    # Changing axes size:
    # TempPosition=LineR1.get_position().bounds
    # LineR1.set_position([TempPosition[0],TempPosition[1],TempPosition[2]*2, TempPosition[3]*2])
    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig, HistObj, HistAx

def Custom_Plot_8(Data1, XLabel='2D detector - \nhorizontal axis (pixels)', YLabel='2D detector - \nvertical axis (pixels)',CLimMin=0,CLimMax=100,PlotSize=[0.2,0.15,0.7,0.8]):
    """
    A function that plot image in parameter Data1 and it plots colorbar. It plots a single axis in a figure
    Parameters:
        Data1 - the image as a 2D numpy array.
        XLabel - x axis label
        YLabel - y axis label
        CLimMax and CLimMin - color map limits
        PlotSize - size of axes in the figure. list of 4 numbers - [left, bottom, width, height]

    """
    SubPltFig = plt.figure()

    ImgR1 = SubPltFig.add_axes(PlotSize)

    ImgR1.imshow(Data1)
    ImgR1.yaxis.set_inverted(0)
    ImgR1.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR1.tick_params(axis='both',labelsize=40)
    ImgR1.set_xlabel(XLabel,fontsize=40)
    ImgR1.set_ylabel(YLabel,fontsize=40)

    # Add colorbar
    ImgR1_Norm = ImgR1.images[0].norm
    ImgR1_CM = ImgR1.images[0].get_cmap()
    SM_Obj = matplotlib.cm.ScalarMappable(cmap=ImgR1_CM, norm=ImgR1_Norm)
    CB_1 = SubPltFig.colorbar(SM_Obj, ax=ImgR1,location='right',orientation='vertical')
    SubPltFig.axes[-1].tick_params(labelsize=40)

    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig, ImgR1, CB_1
def Add_Line_to_Axes(Ax):
    pass

def Plot_for_Colleagues_1(Rings1, Rings2,Rings3,Background1,Background2,Background3,SingleRegion1,SingleRegion2,SingleRegion3):
    SubPltFig = plt.figure()
    Rows = 3
    Columns = 3
    Grid_SubPltFig = matplotlib.gridspec.GridSpec(Rows, Columns)
    Grid_SubPltFig.update(top=1 - 0.03, bottom=0.08, left=0.01, right=1 - 0.02, wspace=0.3, hspace=0.2)

    Img1 = SubPltFig.add_subplot(Grid_SubPltFig[0,0])
    Img2 = SubPltFig.add_subplot(Grid_SubPltFig[1,0])
    Img3 = SubPltFig.add_subplot(Grid_SubPltFig[2,0])
    Reg1 = SubPltFig.add_subplot(Grid_SubPltFig[0,1])
    Reg2 = SubPltFig.add_subplot(Grid_SubPltFig[1,1])
    Reg3 = SubPltFig.add_subplot(Grid_SubPltFig[2,1])
    Back1 = SubPltFig.add_subplot(Grid_SubPltFig[0,2])
    Back2 = SubPltFig.add_subplot(Grid_SubPltFig[1,2])
    Back3 = SubPltFig.add_subplot(Grid_SubPltFig[2,2])


    Img1.imshow(Rings1)
    Img1.yaxis.set_inverted(0)
    Img1.images[0].set_cmap('inferno')
    Img1.images[0].set_clim(vmin=0, vmax=600)
    Img1.tick_params(axis='both',labelsize=20)
    #Img1.set_xlabel(XLabel,fontsize=40)
    #Img1.set_ylabel(YLabel,fontsize=40)
    My_Colorbar(Img1)

    Img2.imshow(Rings2)
    Img2.yaxis.set_inverted(0)
    Img2.images[0].set_cmap('inferno')
    Img2.images[0].set_clim(vmin=0, vmax=900)
    Img2.tick_params(axis='both',labelsize=20)
    My_Colorbar(Img2)

    Img3.imshow(Rings3)
    Img3.yaxis.set_inverted(0)
    Img3.images[0].set_cmap('inferno')
    Img3.images[0].set_clim(vmin=0, vmax=1000)
    Img3.tick_params(axis='both',labelsize=20)
    My_Colorbar(Img3)

    Reg1.imshow(SingleRegion1)
    Reg1.yaxis.set_inverted(0)
    Reg1.images[0].set_cmap('inferno')
    Reg1.images[0].set_clim(vmin=0, vmax=25)
    Reg1.tick_params(axis='both',labelsize=20)
    My_Colorbar(Reg1)

    Reg2.imshow(SingleRegion2)
    Reg2.yaxis.set_inverted(0)
    Reg2.images[0].set_cmap('inferno')
    Reg2.images[0].set_clim(vmin=0, vmax=30)
    Reg2.tick_params(axis='both',labelsize=20)
    My_Colorbar(Reg2)

    Reg3.imshow(SingleRegion3)
    Reg3.yaxis.set_inverted(0)
    Reg3.images[0].set_cmap('inferno')
    Reg3.images[0].set_clim(vmin=0, vmax=50)
    Reg3.tick_params(axis='both',labelsize=20)
    My_Colorbar(Reg3)

    ErrorBars = numpy.sqrt(Background1)
    Back1.errorbar(numpy.arange(Background1.shape[0]), Background1, yerr=ErrorBars, linestyle='',marker='p')
    XLabel = ''
    YLabel = 'Pairs counts'
    Back1.yaxis.set_inverted(0)
    Back1.tick_params(axis='both',labelsize=20)
    Back1.set_xlabel(XLabel,fontsize=20)
    Back1.set_ylabel(YLabel,fontsize=20)

    ErrorBars = numpy.sqrt(Background2)
    Back2.errorbar(numpy.arange(Background2.shape[0]), Background2, yerr=ErrorBars, linestyle='',marker='p')
    XLabel = ''
    YLabel = 'Pairs counts'
    Back2.yaxis.set_inverted(0)
    Back2.tick_params(axis='both',labelsize=20)
    Back2.set_xlabel(XLabel,fontsize=20)
    Back2.set_ylabel(YLabel,fontsize=20)

    ErrorBars = numpy.sqrt(Background3)
    Back3.errorbar(numpy.arange(Background3.shape[0]), Background3, yerr=ErrorBars, linestyle='',marker='p')
    XLabel = 'Arbitrary index of small areas on the left'
    YLabel = 'Pairs counts'
    Back3.yaxis.set_inverted(0)
    Back3.tick_params(axis='both',labelsize=20)
    Back3.set_xlabel(XLabel,fontsize=20)
    Back3.set_ylabel(YLabel,fontsize=20)

    return SubPltFig

def Plot_for_Colleagues_2(Rings1, Rings2,Rings3):
    SubPltFig = plt.figure()
    Rows = 3
    Columns = 1
    Grid_SubPltFig = matplotlib.gridspec.GridSpec(Rows, Columns)
    Grid_SubPltFig.update(top=1 - 0.03, bottom=0.08, left=0.01, right=1 - 0.02, wspace=0.3, hspace=0.2)

    Img1 = SubPltFig.add_subplot(Grid_SubPltFig[0,0])
    Img2 = SubPltFig.add_subplot(Grid_SubPltFig[1,0])
    Img3 = SubPltFig.add_subplot(Grid_SubPltFig[2,0])

    Img1.imshow(Rings1)
    Img1.yaxis.set_inverted(0)
    Img1.images[0].set_cmap('inferno')
    Img1.images[0].set_clim(vmin=0, vmax=230)
    Img1.tick_params(axis='both',labelsize=20)
    #Img1.set_xlabel(XLabel,fontsize=40)
    #Img1.set_ylabel(YLabel,fontsize=40)
    My_Colorbar(Img1)

    Img2.imshow(Rings2)
    Img2.yaxis.set_inverted(0)
    Img2.images[0].set_cmap('inferno')
    Img2.images[0].set_clim(vmin=0, vmax=150)
    Img2.tick_params(axis='both',labelsize=20)
    My_Colorbar(Img2)

    Img3.imshow(Rings3)
    Img3.yaxis.set_inverted(0)
    Img3.images[0].set_cmap('inferno')
    Img3.images[0].set_clim(vmin=0, vmax=200)
    Img3.tick_params(axis='both',labelsize=20)
    My_Colorbar(Img3)
    return SubPltFig

def Plot_for_Colleagues_3(Rings1, Rings2,Rings3):
    SubPltFig = plt.figure()
    Rows = 3
    Columns = 1
    Grid_SubPltFig = matplotlib.gridspec.GridSpec(Rows, Columns)
    Grid_SubPltFig.update(top=1 - 0.03, bottom=0.08, left=0.01, right=1 - 0.02, wspace=0.3, hspace=0.2)

    Img1 = SubPltFig.add_subplot(Grid_SubPltFig[0,0])
    Img2 = SubPltFig.add_subplot(Grid_SubPltFig[1,0])
    Img3 = SubPltFig.add_subplot(Grid_SubPltFig[2,0])

    Img1.imshow(Rings1)
    Img1.yaxis.set_inverted(0)
    Img1.images[0].set_cmap('inferno')
    Img1.images[0].set_clim(vmin=460, vmax=610)
    Img1.tick_params(axis='both',labelsize=20)
    #Img1.set_xlabel(XLabel,fontsize=40)
    #Img1.set_ylabel(YLabel,fontsize=40)
    My_Colorbar(Img1)

    Img2.imshow(Rings2)
    Img2.yaxis.set_inverted(0)
    Img2.images[0].set_cmap('inferno')
    Img2.images[0].set_clim(vmin=740, vmax=900)
    Img2.tick_params(axis='both',labelsize=20)
    My_Colorbar(Img2)

    Img3.imshow(Rings3)
    Img3.yaxis.set_inverted(0)
    Img3.images[0].set_cmap('inferno')
    Img3.images[0].set_clim(vmin=950, vmax=1200)
    Img3.tick_params(axis='both',labelsize=20)
    My_Colorbar(Img3)
    return SubPltFig

def Plot_for_CLEO_Main(Data1,Data2=0,Data3=0,Data4=0,Data5=0,Data6=0):
    SubPltFig = plt.figure()
    Rows = 2
    Columns = 3

    Grid_SubPltFig = matplotlib.gridspec.GridSpec(Rows, Columns)
    Grid_SubPltFig.update(top=1 - 0.01, bottom=0.11, left=0.01, right=1 - 0.01, wspace=0.01, hspace=0.15)

    ImgR1 = SubPltFig.add_subplot(Grid_SubPltFig[0,0])
    ImgR2 = SubPltFig.add_subplot(Grid_SubPltFig[0,1])
    ImgR3 = SubPltFig.add_subplot(Grid_SubPltFig[0,2])
    ImgR4 = SubPltFig.add_subplot(Grid_SubPltFig[1,0])
    ImgR5 = SubPltFig.add_subplot(Grid_SubPltFig[1,1])
    ImgR6 = SubPltFig.add_subplot(Grid_SubPltFig[1,2])

    ImgR1.imshow(Data1)
    CLimMin = 0
    CLimMax = 160
    ImgR1.yaxis.set_inverted(0)
    ImgR1.set_box_aspect(1)
    ImgR1.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR1.set_xticks([0, 10,20,30])
    ImgR1.tick_params(axis='both',labelsize=30)
    #ImgR1.set_xlabel('2D detector - horizontal axis (pixels)',fontsize=20)
    ImgR1.set_ylabel('2D detector - vertical axis (pixels)',fontsize=30)
    ImgR1.axes.yaxis.set_label_coords(-0.2, -0.2)

    ImgR2.imshow(Data2)
    CLimMin = 0
    CLimMax = 160
    ImgR2.yaxis.set_inverted(0)
    ImgR2.set_box_aspect(1)
    ImgR2.images[0].set_cmap('inferno')
    ImgR2.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR2.set_xticks([0, 10, 20, 30])
    ImgR2.tick_params(axis='both',labelsize=30)
    #ImgR2.set_xlabel('',fontsize=20)
    #ImgR2.set_ylabel('',fontsize=20)

    ImgR3.imshow(Data3)
    CLimMin = 0
    CLimMax = 160
    ImgR3.yaxis.set_inverted(0)
    ImgR3.set_box_aspect(1)
    ImgR3.images[0].set_cmap('inferno')
    ImgR3.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR3.set_xticks([0, 10, 20, 30])
    ImgR3.tick_params(axis='both',labelsize=30)
    #ImgR3.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    #ImgR3.set_ylabel('2D detector - \nvertical axis (pixels)',fontsize=20)


    ImgR4.imshow(Data4)
    CLimMin = 0
    CLimMax = 30
    ImgR4.yaxis.set_inverted(0)
    ImgR4.set_box_aspect(1)
    ImgR4.images[0].set_cmap('inferno')
    ImgR4.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR4.set_xticks([0, 10, 20, 30])
    ImgR4.tick_params(axis='both',labelsize=30)
    #ImgR4.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    #ImgR4.set_ylabel('2D detector - \nvertical axis (pixels)',fontsize=20)

    ImgR5.imshow(Data5)
    CLimMin = 0
    CLimMax = 60
    ImgR5.yaxis.set_inverted(0)
    ImgR5.set_box_aspect(1)
    ImgR5.images[0].set_cmap('inferno')
    ImgR5.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR5.set_xticks([0, 10, 20, 30])
    ImgR5.tick_params(axis='both',labelsize=30)
    ImgR5.set_xlabel('2D detector - horizontal axis (pixels)',fontsize=30)
    #ImgR1.axes.xaxis.set_label_coords(-0.2, -0.2)
    #ImgR5.set_ylabel('2D detector - \nvertical axis (pixels)',fontsize=20)

    if type(Data6) == list:
        CLimMin = 0
        CLimMax = 1

        ColorsList = ['b','g','r']
        IndNum = 0
        for CurrentImage in Data6:
            if type(CurrentImage) == numpy.ndarray:
                ImgR6.imshow(CurrentImage)
                ColorMapObj = matplotlib.colors.ListedColormap([(1,1,1,0),ColorsList[IndNum]],name='Haimcm1')
                ImgR6.images[IndNum].set_cmap(ColorMapObj)
                ImgR6.images[IndNum].set_clim(vmin=CLimMin, vmax=CLimMax)
                IndNum += 1


        ImgR6.yaxis.set_inverted(0)
        ImgR6.set_box_aspect(1)
        ImgR6.set_xticks([0, 10, 20, 30])
        ImgR6.tick_params(axis='both',labelsize=30)
    #ImgR6.set_xlabel('2D detector - horizontal axis (pixels)',fontsize=30)

    #ImgR1.set_aspect(2) # changes rate between scale of a unit in x and y. If
    # Changing axes size:
    # TempPosition=LineR1.get_position().bounds
    # LineR1.set_position([TempPosition[0],TempPosition[1],TempPosition[2]*2, TempPosition[3]*2])

    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig, ImgR1, ImgR2, ImgR3, ImgR4, ImgR5, ImgR6

def Plot_for_CLEO_Main_2(Data1,Data2=0,Data3=0):
    SubPltFig = plt.figure()
    Rows = 2
    Columns = 3

    Grid_SubPltFig = matplotlib.gridspec.GridSpec(Rows, Columns)
    Grid_SubPltFig.update(top=1 - 0.01, bottom=0.11, left=0.06, right=1 - 0.01, wspace=0.13, hspace=0.15)

    ImgR1 = SubPltFig.add_subplot(Grid_SubPltFig[0,0])
    ImgR2 = SubPltFig.add_subplot(Grid_SubPltFig[0,1])
    ImgR3 = SubPltFig.add_subplot(Grid_SubPltFig[0,2])
    ImgR4 = SubPltFig.add_subplot(Grid_SubPltFig[1,0])
    ImgR5 = SubPltFig.add_subplot(Grid_SubPltFig[1,1])
    ImgR6 = SubPltFig.add_subplot(Grid_SubPltFig[1,2])

    ImgR1.imshow(Data1)
    CLimMin = 0
    CLimMax = 260
    ImgR1.yaxis.set_inverted(0)
    ImgR1.set_box_aspect(1)
    ImgR1.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR1.set_xticks([0, 10,20,30])
    ImgR1.tick_params(axis='both',labelsize=30)
    #ImgR1.set_xlabel('2D detector - horizontal axis (pixels)',fontsize=20)
    ImgR1.set_ylabel('2D detector - vertical axis (pixels)',fontsize=30)
    ImgR1.axes.yaxis.set_label_coords(-0.2, -0.2)
    My_Colorbar(ImgR1,Tick_Label_Size=26)

    ImgR2.imshow(Data2)
    CLimMin = 0
    CLimMax = 260
    ImgR2.yaxis.set_inverted(0)
    ImgR2.set_box_aspect(1)
    ImgR2.images[0].set_cmap('inferno')
    ImgR2.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR2.set_xticks([0, 10, 20, 30])
    ImgR2.tick_params(axis='both',labelsize=30)
    My_Colorbar(ImgR2,Tick_Label_Size=26)
    #ImgR2.set_xlabel('2D detector - horizontal axis (pixels)', fontsize=30)
    #ImgR2.axes.xaxis.set_label_coords(0.5, -0.1)

    ImgR3.imshow(Data3)
    CLimMin = 0
    CLimMax = 260
    ImgR3.yaxis.set_inverted(0)
    ImgR3.set_box_aspect(1)
    ImgR3.images[0].set_cmap('inferno')
    ImgR3.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR3.set_xticks([0, 10, 20, 30])
    ImgR3.tick_params(axis='both',labelsize=30)
    #ImgR3.set_xlabel('2D detector - \nhorizontal axis (pixels)',fontsize=20)
    #ImgR3.set_ylabel('2D detector - \nvertical axis (pixels)',fontsize=20)
    My_Colorbar(ImgR3,Tick_Label_Size=26)

    # Plot simulations
    Data4_1,Data4_2,Data6_1,Data6_2,Data5,Custom_CMap = SimulationPlot.SimulationPlots()

    ImgR4.imshow(Data4_1,cmap='inferno')
    ImgR4.imshow(Data4_2, cmap=Custom_CMap)
    #CLimMin = 0
    #CLimMax = 160
    ImgR4.yaxis.set_inverted(0)
    ImgR4.set_box_aspect(1)
    #ImgR4.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    ImgR4.set_xticks([0, 10, 20, 30])
    ImgR4.tick_params(axis='both',labelsize=30)
    #ImgR4.set_xlabel('2D detector - horizontal axis (pixels)', fontsize=30)
    #ImgR4.axes.xaxis.set_label_coords(0.5, -0.1)

    ImgR5.imshow(Data5,cmap='inferno')
    ImgR5.yaxis.set_inverted(0)
    ImgR5.set_box_aspect(1)
    ImgR5.set_xticks([0, 10, 20, 30])
    ImgR5.tick_params(axis='both',labelsize=30)
    ImgR5.set_xlabel('2D detector - horizontal axis (pixels)', fontsize=30)
    ImgR5.axes.xaxis.set_label_coords(0.5, -0.15)

    ImgR6.imshow(Data6_1,cmap='inferno')
    ImgR6.imshow(Data6_2, cmap=Custom_CMap)
    ImgR6.yaxis.set_inverted(0)
    ImgR6.set_box_aspect(1)
    ImgR6.set_xticks([0, 10, 20, 30])
    ImgR6.tick_params(axis='both',labelsize=30)

    #ImgR6.set_xlabel('2D detector - horizontal axis (pixels)',fontsize=30)

    #ImgR1.set_aspect(2) # changes rate between scale of a unit in x and y. If
    # Changing axes size:
    # TempPosition=LineR1.get_position().bounds
    # LineR1.set_position([TempPosition[0],TempPosition[1],TempPosition[2]*2, TempPosition[3]*2])

    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig, ImgR1, ImgR2, ImgR3, ImgR4, ImgR5,ImgR6

def Plot_for_Paper_Precedure1(Pairs_Image_Tresh_Low, Pairs_Image_Tresh_Med, Pairs_Image_Tresh_High):
    PltFig = plt.figure()
    Rows = 1
    Columns = 3

    Grid_SubPltFig = matplotlib.gridspec.GridSpec(Rows, Columns)
    Grid_SubPltFig.update(top=1 - 0.05, bottom=0.11, left=0.06, right=1 - 0.01, wspace=0.3, hspace=0.2)

    ImgR1 = PltFig.add_subplot(Grid_SubPltFig[0,0])
    ImgR2 = PltFig.add_subplot(Grid_SubPltFig[0,1])
    ImgR3 = PltFig.add_subplot(Grid_SubPltFig[0,2])

    ImgR1.imshow(Pairs_Image_Tresh_Low)
    CLimMin = 0
    CLimMax = int(Pairs_Image_Tresh_Low.max()/2)
    ImgR1.yaxis.set_inverted(0)
    ImgR1.set_box_aspect(1)
    ImgR1.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    #ImgR1.set_xticks([0, 10,20,30])
    ImgR1.tick_params(axis='both',labelsize=30)
    #ImgR1.set_xlabel('2D detector - horizontal axis (pixels)',fontsize=20)
    ImgR1.set_ylabel('Vertical axis (pixels)',fontsize=30)
    ImgR1.axes.yaxis.set_label_coords(-0.2, 0.5)
    My_Colorbar(ImgR1,Tick_Label_Size=22)

    ImgR2.imshow(Pairs_Image_Tresh_Med)
    CLimMin = 0
    CLimMax = int(Pairs_Image_Tresh_Med.max()/2)
    ImgR2.yaxis.set_inverted(0)
    ImgR2.set_box_aspect(1)
    ImgR2.images[0].set_cmap('inferno')
    ImgR2.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    #ImgR2.set_xticks([0, 10,20,30])
    ImgR2.tick_params(axis='both',labelsize=30)
    My_Colorbar(ImgR2,Tick_Label_Size=22)
    ImgR2.set_xlabel('Horizontal axis (pixels)', fontsize=30)
    ImgR2.axes.xaxis.set_label_coords(0.2, -0.2)


    ImgR3.imshow(Pairs_Image_Tresh_High)
    CLimMin = 0
    CLimMax = int(Pairs_Image_Tresh_High.max()/2)
    ImgR3.yaxis.set_inverted(0)
    ImgR3.set_box_aspect(1)
    ImgR3.images[0].set_cmap('inferno')
    ImgR3.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    #ImgR3.set_xticks([0, 10,20,30])
    ImgR3.tick_params(axis='both',labelsize=30)
    My_Colorbar(ImgR3,Tick_Label_Size=22)
    return ImgR1,ImgR2,ImgR3,Grid_SubPltFig

def Plot_for_Paper_Precedure2(Pairs_Image_Tresh_Low, Pairs_Image_Tresh_Med, Pairs_Image_Tresh_High,Fixed_Counts_Low, Fixed_Counts_Med, Fixed_Counts_High):
    PltFig = plt.figure()
    Rows = 2
    Columns = 3

    Grid_SubPltFig = matplotlib.gridspec.GridSpec(Rows, Columns)
    Grid_SubPltFig.update(top=1 - 0.02, bottom=0.1, left=0.06, right=1 - 0.01, wspace=0.01, hspace=0.3)
    '''
    ImgR1 = PltFig.add_subplot(Grid_SubPltFig[0,0])
    ImgR2 = PltFig.add_subplot(Grid_SubPltFig[1,0])
    ImgR3 = PltFig.add_subplot(Grid_SubPltFig[2,0])
    Line1 = PltFig.add_subplot(Grid_SubPltFig[0,1])
    Line2 = PltFig.add_subplot(Grid_SubPltFig[1,1])
    Line3 = PltFig.add_subplot(Grid_SubPltFig[2,1])
    '''
    ImgR1 = PltFig.add_subplot(Grid_SubPltFig[0,0])
    ImgR2 = PltFig.add_subplot(Grid_SubPltFig[0,1])
    ImgR3 = PltFig.add_subplot(Grid_SubPltFig[0,2])
    Line1 = PltFig.add_subplot(Grid_SubPltFig[1,0])
    Line2 = PltFig.add_subplot(Grid_SubPltFig[1,1])
    Line3 = PltFig.add_subplot(Grid_SubPltFig[1,2])

    ImgR1.imshow(Pairs_Image_Tresh_Low)
    CLimMin = 0
    CLimMax = int(45/2)
    ImgR1.yaxis.set_inverted(0)
    ImgR1.set_box_aspect(1)
    ImgR1.images[0].set_cmap('inferno')
    ImgR1.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    #ImgR1.set_xticks([0, 10,20,30])
    ImgR1.tick_params(axis='both',labelsize=26)
    #ImgR1.set_xlabel('2D detector - horizontal axis (pixels)',fontsize=20)
    ImgR1.set_ylabel('Vertical axis (pixels)',fontsize=26)
    ImgR1.axes.yaxis.set_label_coords(-0.25, 0.4)
    My_Colorbar(ImgR1,Tick_Label_Size=26)

    ImgR2.imshow(Pairs_Image_Tresh_Med)
    CLimMin = 0
    CLimMax = int(35/2)
    ImgR2.yaxis.set_inverted(0)
    ImgR2.set_box_aspect(1)
    ImgR2.images[0].set_cmap('inferno')
    ImgR2.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    #ImgR2.set_xticks([0, 10,20,30])
    ImgR2.tick_params(axis='both',labelsize=26)
    ImgR2.set_xlabel('Horizontal axis (pixels)', fontsize=26)
    My_Colorbar(ImgR2,Tick_Label_Size=26)

    ImgR3.imshow(Pairs_Image_Tresh_High)
    CLimMin = 0
    CLimMax = int(12/2)
    ImgR3.yaxis.set_inverted(0)
    ImgR3.set_box_aspect(1)
    ImgR3.images[0].set_cmap('inferno')
    ImgR3.images[0].set_clim(vmin=CLimMin,vmax=CLimMax)
    #ImgR3.set_xticks([0, 10,20,30])
    ImgR3.tick_params(axis='both',labelsize=26)
    My_Colorbar(ImgR3,Tick_Label_Size=26)

    ErrorBars = numpy.sqrt(Fixed_Counts_Low)
    Line1.errorbar(numpy.arange(Fixed_Counts_Low.shape[0]), Fixed_Counts_Low, yerr=ErrorBars, linestyle='',marker='p')
    Line1.tick_params(axis='both', labelsize=26)
    Line1.set_ylabel('Number of pairs (Counts)', fontsize=26)
    Line1.axes.yaxis.set_label_coords(-0.3, 0.4)
    Line1.set_box_aspect(1)
    ErrorBars = numpy.sqrt(Fixed_Counts_Med)
    Line2.errorbar(numpy.arange(Fixed_Counts_Med.shape[0]), Fixed_Counts_Med, yerr=ErrorBars, linestyle='',marker='p')
    Line2.tick_params(axis='both', labelsize=26)
    Line2.set_xlabel('Fixed area index', fontsize=26)
    Line2.set_box_aspect(1)
    ErrorBars = numpy.sqrt(Fixed_Counts_High)
    Line3.errorbar(numpy.arange(Fixed_Counts_High.shape[0]), Fixed_Counts_High, yerr=ErrorBars, linestyle='',marker='p')
    Line3.tick_params(axis='both', labelsize=26)
    Line3.set_box_aspect(1)


    return ImgR1,ImgR2,ImgR3, Line1, Line2, Line3

def Plot_Line_Between_Points(X1,Y1,X2,Y2, Old_Axis=None, XLabel='2D detector - horizontal axis (pixels)',YLabel='2D detector - vertical axis (pixels)',PlotSize=[0.06,0.15,0.9,0.9]):
    if isinstance(Old_Axis,type(None)):
        SubPltFig = plt.figure()
        Ax1 = SubPltFig.add_axes(PlotSize)
        Ax1.set_xlabel(XLabel, fontsize=20)
        Ax1.set_ylabel(YLabel, fontsize=20)
        Ax1.plot([X1, X2], [Y1, Y2], linestyle='--', marker='D')

    else:
        Old_Axis.plot([X1, X2], [Y1, Y2], linestyle='--', marker='D')
        Ax1 = Old_Axis
        SubPltFig = Ax1.figure

    SubPltFig.canvas.draw()
    SubPltFig.show()
    return SubPltFig, Ax1

def Plor_Circle_from_Radius(X,Y,Radius,Old_Axis=None, Line_Width = 10,Line_Color='r', XLabel='2D detector - horizontal axis (pixels)', YLabel='2D detector - vertical axis (pixels)', PlotSize=[0.06,0.15,0.9,0.9]):
    if isinstance(Old_Axis,type(None)):
        SubPltFig = plt.figure()
        Ax1 = SubPltFig.add_axes(PlotSize)
        Ax1.set_xlabel(XLabel, fontsize=20)
        Ax1.set_ylabel(YLabel, fontsize=20)
    else:
        Ax1 = Old_Axis
        SubPltFig = Ax1.figure

    Circ = Patch.Circle((X,Y), Radius,fill=False,linewidth=Line_Width,edgecolor=Line_Color)
    Ax1.add_patch(Circ)

    return SubPltFig,Ax1


class Plotting_Results_Class:
    '''
    Object of this class takes numpy arrays and plots them.
    The object stores all the figures and helps display and manipulate them.
    It mainly simplifies what Matplotlib is already doing.
    It manages the created figures and allows them to be opened again after closing.
    It allows plotting together few images on the same axes
    '''

    def __init__(self, FrameData = None, AxTitle = 'First figure'):
        '''
        Should give numpy array as parameter, and it will be the first figure, axes and image
        Should also give title to the axis
        '''
        self.FigList = list()
        print('A new plotting object was generated! Congrats!')
        try:
            self.Add_Fig(AxTitle)
            if type(FrameData) == numpy.ndarray:
                self.Add_Image(0,FrameData)
        except:
            print('Object initialization error - please check the parameters given to the class')

    def Add_Fig(self, AxTitle = None):
        '''
        Method that creates new figure and saves it in the attribute FigList of the object.
        Creates one axis in the figure. In this class I assume there is only one axis in each figure
        '''
        try:
            Fig = plt.figure()
            self.FigList.append(Fig)
            Fig.gca()
        except:
            print('Add_Fig method error - looks fundamental error - try to create new object from class Plotting_Results_Class')
        try:
            if AxTitle == None:
                Fig.axes[0].set_title('Figure number ' + str(len(self.FigList)))
            else:
                Fig.axes[0].set_title(AxTitle)
        except:
            print('Add_Fig method error - while updating the title of the axis. Please check the axis title that was given in parameter AxTitle')


    def Add_Image(self,Fig_Ind,FrameData):
        try:
            self.FigList[Fig_Ind].axes[0].imshow(FrameData)
            self.FigList[Fig_Ind].canvas.draw()
        except:
            print('Method Add_Image error - check given 2D numpy array of image is correct and check figure index given axists in attribute FigList')


    def Set_Axes_Parameters(self,Fig_Ind,Param=None):
        '''
        Method that changes title, alpha, axes names and axis inversion for axis in given figure index
        '''
        try:
            if Param != None:
                # Trying to work with set:
                Available_Parameters = set(['Title','Color','Facecolor','Axis Alpha','Y Invert', 'X Invert'])
                Required_Parameters = set(Param.keys())
                print('The available parameters to change: {}'.format(Available_Parameters))
                print('The available parameters from the requested ones are: {}'.format(Available_Parameters.intersection(Required_Parameters)))
                if Param.__contains__('Title'):
                    self.FigList[Fig_Ind].axes[0].set_title(Param['Title'])
                if Param.__contains__('Axis Alpha'):
                    Old_FaceColor = list(self.FigList[Fig_Ind].axes[0].get_facecolor())
                    if len(Old_FaceColor) == 4:
                        Old_FaceColor[-1] = Param['Axis Alpha']
                        self.FigList[Fig_Ind].axes[0].set_facecolor(Old_FaceColor)
                    else:
                        print('Something is wrong with the facecolor value of the axis!')
                if Param.__contains__('Y Invert'):
                    self.FigList[Fig_Ind].axes[0].yaxis.set_inverted(Param['Y Invert'])
                if Param.__contains__('X Invert'):
                    self.FigList[Fig_Ind].axes[0].xaxis.set_inverted(Param['X Invert'])

                self.FigList[Fig_Ind].canvas.draw()
        except:
            print('Set_Axes_Parameters method error - check the figure index and the parameters requested to change')

    def Set_Image_Parameters(self,Fig_Ind, Im_Ind,Param=None):
        '''
        Method that changes color map, color map limits, alpha
        Parameters are the figure index, the image index and a dictionary with image parameters
        '''
        try:
            if Param != None:
                # Trying to work with set:
                Available_Parameters = set(['Color Map','CMap Max','CMap Min','Alpha'])
                Required_Parameters = set(Param.keys())
                print('The available parameters to change: {}'.format(Available_Parameters))
                print('The available parameters from the requested ones are: {}'.format(Available_Parameters.intersection(Required_Parameters)))
                if Param.__contains__('Color Map'):
                    # Available cmaps: gray,pink,spring,summer, autumn, winter,cool,hot,copper etc. (https://matplotlib.org/stable/tutorials/colors/colormaps.html#ibm)
                    self.FigList[Fig_Ind].axes[0].images[Im_Ind].set_cmap(Param['Color Map'])
                if Param.__contains__('Alpha'):
                    self.FigList[Fig_Ind].axes[0].images[Im_Ind].set_alpha(Param['Alpha'])
                if Param.__contains__('CMap Max'):
                    self.FigList[Fig_Ind].axes[0].images[Im_Ind].set_clim(vmax=Param['CMap Max'])
                if Param.__contains__('CMap Min'):
                    self.FigList[Fig_Ind].axes[0].images[Im_Ind].set_clim(vmin=Param['CMap Min'])

                self.FigList[Fig_Ind].canvas.draw()
        except:
            print('Set_Image_Parameters method error - check the figure index, image index and the parameters requested to change')


    def Print_All_Figs(self):
        print('All the figures and axes in this object:')
        for ind,Fig in enumerate(self.FigList):
            print('Figure index: {}, Axes title: {}'.format(ind,Fig.axes[0].get_title()))
            for Imag_Ind, Imag in enumerate(Fig.axes[0].images):
                print('\tImage index: {}, Axis title: {}'.format(Imag_Ind, Imag.axes.get_title()))

    def Show_All_Figs(self, Block = True):
        '''
        A method that plots all the figures in the object FigList attribute.
        Using pyplot.show() is not enough:
        If just running the program, it closes and all the data is lost after closing the figures.
        If working in interpreter mode, after closing the figures, pyplot.show() doesn't show them anymore.
        Also, if using  pyplot.show(block=False) to work with the figures while they are open, it works only in interpreter mode but the figures don't apear if just running the program.
        '''

        try:
            for Fig_Ind,Fig in enumerate(self.FigList,start=1):
                #plt.figure(Fig)
                Fig.show()
                #Old part, please ignore
                #Temp_Fig = plt.figure(Fig_Ind)
                #Temp_Man = Temp_Fig.canvas.manager
                #Temp_Man.canvas.figure = Fig

            # Need this in non-interpreter mode to hold the figures alive and not closing the program
            print('Note: If you want to work in interactive mode (not blocking the interpreter and allowing editing the figures), give parameter False to the method')
            plt.show(block=Block)
        except:
            print('Show_All_Figs method error - check that the list of figures attribute FigList is not empty')

    def Turn_Off_Images_In_Fig(self,Fig_Ind):
        try:
            for Imag in self.FigList[Fig_Ind].axes[0].images:
                Imag.set_visible(False)
            self.FigList[Fig_Ind].canvas.draw()
        except:
            print('Turn_Off_Images_In_Fig method error - please check the index of the figure that was given in parameter Fig_Ind')

    def Turn_On_Next_Image(self,Fig_Ind):
        '''
        Method that turns on the next image.
        Should give the index of the figure.
        '''
        try:
            for Imag in self.FigList[Fig_Ind].axes[0].images:
                if Imag.get_visible() == False:
                    Imag.set_visible(True)
                    break
            self.FigList[Fig_Ind].canvas.draw()
        except:
            print('Turn_On_Next_Image method error - please check the figure index that was provided')

    def Switch_One_Image(self,Fig_Ind,Image_Ind):
        '''
        Method that turns on or off one image.
        Should give as parameters the index of the figure and the image
        '''
        try:
            Imag = self.FigList[Fig_Ind].axes[0].images[Image_Ind]
            Im_Switch = Imag.get_visible()
            if Im_Switch:
                Imag.set_visible(False)
            else:
                Imag.set_visible(True)
            self.FigList[Fig_Ind].canvas.draw()
        except:
            print('Switch_One_Image method error - please check indexes of the figure or the image')