import csv
import glob
import os
import os.path
from collections import defaultdict, OrderedDict

import numpy as np
import wx
import wx.lib.agw.cubecolourdialog as ccd
from wx.lib.floatcanvas import NavCanvas, FloatCanvas

# import DotSizeFrame

IMAGE_FORMATS = [".jpg", ".png", ".ppm", ".pgm", ".gif", ".tif", ".tiff", ]


class FaceMapperFrame(wx.Frame):
    def __init__(self, parent, id, name, image_dir, n_points=None, scale=1.0, isVideo=False):
        wx.Frame.__init__(self, parent, id, name)

        if isVideo:
            os.mkdir(image_dir + '_PICS')
            print("ffmpeg -i {0} -vf fps=1/5 {1}".format(image_dir, image_dir + '_PICS/out%02d.png'))
            os.system("ffmpeg -i {0} -vf fps=1/5 {1}".format(image_dir, image_dir + '_PICS/out%02d.png'))
            image_dir = image_dir + '_PICS'
        # ---------------- Basic Data -------------------
        self.image_dir = image_dir
        self.n_points = n_points
        self.image_names = []
        self.current_image = None
        self.image_name = None
        self.prev_image_name = None
        self.scale = scale
        for files in IMAGE_FORMATS:
            self.image_names.extend([os.path.basename(x) for x in glob.glob(self.image_dir + '/*{0}'.format(files))])

        self.image_names.sort()
        self.filename = None
        self.coords = defaultdict()
        self.faceParts = OrderedDict()
        self.faceLabels = []

        self.firstColors = True
        self.resetFaceParts(firstTime=True)
        self.resetFaceLabels()

        self.faceNums = []
        self.resetFaceNums()

        self.first_click = True

        self.dotNum = 0
        self.partCounter = 1
        self.totalDotNum = 0
        self.dotDiam = 1.1
        self.colourData = wx.ColourData()

        for facePart in self.faceParts.keys():
            self.totalDotNum += self.faceParts[facePart][1]

        self.nullArray = np.array([-1.0, -1.0, -1.0, -1.0])
        self.coordMatrix = np.zeros((len(self.image_names), self.totalDotNum, 4))
        self.coordMatrix.fill(-1.0)

        # ------------- Other Components ----------------
        self.CreateStatusBar()
        # ------------------- Menu ----------------------
        filemenu = wx.Menu()
        id_about = wx.NewId()
        id_open = wx.NewId()
        id_save = wx.NewId()
        id_save_as = wx.NewId()
        id_exit = wx.NewId()
        id_dotSize = wx.NewId()
        # File Menu
        filemenu.Append(wx.ID_ABOUT, wx.EmptyString)
        filemenu.AppendSeparator()
        filemenu.Append(wx.ID_OPEN, wx.EmptyString)
        filemenu.Append(wx.ID_SAVE, wx.EmptyString)
        filemenu.Append(wx.ID_SAVEAS, wx.EmptyString)
        filemenu.AppendSeparator()
        filemenu.Append(wx.ID_EXIT, wx.EmptyString)

        # options = wx.Menu()
        # options.Append(id_dotSize, "&Change Dot Size")

        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")  # Adding the "filemenu" to the MenuBar
        # menuBar.Append(options, "&Options")
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.

        # ----------------- Image List ------------------
        self.leftBox = wx.BoxSizer(wx.VERTICAL)
        self.list = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER,
                               choices=sorted(self.image_names))
        self.leftBox.Add(self.list, 1, wx.EXPAND)

        # ----------------- Image Display ---------------
        NC = NavCanvas.NavCanvas(self, Debug=0, BackgroundColor="BLACK")
        self.Canvas = NC.Canvas
        self.Canvas.MinScale = 14
        self.Canvas.MaxScale = 500
        self.imHeight = 50

        # ----------------- Counter Display ------------
        self.counterBox = wx.BoxSizer(wx.VERTICAL)
        self.counterList = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER | wx.LB_SORT,
                                      choices=self.faceLabels)
        self.saveButton = wx.Button(self, wx.NewId(), label='Press to Save and Continue')
        self.counterBox.Add(self.counterList, 1, wx.EXPAND)
        self.counterBox.Add(self.saveButton, 1, wx.EXPAND)

        # ----------------- Window Layout  -------------
        self.mainBox = wx.BoxSizer(wx.HORIZONTAL)
        self.mainBox.Add(self.leftBox, 1, wx.EXPAND)
        self.mainBox.Add(NC, 3, wx.EXPAND)
        self.mainBox.Add(self.counterBox, 1, wx.EXPAND)

        self.box = wx.BoxSizer(wx.VERTICAL)
        self.box.Add(self.mainBox, 5, wx.EXPAND)

        self.selectionText = wx.StaticText(self, wx.NewId(), label='Nothing currently selected')
        self.box.Add(self.selectionText, 1, wx.EXPAND)

        self.SetAutoLayout(True)
        self.SetSizer(self.box)
        self.Layout()

        # -------------- Event Handling ----------------
        self.Bind(wx.EVT_LISTBOX, self.onSelect, id=self.list.GetId())
        self.Bind(wx.EVT_LISTBOX, self.colorSelect, id=self.counterList.GetId())
        self.Bind(wx.EVT_BUTTON, self.onButtonSave, id=self.saveButton.GetId())
        self.BindAllMouseEvents()
        self.Bind(wx.EVT_MENU, self.onOpen, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self.onSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.onSaveAs, id=wx.ID_SAVEAS)

    def resetFaceParts(self, firstTime):
        if self.firstColors:
            self.faceParts.clear()
            self.faceParts["Left Eye"] = [0, 6, wx.TheColourDatabase.Find('Red').GetAsString()]
            self.faceParts["Right Eye"] = [0, 6, wx.TheColourDatabase.Find('Orange').GetAsString()]
            self.faceParts["Mouth"] = [0, 20, wx.TheColourDatabase.Find('Yellow').GetAsString()]
            self.faceParts["Jaw"] = [0, 17, wx.TheColourDatabase.Find('Green').GetAsString()]
            self.faceParts["Eyebrows"] = [0, 10, wx.TheColourDatabase.Find('Blue').GetAsString()]
            self.faceParts["Nose"] = [0, 9, wx.TheColourDatabase.Find('Violet').GetAsString()]
        else:
            self.faceParts["Left Eye"] = [0, 6, self.faceParts["Left Eye"][2]]
            self.faceParts["Right Eye"] = [0, 6, self.faceParts["Right Eye"][2]]
            self.faceParts["Mouth"] = [0, 20, self.faceParts["Mouth"][2]]
            self.faceParts["Jaw"] = [0, 17, self.faceParts["Jaw"][2]]
            self.faceParts["Eyebrows"] = [0, 10, self.faceParts["Eyebrows"][2]]
            self.faceParts["Nose"] = [0, 9, self.faceParts["Nose"][2]]

    def resetFaceLabels(self):
        self.faceLabels = []
        partIndex = 1
        for facePart in self.faceParts.keys():
            self.faceLabels.append("{0}. {1}: {2} out of {3}".format(partIndex, facePart, self.faceParts[facePart][0],
                                                                     self.faceParts[facePart][1]))
            partIndex += 1

    def resetFaceNums(self):
        self.faceNums.clear()
        for facePart in self.faceParts.keys():
            split = facePart.split()
            abbr = ''
            for i in range(len(split)):
                abbr += split[i].strip()[0]
            for index in range(self.faceParts[facePart][1]):
                self.faceNums.append(abbr + str(index + 1))

    def BindAllMouseEvents(self):
        self.Canvas.Unbind(FloatCanvas.EVT_MOTION)
        self.Canvas.Bind(FloatCanvas.EVT_LEFT_DOWN, self.OnLeftDown)


    def openCSVFile(self, path):

        reader = csv.reader(open(path, "rb"))
        first = True
        eyes = False
        coords = {}
        for row in reader:
            filename = row[0]
            row = row[1:]

            if len(row) % 2 != 0:
                print("Error Loading File")
                raise TypeError("Odd number of values in this row")

            points = []
            for i in range(0, len(row), 2):
                point = (float(row[i]), float(row[i + 1]))
                points.append(point)

            coords[filename] = points
        print("CSV File Data: ", coords)
        self.coords = coords

    def onButtonSave(self, event):
        i = self.image_names.index(self.image_name)
        if len(self.image_names) > 1:
            if self.image_name:
                self.prev_image_name = self.image_name
                if self.n_points != None and len(self.coords[self.image_name]) != self.n_points:
                    print("ERROR: incorrect number of points.")

            self.image_name = self.image_names[i + 1]
            self.imageIndex = self.image_names.index(self.image_name)
            self.mirrorImage(event, shouldSave=True)
        else:
            print('You\'re Done!')

    def mirrorImage(self, event, shouldSave):
        if self.imageIndex >= 1 and np.array_equal(self.coordMatrix[self.imageIndex, 0,],
                                                                           self.nullArray):
            self.coordMatrix[self.imageIndex,] = self.coordMatrix[self.imageIndex - 1,]
            for circle in self.coordMatrix[self.imageIndex,]:
                if not np.array_equal(circle, self.nullArray):
                    circle[2] = 0

        filename = os.path.join(self.image_dir, self.image_name)
        self.current_image = wx.Image(filename)

        # if not self.prev_image_name:
        #    self.first_click = True
        self.list.SetSelection(self.imageIndex)
        self.DisplayImage(Zoom=True)
        if shouldSave:
            self.onSave(event)


    def save(self, path):
        ''' Save the coords to a csv file. '''
        writer = csv.writer(open(path, 'w'))

        firstRow = [' ']
        for faceNum in self.faceNums:
            firstRow.append(faceNum)
        writer.writerow(firstRow)
        for image in self.image_names:
            if not np.array_equal(self.coordMatrix[self.image_names.index(image), 0,], self.nullArray):
                row = [image]
                for point in self.coordMatrix[self.image_names.index(image),]:
                    if not np.array_equal(self.nullArray, point):
                        row.append(point[0])
                        row.append(point[1])
                writer.writerow(row)

    def onSelect(self, event):
        # if self.image_name:
        #    self.prev_image_name = self.image_name
        #    if self.n_points != None and len(self.coords[self.image_name]) != self.n_points:
        #        print("ERROR: incorrect number of points.")

        self.image_name = event.GetString()
        self.imageIndex = self.image_names.index(self.image_name)
        for i in range(len(self.image_names)):
            if i != self.imageIndex:
                for circle in self.coordMatrix[i,]:
                    if not np.array_equal(circle, self.nullArray):
                        circle[2] = 0
        self.mirrorImage(event, shouldSave=False)

    def colorSelect(self, event):
        num = event.GetInt()
        name = list(self.faceParts.keys())[num]
        self.colourData.SetColour(self.faceParts[name][2])
        colorDlg = ccd.CubeColourDialog(self, self.colourData)
        if colorDlg.ShowModal() == wx.ID_OK:
            self.firstColors = False
            self.colourData = colorDlg.GetColourData()
            self.faceParts[name][2] = self.colourData.GetColour().GetAsString()
            self.DisplayImage(Zoom=False)

    def DisplayImage(self, Zoom):
        if Zoom:
            self.Canvas.InitAll()
            if self.current_image:
                im = self.current_image.Copy()
                # self.imHeight = im.GetHeight()
                bm = im.ConvertToBitmap()
                self.Canvas.AddScaledBitmap(bm, XY=(0, 0), Height=self.imHeight, Position='tl')
                self.Canvas.ZoomToBB()

        self.resetFaceParts(firstTime=True)
        partCounter = 0
        for index, circle in enumerate(self.coordMatrix[self.imageIndex,]):
            if not (np.array_equal(circle, self.nullArray)) and circle[2] == 0:
                if circle[3] == -1.0:
                    circle[3] = self.dotDiam
                diam = circle[3]

                if index < len(self.Canvas._ForeDrawList):
                    self.Canvas._ForeDrawList[index].XY = circle[0:2]
                    self.Canvas._ForeDrawList[index].SetDiameter(diam)
                else:
                    C = FloatCanvas.Circle(XY=circle[0:2], Diameter=diam, LineWidth=.5, LineColor='Red',
                                           FillStyle='Transparent', InForeground=True)

                    C = self.Canvas.AddObject(C)
                    C.Bind(FloatCanvas.EVT_FC_LEFT_DOWN, self.CircleLeftDown)
                    C.Bind(FloatCanvas.EVT_FC_RIGHT_DOWN, self.CircleResize)
                    C.Bind(FloatCanvas.EVT_FC_ENTER_OBJECT, self.CircleHover)
                    C.Bind(FloatCanvas.EVT_FC_LEAVE_OBJECT, self.selectionReset)
                circle[2] = 1

        for circle in self.Canvas._ForeDrawList:
            facePart = self.faceParts[list(self.faceParts.keys())[partCounter]]
            facePart[0] += 1
            circle.SetColor(facePart[2])
            if facePart[0] == facePart[1]:
                partCounter += 1
            self.makeFaceLabel(circle)

        self.counterList.Clear()
        self.resetFaceLabels()
        self.counterList.Set(self.faceLabels)
        self.Canvas.Draw()

    def OnLeftDown(self, event):
        self.AddCoords(event)

    def AddCoords(self, event):
        if len(self.Canvas._ForeDrawList) < self.totalDotNum:
            self.coordMatrix[self.imageIndex, len(self.Canvas._ForeDrawList), 0:3] = np.append(event.Coords,
                                                                                               np.array([0.0]))
            self.DisplayImage(Zoom=False)



    def CircleLeftDown(self, object):
        self.draggingCircle = object
        self.draggingCircleIndex = self.Canvas._ForeDrawList.index(self.draggingCircle)
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.drag)
        self.Canvas.Draw()

    def drag(self, event):
        if wx.GetMouseState().LeftIsDown():
            self.coordMatrix[self.imageIndex, self.draggingCircleIndex, 0:2] = event.Coords
            self.Canvas._ForeDrawList[self.draggingCircleIndex].XY = event.Coords
            self.Canvas.Draw()
        else:
            self.BindAllMouseEvents()


    def CircleResize(self, object):
        self.preSizeCoords = object.XY
        self.reSizingCircle = object
        self.reSizingCircleIndex = self.Canvas._ForeDrawList.index(self.reSizingCircle)
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.resize)

    def resize(self, event):
        if wx.GetMouseState().RightIsDown():
            self.diffCoords = event.Coords - self.preSizeCoords + self.reSizingCircle.WH
            diff = (self.diffCoords[0] + self.diffCoords[1]) / 2
            self.coordMatrix[self.imageIndex, self.reSizingCircleIndex, 3] = diff
            self.reSizingCircle.SetDiameter(diff)
            self.Canvas.Draw()
        else:
            self.BindAllMouseEvents()

    def CircleHover(self, object):
        self.selectionText.SetLabel(self.faceNums[self.Canvas._ForeDrawList.index(object)] + 'is selected')

    def selectionReset(self, object):
        self.selectionText.SetLabel('No Selections')

    def onOpen(self, event):
        print("Open")
        fd = wx.FileDialog(self, style=wx.FD_OPEN)
        fd.ShowModal()
        self.filename = fd.GetPath()
        print("On Open...", self.filename)

        self.openCSVFile(self.filename)

    def onSave(self, event):
        if self.filename == None:
            # In this case perform a "Save As"
            self.onSaveAs(event)
        else:
            self.save(self.filename)

    def onSaveAs(self, event):
        fd = wx.FileDialog(self, message="Save the coordinates as...", style=wx.FD_SAVE,
                           wildcard="Comma separated value (*.csv)|*.csv")
        fd.ShowModal()
        self.filename = fd.GetPath()

        self.save(self.filename)

    def onClose(self, event):
        dlg = wx.MessageDialog(self, message="Would you like to save the coordinates before exiting?",
                               style=wx.YES_NO | wx.YES_DEFAULT)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            print("Saving...")
            self.onSave(event)
        else:
            print("Discarding changes...")

        # Pass this on to the default handler.
        event.Skip()

    def removeArray(self, L, arr):
        ind = 0
        size = len(L)
        while ind != size and not np.array_equal(L[ind][0:2], arr):
            ind += 1
        if ind != size:
            L[ind] = self.nullArray
            return ind
        else:
            raise ValueError('array not found in list.')

    def makeFaceLabel(self, circle):
        if not np.array_equal(circle.XY, self.nullArray[0:2]):
            return self.faceNums[self.Canvas._ForeDrawList.index(circle)]

if __name__ == '__main__':
    app = wx.App(False)
    type_dialog = wx.SingleChoiceDialog(None, message="Options", caption="Select either",
                                        choices=["video", "image directory"])
    choice = type_dialog.ShowModal()
    frame = None
    if choice == wx.ID_OK:
        if type_dialog.GetStringSelection() == "image directory":
            dir_dialog = wx.DirDialog(None, message="Please select a directory that contains images.")
            err = dir_dialog.ShowModal()
            image_dir = '.'
            if (err == wx.ID_OK):
                image_dir = dir_dialog.GetPath()
            else:
                print("Error getting path:", err)

            print("Image Dir", image_dir)
            scale = 1.0

            frame = FaceMapperFrame(None, wx.ID_ANY, "FaceMapper", image_dir, n_points=None, scale=scale, isVideo=False)
        else:
            file_dialog = wx.FileDialog(None, message="Please select a video file.")
            err = file_dialog.ShowModal()
            video = '.'
            if (err == wx.ID_OK):
                video = file_dialog.GetPath()
            else:
                print("Error getting video file")
            print("Video", video)
            scale = 1.0

            frame = FaceMapperFrame(None, wx.ID_ANY, "FaceMapper", video, n_points=None, scale=scale, isVideo=True)

        frame.Show(True)
        app.MainLoop()
