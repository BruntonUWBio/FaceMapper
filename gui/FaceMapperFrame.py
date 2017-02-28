import colorsys
import csv
import glob
import os
import os.path
from collections import defaultdict, OrderedDict

import numpy as np
import wx
import wx.lib.agw.cubecolourdialog as ccd
from wx.lib.floatcanvas import NavCanvas, FloatCanvas, GUIMode

# Defines the list of image formats
IMAGE_FORMATS = [".jpg", ".png", ".ppm", ".pgm", ".gif", ".tif", ".tiff", ".jpe"]


class FaceMapperFrame(wx.Frame):
    def __init__(self, parent, id, name, image_dir, n_points=None, scale=1.0, is_video=False):
        wx.Frame.__init__(self, parent, id, name)

        if is_video:
            output_dlg = wx.DirDialog(None, message='Please select an output directory', defaultPath=image_dir)
            if output_dlg.ShowModal() == wx.ID_OK:
                self.image_dir = output_dlg.GetPath()
                os.system("ffmpeg -i {0} -vf fps=1/5 {1}".format(image_dir, self.image_dir + '/out%02d.png'))

        # ---------------- Basic Data -------------------
        else:
            self.image_dir = image_dir
        self.n_points = n_points
        self.image_names = []
        self.imageIndex = 0
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

        self.shownLabels = []

        self.selections = OrderedDict()

        # ---------- Colors ----
        self.colordb = wx.ColourDatabase()
        self.colordb.AddColour("Left Eye", wx.TheColourDatabase.Find('Red'))
        self.colordb.AddColour("Right Eye", wx.TheColourDatabase.Find('Orange'))
        self.colordb.AddColour("Mouth", wx.TheColourDatabase.Find('Yellow'))
        self.colordb.AddColour("Jaw", wx.TheColourDatabase.Find('Green'))
        self.colordb.AddColour("Eyebrows", wx.TheColourDatabase.Find('Blue'))
        self.colordb.AddColour("Nose", wx.TheColourDatabase.Find('Violet'))

        self.reset_face_parts()
        self.reset_face_labels()

        self.faceNums = []
        self.reset_face_num()

        self.first_click = True
        self.are_selecting_multiple = False
        self.select_rectangle = None
        self.pre_drag_coords = None


        self.dotNum = 0
        self.partCounter = 1
        self.totalDotNum = 0
        self.dotDiam = 1.1
        self.colourData = wx.ColourData()

        for facePart in self.faceParts.keys():
            self.totalDotNum += self.faceParts[facePart][1]

        self.coord_keys = [
            'x',
            'y',
            'drawn',
            'diameter',
            'visible',
        ]

        self.nullArray = np.array([-1.0, -1.0, -1.0, -1.0])
        self.coordMatrix = np.zeros((len(self.image_names), self.totalDotNum, 4))
        self.coordMatrix.fill(-1.0)

        # ------------- Other Components ----------------
        self.CreateStatusBar()
        # ------------------- Menu ----------------------
        file_menu = wx.Menu()

        # File Menu
        file_menu.Append(wx.ID_ABOUT, wx.EmptyString)
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_OPEN, wx.EmptyString)
        file_menu.Append(wx.ID_SAVE, wx.EmptyString)
        file_menu.Append(wx.ID_SAVEAS, wx.EmptyString)
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, wx.EmptyString)

        # Creating the menubar.
        menu_bar = wx.MenuBar()
        menu_bar.Append(file_menu, "&File")  # Adding the "file_menu" to the MenuBar
        self.SetMenuBar(menu_bar)  # Adding the MenuBar to the Frame content.

        # ----------------- Image List ------------------
        self.leftBox = wx.BoxSizer(wx.VERTICAL)
        self.list = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER, name='List of Files',
                               choices=sorted(self.image_names))
        self.leftBox.Add(self.list, 1, wx.EXPAND)

        # ----------------- Image Display ---------------
        # n_c = SelectCanvas.SelectCanvas(self, Debug=0, BackgroundColor="Black")
        n_c = NavCanvas.NavCanvas(self, Debug=0, BackgroundColor="BLACK")
        self.Canvas = n_c.Canvas
        self.Canvas.MinScale = 14
        self.Canvas.MaxScale = 200
        self.imHeight = 500

        # ----------------- Counter Display ------------
        self.counterBox = wx.BoxSizer(wx.VERTICAL)
        self.counterList = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER | wx.LB_SORT,
                                      name='Click on a category to change its color',
                                      choices=self.faceLabels)
        self.saveButton = wx.Button(self, wx.NewId(), label='Press to Save and Continue')
        self.labelButton = wx.Button(self, wx.NewId(), label='Show Labels')
        self.counterBox.Add(self.counterList, 1, wx.EXPAND)
        self.counterBox.Add(self.labelButton, .5, wx.EXPAND)
        self.counterBox.Add(self.saveButton, .5, wx.EXPAND)

        # ----------------- Window Layout  -------------
        self.mainBox = wx.BoxSizer(wx.HORIZONTAL)
        self.mainBox.Add(self.leftBox, 1, wx.EXPAND)
        self.mainBox.Add(n_c, 3, wx.EXPAND)
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
        self.Bind(wx.EVT_LISTBOX, self.color_select, id=self.counterList.GetId())
        self.Bind(wx.EVT_BUTTON, self.onbuttonsave, id=self.saveButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.show_labels, id=self.labelButton.GetId())
        self.BindAllMouseEvents()
        self.Bind(wx.EVT_MENU, self.on_open, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self.on_save, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.on_save_as, id=wx.ID_SAVEAS)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Canvas.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
        self.Canvas.Bind(wx.EVT_KEY_UP, self.onKeyRelease)

        # --------- Hotkeys -----
        self.pressedKeys = {
            wx.WXK_CONTROL: False
        }

    # Sets the face parts to their default values, with default colors
    def reset_face_parts(self):
        self.faceParts["Left Eye"] = [0, 6, self.colordb.Find("Left Eye")]
        self.faceParts["Right Eye"] = [0, 6, self.colordb.Find("Right Eye")]
        self.faceParts["Mouth"] = [0, 20, self.colordb.Find("Mouth")]
        self.faceParts["Jaw"] = [0, 17, self.colordb.Find("Jaw")]
        self.faceParts["Eyebrows"] = [0, 10, self.colordb.Find("Eyebrows")]
        self.faceParts["Nose"] = [0, 9, self.colordb.Find("Nose")]

    # Makes face labels based on the faceparts
    def reset_face_labels(self):
        self.faceLabels = []
        part_index = 1
        for facePart in self.faceParts.keys():
            self.faceLabels.append("{0}. {1}: {2} out of {3}".format(part_index, facePart, self.faceParts[facePart][0],
                                                                     self.faceParts[facePart][1]))
            part_index += 1

    # Makes the numbers based on face parts
    def reset_face_num(self):
        self.faceNums.clear()
        for facePart in self.faceParts.keys():
            split = facePart.split()
            abbr = ''
            for i in range(len(split)):
                abbr += split[i].strip()[0]
            for index in range(self.faceParts[facePart][1]):
                self.faceNums.append(abbr + str(index + 1))

    # Resets mouse events, only tracks left down, right down, and multiple select
    def BindAllMouseEvents(self):
        self.Canvas.Unbind(FloatCanvas.EVT_MOUSEWHEEL)
        self.Canvas.Unbind(FloatCanvas.EVT_LEFT_UP)
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.multiSelect)
        self.Canvas.Bind(FloatCanvas.EVT_LEFT_DOWN, self.on_left_down)
        self.Canvas.Bind(FloatCanvas.EVT_RIGHT_DOWN, self.clear_selections)

    # Sets coordinates based on CSV
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

    # Triggers on pressing "save and continue"
    def onbuttonsave(self, event):
        i = self.image_names.index(self.image_name)
        if len(self.image_names) > 1:
            if self.image_name:
                self.prev_image_name = self.image_name
                if self.n_points != None and len(self.coords[self.image_name]) != self.n_points:
                    print("ERROR: incorrect number of points.")

            self.image_name = self.image_names[i + 1]
            self.imageIndex = self.image_names.index(self.image_name)
            self.mirrorim(event, shouldsave=True)
        else:
            print('You\'re Done!')

    # Mirrors coordinates from previous image, if previous image exists
    def mirrorim(self, event, shouldsave):
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
        self.remove_labels()
        self.DisplayImage(Zoom=True)
        if shouldsave:
            self.on_save(event)

    # Save coordiantes to a csv file
    def save(self, path):
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

    # Triggers on event selection
    def onSelect(self, event):

        self.image_name = event.GetString()
        self.imageIndex = self.image_names.index(self.image_name)
        for i in range(len(self.image_names)):
            if i != self.imageIndex:
                for circle in self.coordMatrix[i,]:
                    if not np.array_equal(circle, self.nullArray):
                        circle[2] = 0
        self.mirrorim(event, shouldsave=False)

    # Triggers on selecting a face part
    def color_select(self, event):
        num = event.GetInt()
        name = list(self.faceParts.keys())[num]
        self.colourData.SetColour(self.faceParts[name][2])
        color_dlg = ccd.CubeColourDialog(self, self.colourData)
        if color_dlg.ShowModal() == wx.ID_OK:
            self.firstColors = False
            self.colourData = color_dlg.GetColourData()
            self.colordb.AddColour(name, self.colourData.GetColour())
            #self.faceParts[name][2] = self.colourData.GetColour().GetAsString()
            self.DisplayImage(Zoom=False)

    # Displays image
    def DisplayImage(self, Zoom):
        if Zoom:
            self.Canvas.InitAll()
            if self.current_image:
                im = self.current_image.Copy()
                # self.imHeight = im.GetHeight()
                bm = im.ConvertToBitmap()
                self.Canvas.AddScaledBitmap(bm, XY=(0, 0), Height=self.imHeight, Position='tl')
                self.dotDiam = self.imHeight / 100
                self.Canvas.ZoomToBB()

        self.reset_face_parts()
        part_counter = 0
        for index, circle in enumerate(self.coordMatrix[self.imageIndex,]):
            if not (np.array_equal(circle, self.nullArray)) and circle[2] == 0:
                if circle[3] == -1.0:
                    circle[3] = self.dotDiam
                diam = circle[3]

                if index < len(self.Canvas._ForeDrawList):
                    self.Canvas._ForeDrawList[index].XY = circle[0:2]
                    self.Canvas._ForeDrawList[index].SetDiameter(diam)
                else:
                    circ = FloatCanvas.Circle(XY=circle[0:2], Diameter=diam, LineWidth=.5, LineColor='Red',
                                           FillStyle='Transparent', InForeground=True)

                    circ = self.Canvas.AddObject(circ)
                    circ.Bind(FloatCanvas.EVT_FC_LEFT_DOWN, self.circle_left_down)
                    circ.Bind(FloatCanvas.EVT_FC_RIGHT_DOWN, self.circle_resize)
                    circ.Bind(FloatCanvas.EVT_FC_ENTER_OBJECT, self.circle_hover)
                    circ.Bind(FloatCanvas.EVT_FC_LEAVE_OBJECT, self.selection_reset)
                circle[2] = 1

        for circle in self.Canvas._ForeDrawList:
            face_part = self.faceParts[list(self.faceParts.keys())[part_counter]]
            face_part[0] += 1
            circle.SetColor(face_part[2].GetAsString())
            circle.SetLineStyle('Solid')
            if face_part[0] == face_part[1]:
                part_counter += 1
            self.make_face_label(circle)

        self.counterList.Clear()
        self.reset_face_labels()
        self.counterList.Set(self.faceLabels)
        self.Canvas.Draw()

    # Triggers on left mouse click
    def on_left_down(self, event):
        if not self.pressedKeys[wx.WXK_CONTROL]:
            self.add_coords(event)

    # Adds location of event to coordinate matrix
    def add_coords(self, event):
        if len(self.Canvas._ForeDrawList) < self.totalDotNum:
            self.coordMatrix[self.imageIndex, len(self.Canvas._ForeDrawList), 0:3] = np.append(event.Coords,
                                                                                               np.array([0.0]))
            self.DisplayImage(Zoom=False)

    # Triggers when clicking inside of a circle
    def circle_left_down(self, object):
        self.draggingCircle = object
        self.draggingCircleIndex = self.Canvas._ForeDrawList.index(self.draggingCircle)
        self.pre_drag_coords = self.draggingCircle.XY
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.drag)
        self.Canvas.Draw()

    # Redraws circle at location of mouse while dragging
    def drag(self, event):
        is_left_down = wx.GetMouseState().LeftIsDown()
        if self.draggingCircle in self.selections:
            if is_left_down:
                for circle in self.selections:
                    self.set_coords(circle, circle.XY + event.Coords - self.pre_drag_coords)
                self.pre_drag_coords = event.Coords
                self.Canvas.Draw()
            else:
                self.BindAllMouseEvents()

        else:
            self.are_selecting_multiple = False
            self.clear_selections(event=None)
            if is_left_down:
                self.set_coords(self.draggingCircle, event.Coords)
                self.draggingCircle.SetLineStyle('Dot')
                self.Canvas.Draw()
            else:
                self.draggingCircle.SetLineStyle('Solid')
                self.BindAllMouseEvents()

    # Triggers when right-clicking a circle
    def circle_resize(self, object):
        self.pre_size_coords = object.XY
        self.resizing_circle = object
        self.resizing_circle_index = self.Canvas._ForeDrawList.index(self.resizing_circle)
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.resize)

    # Redraws circle at size based on dragging mouse
    def resize(self, event):
        is_right_down = wx.GetMouseState().RightIsDown()
        curr_coords = event.Coords
        if is_right_down:
            diffCoords = (abs(curr_coords) - abs(self.pre_size_coords)) * .25 + abs(self.resizing_circle.WH)
            diff = (diffCoords[0] + diffCoords[1])
            self.coordMatrix[self.imageIndex, self.resizing_circle_index, 3] = diff
            self.resizing_circle.SetDiameter(diff)
            self.Canvas.Draw()
        else:
            self.BindAllMouseEvents()

    # Triggers when hovering over circle
    def circle_hover(self, object):
        if not self.are_selecting_multiple:
            self.selectionText.SetLabel('Hovering Over ' + self.make_face_label(object))
            self.Canvas.Bind(FloatCanvas.EVT_MOUSEWHEEL, self.on_cmd_scroll)
            self.scrollingCircle = object

    def displaySelections(self):
        if len(self.selections) >= 1:
            selection_text = ('Current selections: ') + self.make_face_label(list(self.selections.keys())[0])
            for i in range(1, len(self.selections.keys())):
                selection_text += ', ' + self.make_face_label(list(self.selections.keys())[i])
            self.selectionText.SetLabel(selection_text)
        else:
            self.selectionText.SetLabel('No Selections')

    # Allows one to change the color of a circle set by pressing CTRL and scrolling
    def on_cmd_scroll(self, event):
        if self.pressedKeys[wx.WXK_CONTROL]:
            part = self.findFacePart(self.scrollingCircle)
            curr_color = self.faceParts[part][2]
            hsv_color = colorsys.rgb_to_hsv(curr_color.Red(), curr_color.Green(), curr_color.Blue())
            if event.GetWheelRotation() > 0:
                delta = .1
            else:
                delta = -.1
            new_color = colorsys.hsv_to_rgb(hsv_color[0] + delta, hsv_color[1], hsv_color[2])
            self.colordb.AddColour(self.findFacePart(self.scrollingCircle),
                                   wx.Colour(new_color[0], new_color[1], new_color[2], alpha=wx.ALPHA_OPAQUE))
            self.DisplayImage(Zoom=False)

    def onKeyPress(self, event):
        self.pressedKeys[event.GetKeyCode()] = True

    def onKeyRelease(self, event):
        self.pressedKeys[event.GetKeyCode()] = False

    def multiSelect(self, event):
        is_left_down = wx.GetMouseState().LeftIsDown()
        if self.pressedKeys[wx.WXK_CONTROL] and is_left_down:
            if not self.are_selecting_multiple:
                self.are_selecting_multiple = True
                self.rectangleStart = event.Coords

            if self.select_rectangle:
                self.Canvas.RemoveObject(self.select_rectangle, ResetBB=False)
                self.select_rectangle = None
            self.select_rectangle = \
                self.Canvas.AddObject(FloatCanvas.Rectangle(
                    XY=self.rectangleStart,
                    WH=event.Coords - self.rectangleStart,
                    LineColor='Red',
                    LineStyle='Dot',
                    LineWidth=1,
                    FillColor='Gray',
                    FillStyle='Transparent'
                ))
            self.Canvas.Draw()
            self.Canvas.Bind(FloatCanvas.EVT_LEFT_UP, self.fin_select)

    def fin_select(self, event):
        # self.selections.clear()
        if self.select_rectangle:
            for circle in self.Canvas._ForeDrawList:
                if self.check_if_contained(circle):
                    self.selections[circle] = circle.XY
                    circle.SetLineStyle('Dot')
            self.Canvas.RemoveObject(self.select_rectangle, ResetBB=False)
            self.select_rectangle = None
        self.BindAllMouseEvents()
        self.displaySelections()
        self.Canvas.Draw()

    def check_if_contained(self, circle):
        c_x = abs(circle.XY[0])
        c_y = abs(circle.XY[1])
        if self.select_rectangle:
            r_x = abs(self.select_rectangle.XY[0])
            r_y = abs(self.select_rectangle.XY[1])
            r_w = abs(self.select_rectangle.WH[0])
            r_h = abs(self.select_rectangle.WH[1])
            return c_x >= r_x and c_x <= r_x + r_w and c_y >= r_y and c_y <= r_y + r_h
        else:
            return False

    def clear_selections(self, event):
        if event and event.EventType == 10232:
            self.are_selecting_multiple = False
        if self.are_selecting_multiple == False:
            for circle in self.Canvas._ForeDrawList:
                circle.SetLineStyle('Solid')
            if self.select_rectangle:
                self.rectangleStart = None
                self.Canvas.RemoveObject(self.select_rectangle)
                self.select_rectangle = None
            self.selections.clear()
            self.displaySelections()
            self.Canvas.Draw()

    # Resets selection text
    def selection_reset(self, object):
        is_left_down = wx.GetMouseState().LeftIsDown()
        is_right_down = wx.GetMouseState().RightIsDown()
        if not (is_right_down or is_left_down or self.are_selecting_multiple):
            self.are_selecting_multiple = False
            self.BindAllMouseEvents()
            self.selections.clear()
            self.selectionText.SetLabel('No Selections')

    # Shows face numbers for each circle
    def show_labels(self, event):
        if len(self.shownLabels) == 0:
            for index, circle in enumerate(self.Canvas._ForeDrawList):
                if not np.array_equal(circle, self.nullArray):
                    T = FloatCanvas.ScaledText(String=self.faceNums[index],
                                               XY=(circle.XY[0] - circle.WH[0], circle.XY[1] - circle.WH[1]),
                                               Size=circle.WH[0] * .85,
                                               Color=circle.LineColor)
                    T = self.Canvas.AddObject(T)
                    self.shownLabels.append(T)
            self.Canvas.Draw()
            self.labelButton.SetLabel('Hide Labels')
        else:
            self.remove_labels()

    # Triggers on opening of CSV file
    def on_open(self, event):
        print("Open")
        fd = wx.FileDialog(self, style=wx.FD_OPEN)
        fd.ShowModal()
        self.filename = fd.GetPath()
        print("On Open...", self.filename)

        self.openCSVFile(self.filename)

    # Triggers on save action
    def on_save(self, event):
        if not self.filename:
            # In this case perform a "Save As"
            self.on_save_as(event)
        else:
            self.save(self.filename)

    # Triggers on save as action
    def on_save_as(self, event):
        fd = wx.FileDialog(self, message="Save the coordinates as...", style=wx.FD_SAVE,
                           wildcard="Comma separated value (*.csv)|*.csv")
        fd.ShowModal()
        self.filename = fd.GetPath()

        self.save(self.filename)

    # Triggers on close action
    def on_close(self, event):
        dlg = wx.MessageDialog(self, message="Would you like to save the coordinates before exiting?",
                               style=wx.YES_NO | wx.YES_DEFAULT)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            print("Saving...")
            self.on_save(event)
        else:
            print("Discarding changes...")

        # Pass this on to the default handler.
        event.Skip()

    # Changes an array to standard null array
    def remove_array(self, L, arr):
        ind = 0
        size = len(L)
        while ind != size and not np.array_equal(L[ind][0:2], arr):
            ind += 1
        if ind != size:
            L[ind] = self.nullArray
            return ind
        else:
            raise ValueError('array not found in list.')

    # Returns face number for a given circle
    def make_face_label(self, circle):
        if not np.array_equal(circle.XY, self.nullArray[0:2]):
            return self.faceNums[self.Canvas._ForeDrawList.index(circle)]

    def remove_labels(self):
        self.Canvas.RemoveObjects(self.shownLabels)
        self.shownLabels.clear()
        self.Canvas.Draw()
        self.labelButton.SetLabel('Show Labels')

    def findFacePart(self, circle):
        circ_index = self.Canvas._ForeDrawList.index(circle)
        for part in list(self.faceParts.keys()):
            if self.faceParts[part][1] < circ_index:
                circ_index -= self.faceParts[part][1]
            else:
                return part

    def set_coords(self, circle, XY):
        circle.XY = XY
        index = self.Canvas._ForeDrawList.index(circle)
        self.coordMatrix[self.imageIndex, index, 0:2] = XY


class SelectorMode(GUIMode.GUIBase):
    def __init__(self):
        GUIMode.GUIBase.__init__(self)

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
            if err == wx.ID_OK:
                image_dir = dir_dialog.GetPath()
            else:
                print("Error getting path:", err)

            print("Image Dir", image_dir)
            scale = 1.0

            frame = FaceMapperFrame(None, wx.ID_ANY, "FaceMapper", image_dir, n_points=None, scale=scale, is_video=False)
        else:
            file_dialog = wx.FileDialog(None, message="Please select a video file.")
            err = file_dialog.ShowModal()
            video = '.'
            if err == wx.ID_OK:
                video = file_dialog.GetPath()
            else:
                print("Error getting video file")
            print("Video", video)
            scale = 1.0

            frame = FaceMapperFrame(None, wx.ID_ANY, "FaceMapper", video, n_points=None, scale=scale, is_video=True)

        frame.Show(True)
        app.MainLoop()
