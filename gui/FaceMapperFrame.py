import colorsys
import csv
import glob
import math
import os
import os.path
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
import wx
import wx.lib.agw.cubecolourdialog as ccd
from skimage.measure import compare_ssim as ssim
from wx.lib.floatcanvas import NavCanvas, FloatCanvas, Utilities

# Defines the list of image formats
IMAGE_FORMATS = [".jpg", ".png", ".ppm", ".pgm", ".gif", ".tif", ".tiff", ".jpe"]


class FaceMapperFrame(wx.Frame):
    def __init__(self, parent, id, name, image_dir, n_points=None, scale=1.0, is_video=False):
        wx.Frame.__init__(self, parent, id, name)
        self.smart_dlg = None
        if not is_video:
            smart_or_dumb_dlg = wx.SingleChoiceDialog(None, message='Please select a frames method',
                                                      caption='Frame Method',
                                                      choices=['Smart Frame Analysis',
                                                               'Manual frame-by-frame'])
            if smart_or_dumb_dlg.ShowModal() == wx.ID_OK:
                if smart_or_dumb_dlg.GetSelection() == 0:
                    self.smart_dlg = True

        if is_video:
            output_dlg = wx.DirDialog(None, message='Please select an output directory', defaultPath=image_dir)
            if output_dlg.ShowModal() == wx.ID_OK:
                smart_or_dumb_dlg = wx.SingleChoiceDialog(None, message='Please select a frames method',
                                                          caption='Frame Method',
                                                          choices=['Smart Frame Analysis',
                                                                   'Manual frame # entry'])
                if smart_or_dumb_dlg.ShowModal() == wx.ID_OK:
                    self.image_dir = output_dlg.GetPath()
                    if smart_or_dumb_dlg.GetSelection() == 'Manual frame # entry':
                        frames_dlg = wx.TextEntryDialog(None, message='Please select frames per second', value='5')
                        if frames_dlg.ShowModal() == wx.ID_OK:
                            os.system('ffmpeg -i "{0}" -vf fps=1/{1} "{2}"'.format(image_dir, frames_dlg.GetValue(),
                                                                                   self.image_dir + '/' + os.path.basename(
                                                                                       image_dir)
                                                                                   + '_out%03d.png'))
                    else:
                        self.smart_dlg = True
                        os.system('ffmpeg -i "{0}" -vf fps=1/{1} "{2}"'.format(image_dir, str(1), self.image_dir + '/'
                                                                               + os.path.basename(
                            image_dir) + '_out%03d.png'))
        # ---------------- Basic Data -------------------
        else:
            self.image_dir = image_dir
        self.n_points = n_points
        self.ssim_threshold = .85
        self.image_names = []
        self.imageIndex = 0
        self.current_image = None
        self.image_name = None
        self.prev_image_name = None
        self.compare_image_name = None
        self.scale = scale
        for files in IMAGE_FORMATS:
            self.image_names.extend([os.path.basename(x) for x in glob.glob(self.image_dir + '/*{0}'.format(files))])

        self.image_names.sort()

        self.faceBB = None

        self.filename = None
        self.coords = defaultdict()
        self.faceParts = OrderedDict()
        self.faceLabels = []

        self.firstColors = True

        self.shownLabels = []

        self.selections = OrderedDict()

        # ---------- Colors ----
        self.color_db = wx.ColourDatabase()
        self.color_db.AddColour("Left Eye", wx.TheColourDatabase.Find('Red'))
        self.color_db.AddColour("Right Eye", wx.TheColourDatabase.Find('Orange'))
        self.color_db.AddColour("Mouth", wx.TheColourDatabase.Find('Yellow'))
        self.color_db.AddColour("Jaw", wx.TheColourDatabase.Find('Green'))
        self.color_db.AddColour("Eyebrows", wx.TheColourDatabase.Find('Blue'))
        self.color_db.AddColour("Nose", wx.TheColourDatabase.Find('Violet'))

        self.face_part_values = OrderedDict()
        self.reset_face_part_values()
        self.reset_face_parts()
        self.reset_face_labels()

        self.faceNums = []
        self.reset_face_num()

        self.first_click = True
        self.are_selecting_multiple = False
        self.select_rectangle = None
        self.rectangleStart = None
        self.draggingCircle = None
        self.draggingCircleIndex = None
        self.pre_size_coords = None
        self.resizing_circle = None
        self.resizing_circle_index = None
        self.scrollingCircle = None
        self.pre_drag_coords = None

        self.dotNum = 0
        self.totalDotNum = 0
        self.dotDiam = 1.1
        self.part_counter = 0
        self.colourData = wx.ColourData()

        for facePart in self.faceParts.keys():
            self.totalDotNum += self.faceParts[facePart][1]

        self.coord_keys = [
            'x',
            'y',
            'drawn',
            'diameter',
            'occluded',
            'face_part',
            'guess'
        ]

        self.nullArray = np.array([-1.0, -1.0])
        self.coordMatrix = np.zeros((len(self.image_names), self.totalDotNum, len(self.coord_keys)))
        self.coordMatrix.fill(-1.0)
        self.default_array = np.zeros(len(self.coord_keys))
        self.default_array.fill(-1.0)
        self.assign_part_nums()

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
        n_c = NavCanvas.NavCanvas(self, Debug=0, BackgroundColor="BLACK")
        self.Canvas = n_c.Canvas
        self.Canvas.MinScale = 14
        self.Canvas.MaxScale = 200
        self.imHeight = 500

        # ----------------- Counter Display ------------
        self.rightBox = wx.BoxSizer(wx.VERTICAL)
        self.counterList = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER | wx.LB_SORT,
                                      name='Click on a category to change its color',
                                      choices=self.faceLabels)
        # self.sample_image_canvas = FloatCanvas.FloatCanvas(self.counterBox, Debug=0, BackgroundColor="Black")
        self.sampleImage = wx.Image('sample_face.PNG', wx.BITMAP_TYPE_ANY)
        self.sampleImage = self.sampleImage.Scale((self.sampleImage.GetWidth() / 1.5),
                                                  (self.sampleImage.GetHeight() / 1.5))
        self.sample_image_bitmap = wx.StaticBitmap(self, wx.NewId(), self.sampleImage.ConvertToBitmap())
        # self.sample_image_canvas.AddBitmap(self.sample_image_bitmap, (0,0))
        self.nextButton = wx.Button(self, wx.NewId(), label='Next Part')

        self.emotion_choices = [
            'Happy',
            'Sad',
            'Sleeping',
            'Angry',
            'Afraid',
            'Disgusted',
            'Neutral',
            'Surprised'
        ]

        self.emotionList = wx.ListBox(self, wx.NewId(), style=wx.LB_MULTIPLE, choices=self.emotion_choices)

        self.saveButton = wx.Button(self, wx.NewId(), label='Save and Continue')
        self.labelButton = wx.Button(self, wx.NewId(), label='Show Labels')
        self.re_mirror_button = wx.Button(self, wx.NewId(), label='Re-Mirror')
        self.rightBox.Add(self.counterList, 1, wx.EXPAND)
        self.rightBox.Add(self.sample_image_bitmap, 4, wx.EXPAND)
        self.rightBox.Add(self.nextButton, .5, wx.EXPAND)
        self.rightBox.Add(self.re_mirror_button, .5, wx.EXPAND)
        self.rightBox.Add(self.emotionList, 1, wx.EXPAND)
        self.rightBox.Add(self.labelButton, .5, wx.EXPAND)
        self.rightBox.Add(self.saveButton, .5, wx.EXPAND)

        # ----------------- Window Layout  -------------
        self.mainBox = wx.BoxSizer(wx.HORIZONTAL)
        self.mainBox.Add(self.leftBox, 1, wx.EXPAND)
        self.mainBox.Add(n_c, 3, wx.EXPAND)
        self.mainBox.Add(self.rightBox, 1, wx.EXPAND)

        self.box = wx.BoxSizer(wx.VERTICAL)
        self.box.Add(self.mainBox, 1, wx.EXPAND)

        self.selectionText = wx.StaticText(self, wx.NewId(), label='Nothing currently selected')
        self.box.Add(self.selectionText, 1, wx.EXPAND)

        #self.SetAutoLayout(True)
        self.SetSizer(self.box)
        self.Layout()

        # -------------- Event Handling ----------------
        self.Bind(wx.EVT_LISTBOX, self.on_select, id=self.list.GetId())
        self.Bind(wx.EVT_LISTBOX, self.color_select, id=self.counterList.GetId())
        self.Bind(wx.EVT_BUTTON, self.on_button_save, id=self.saveButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.show_labels, id=self.labelButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.next_part, id=self.nextButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.re_mirror, id=self.re_mirror_button.GetId())
        self.bind_all_mouse_events()
        self.Bind(wx.EVT_MENU, self.on_open, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self.on_save, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.on_save_as, id=wx.ID_SAVEAS)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Canvas.Bind(wx.EVT_KEY_DOWN, self.on_key_press)
        self.Canvas.Bind(wx.EVT_KEY_UP, self.on_key_release)

        # --------- Hotkeys -----
        self.pressedKeys = {
            wx.WXK_CONTROL: False,
            wx.WXK_DELETE: False
        }

        # Create the default facenums list
        self.default_face_nums = []
        for facePart in self.faceParts.keys():
            split = facePart.split()
            abbr = ''
            for i in range(len(split)):
                abbr += split[i].strip()[0]
            for index in range(self.faceParts[facePart][1]):
                self.default_face_nums.append(abbr + str(index + 1))

    #Resets face part values to their defaults
    def reset_face_part_values(self):
        self.face_part_values.clear()
        self.face_part_values["Left Eye"] = 6
        self.face_part_values["Right Eye"] = 6
        self.face_part_values["Mouth"] = 20
        self.face_part_values["Jaw"] = 17
        self.face_part_values["Eyebrows"] = 10
        self.face_part_values["Nose"] = 9

    # Sets the face parts to their default values, with default colors
    def reset_face_parts(self):
        self.faceParts.clear()
        for facePart in list(self.face_part_values.keys()):
            self.faceParts[facePart] = [0, int(self.face_part_values[facePart]), self.color_db.Find(facePart)]

    # Makes face labels based on the face parts
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

    def assign_part_nums(self):
        for image in self.coordMatrix:
            part_num = 0
            counter = 0
            for index, circle in enumerate(image):
                circle[5] = float(part_num)
                val = self.dict_index(self.face_part_values, part_num)
                if index - counter + 1 == val:
                    part_num += 1
                    counter += val

    # Resets mouse events, only tracks left down, right down, and multiple select
    def bind_all_mouse_events(self):
        self.Canvas.Unbind(FloatCanvas.EVT_MOUSEWHEEL)
        self.Canvas.Unbind(FloatCanvas.EVT_LEFT_UP)
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.multi_select)
        self.Canvas.Bind(FloatCanvas.EVT_LEFT_DOWN, self.on_left_down)
        self.Canvas.Bind(FloatCanvas.EVT_RIGHT_DOWN, self.on_right_click)

    # Sets coordinates based on CSV
    def open_csv_file(self, path):

        reader = csv.reader(open(path, "rb"))
        # coords = {}
        # for row in reader:
        #    filename = row[0]
        #    row = row[1:]
        #
        #    if len(row) % 2 != 0:
        #        print("Error Loading File")
        #        raise TypeError("Odd number of values in this row")
        #
        #    points = []
        #    for i in range(0, len(row), 2):
        #        point = (float(row[i]), float(row[i + 1]))
        #        points.append(point)

        #    coords[filename] = points
        # print("CSV File Data: ", coords)
        # self.coords = coords
        for index, row in enumerate(reader):
            filename = row[0]
            if filename == self.image_names[index]:
                points = []
                for i in range(0, len(row), 2):
                    points.append(np.array([float(row[i]), float(row[i + 1])]))
                for ind, point in enumerate(points):
                    self.coordMatrix[index, ind, 0:2] = point
                    self.coordMatrix[index, ind, 2] = 0

    # Triggers on pressing "save and continue"
    def on_button_save(self, event):
        i = self.image_names.index(self.image_name)
        if len(self.image_names) > 1:
            if self.image_name:
                self.prev_image_name = self.image_name
                if self.n_points is not None and len(self.coords[self.image_name]) != self.n_points:
                    print("ERROR: incorrect number of points.")

            self.image_name = self.image_names[i + 1]
            self.imageIndex = self.image_names.index(self.image_name)
            self.mirror_im(event, should_save=True, check_ssim_if_smart=True)
        else:
            print('You\'re Done!')

    # Mirrors coordinates from previous image, if previous image exists
    def mirror_im(self, event, should_save, check_ssim_if_smart):
        cv_prev_image = None
        if self.imageIndex >= 1 and self.circ_is_null(self.coordMatrix[self.imageIndex, 0,]):
            self.coordMatrix[self.imageIndex,] = self.coordMatrix[self.imageIndex - 1,]
            for circle in self.coordMatrix[self.imageIndex,]:
                if not self.circ_is_null(circle):
                    circle[2] = 0

        filename = os.path.join(self.image_dir, self.image_name)
        self.current_image = wx.Image(filename)
        self.list.SetSelection(self.imageIndex)
        self.remove_labels()
        self.display_image(zoom=True)

        if self.smart_dlg and check_ssim_if_smart:
            if self.prev_image_name:
                cv_prev_image = cv2.imread(os.path.join(self.image_dir, self.prev_image_name))
                self.prev_image_name = None
            cv_curr_image = cv2.imread(filename)

            if self.faceBB is not None:
                world_face_bb = Utilities.BBox.fromPoints([abs(self.faceBB[0]), abs(self.faceBB[1])])
                min_x = int(world_face_bb[0][0])
                min_y = int(world_face_bb[0][1])
                max_x = int(world_face_bb[1][0])
                max_y = int(world_face_bb[1][1])
                cropped_prev_cv_image = cv_prev_image[min_y:max_y, min_x:max_x]
                cropped_curr_cv_image = cv_curr_image[min_y:max_y, min_x:max_x]
                ssim_index = ssim(cropped_prev_cv_image, cropped_curr_cv_image, multichannel=True)
                if ssim_index > self.ssim_threshold:
                    self.on_button_save(event=None)
                else:
                    self.prev_image_name = self.image_name

        if should_save:
            self.on_save(event)

    def re_mirror(self, event):
        if self.imageIndex >= 1:
            self.recur_mirror(self.imageIndex)
            self.Canvas.Draw()

    def recur_mirror(self, index):
        if not self.no_dots(index):
            dl = self.draw_list(index)
            for ind in dl.keys():
                # circ_ind = self.find_circle_coord_ind(dl[ind][0:2], ind=index)
                prev_circ = self.coordMatrix[index - 1, ind]
                if not self.circ_is_null(prev_circ):
                    self.set_coords(self.find_circle(self.curr_image_points()[ind][0:2]), prev_circ[0:2],
                                    im_ind=index)
            self.recur_mirror(index + 1)

    # Save coordinates to a csv file
    def save(self, path):
        writer = csv.writer(open(path, 'w'))
        first_row = [' ', ' ']
        for faceNum in self.default_face_nums:
            first_row.append(faceNum)
            first_row.append('')
            first_row.append('guess')
        writer.writerow(first_row)
        for index, image in enumerate(self.image_names):
            if not self.circ_array_is_null(self.curr_image_points(index)):
                row = [image, ', '.join([self.emotion_choices[i] for i in self.emotionList.GetSelections()])]
                for point in self.curr_image_points(index):
                    if not self.circ_is_null(point) or self.is_occluded(point):
                        row.append(point[0])
                        row.append(point[1])
                        row.append(point[self.coord_keys.index('guess')])
                writer.writerow(row)

    # Triggers on event selection
    def on_select(self, event):
        self.image_name = event.GetString()
        self.imageIndex = self.image_names.index(self.image_name)
        self.select_im(index=self.imageIndex)

    def select_im(self, index):
        for i in range(len(self.image_names)):
            if i != index:
                for circle in self.coordMatrix[i,]:
                    if not self.circ_is_null(circle):
                        circle[2] = 0
        self.mirror_im(event=None, should_save=False, check_ssim_if_smart=False)

    # Triggers on selecting a face part
    def color_select(self, event):
        list_dlg = wx.SingleChoiceDialog(self, message="Choose an option", caption="list option",
                                         choices=['Reset Num', 'Choose Color'])
        if list_dlg.ShowModal() == wx.ID_OK:
            if list_dlg.GetSelection() == 'Choose Color':
                num = event.GetInt()
                name = list(self.faceParts.keys())[num]
                self.colourData.SetColour(self.faceParts[name][2])
                color_dlg = ccd.CubeColourDialog(self, self.colourData)
                if color_dlg.ShowModal() == wx.ID_OK:
                    self.firstColors = False
                    self.colourData = color_dlg.GetColourData()
                    self.color_db.AddColour(name, self.colourData.GetColour())
                    self.display_image(zoom=False)
            else:
                self.reset_face_part_values()
                self.reset_face_parts()
                self.reset_face_num()
                self.make_face_label_list()

    # Displays image
    def display_image(self, zoom):
        if zoom:
            self.Canvas.InitAll()
            if self.current_image:
                im = self.current_image.Copy()
                # self.imHeight = im.GetHeight()
                bm = im.ConvertToBitmap()
                self.Canvas.AddScaledBitmap(bm, XY=(0, 0), Height=self.imHeight, Position='tl')
                self.dotDiam = self.imHeight / 100

        self.reset_face_parts()
        dl = self.draw_list()
        dl_key_list = sorted(dl.keys())
        for index in dl_key_list:
            coord_circle = dl[index]
            if coord_circle[2] == 0:
                if index >= 1:
                    self.dotDiam = dl[dl_key_list[dl_key_list.index(index) - 1]][3]
                if coord_circle[3] == -1.0:
                    coord_circle[3] = self.dotDiam
                diam = coord_circle[3]

                if index < len(self.Canvas._ForeDrawList):
                    circle = self.find_circle(coord_circle[0:2])
                    circle.XY = coord_circle[0:2]
                    circle.SetDiameter(diam)
                else:
                    circ = FloatCanvas.Circle(XY=coord_circle[0:2], Diameter=diam, LineWidth=.5, LineColor='Red',
                                              FillStyle='Transparent', InForeground=True)
                    circ = self.Canvas.AddObject(circ)
                    circ.Bind(FloatCanvas.EVT_FC_LEFT_DOWN, self.circle_left_down)
                    circ.Bind(FloatCanvas.EVT_FC_RIGHT_DOWN, self.circle_resize)
                    circ.Bind(FloatCanvas.EVT_FC_ENTER_OBJECT, self.circle_hover)
                    circ.Bind(FloatCanvas.EVT_FC_LEAVE_OBJECT, self.selection_reset)
                    circ.Bind(FloatCanvas.EVT_FC_LEFT_DCLICK, self.mark_guess)
                    circ.Bind(FloatCanvas.EVT_FC_RIGHT_DCLICK, self.mark_not_guess)
                coord_circle[2] = 1

        self.make_face_label_list()

        self.Canvas.Draw()
        if len(self.Canvas._ForeDrawList) >= 1:
            self.faceBB = Utilities.BBox.fromPoints([circ.XY for circ in self.Canvas._ForeDrawList])
        if zoom:
            if len(self.Canvas._ForeDrawList) >= 1 and self.distance(p1=self.faceBB[0],
                                                                     p2=self.faceBB[1]) >= self.imHeight / 10:
                self.Canvas.ZoomToBB(self.faceBB)
            else:
                self.Canvas.ZoomToBB()

    # Triggers on left mouse click
    def on_left_down(self, event):
        if not self.pressedKeys[wx.WXK_CONTROL]:
            self.add_coords(event.Coords)

    # Adds location of event to coordinate matrix
    def add_coords(self, coords):
        if len(self.Canvas._ForeDrawList) < self.totalDotNum:
            free_pos = self.find_first_free_pos()
            self.coordMatrix[self.imageIndex, free_pos, 0:3] = np.append(coords, np.array([0.0]))
            self.coordMatrix[self.imageIndex, free_pos, self.coord_keys.index('guess')] = 0.0
            self.display_image(zoom=False)

    # Triggers when clicking inside of a circle
    def circle_left_down(self, circle):
        if wx.GetKeyState(wx.WXK_CONTROL):
            self.select_part(self.curr_image_points()[self.find_circle_coord_ind(circle.XY)][5])
        else:
            self.draggingCircle = circle
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
                self.Canvas.Draw()
                self.pre_drag_coords = event.Coords
            else:
                self.bind_all_mouse_events()

        else:
            self.are_selecting_multiple = False
            self.on_right_click(event=None)
            if is_left_down:
                self.set_coords(self.draggingCircle, event.Coords)
                self.draggingCircle.SetLineStyle('Dot')
                self.Canvas.Draw()
            else:
                self.draggingCircle.SetLineStyle('Solid')
                self.bind_all_mouse_events()

    # Triggers when right-clicking a circle
    def circle_resize(self, circle):
        self.pre_size_coords = circle.XY
        self.resizing_circle = circle
        self.resizing_circle_index = self.Canvas._ForeDrawList.index(self.resizing_circle)
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.resize)

    # Redraws circle at size based on dragging mouse
    def resize(self, event):
        is_right_down = wx.GetMouseState().RightIsDown()
        curr_y_coord = event.Coords[1]
        if is_right_down:
            diff_coords = (curr_y_coord - self.pre_size_coords[1]) * .1 + abs(self.resizing_circle.WH)
            diff = (diff_coords[0] + diff_coords[1])
            self.coordMatrix[self.imageIndex, self.resizing_circle_index, 3] = diff
            self.resizing_circle.SetDiameter(diff)
            self.pre_size_coords = event.Coords
            self.Canvas.Draw()

        else:
            self.bind_all_mouse_events()

    # Triggers when hovering over circle
    def circle_hover(self, circle):
        if not self.are_selecting_multiple:
            self.selectionText.SetLabel('Hovering Over ' + str(self.make_face_label(circle)) + '\n' + 'Guess Status: {0}'.format(self.curr_image_points()[self.find_circle_coord_ind(circle.XY)][self.coord_keys.index('guess')]))
            self.Canvas.Bind(FloatCanvas.EVT_MOUSEWHEEL, self.on_cmd_scroll)
            self.scrollingCircle = circle

    def display_selections(self):
        if len(self.selections) >= 1:
            selection_text = 'Current selections: ' + self.make_face_label(list(self.selections.keys())[0])
            for i in range(1, len(self.selections.keys())):
                selection_text += ', ' + self.make_face_label(list(self.selections.keys())[i])
            selection_dict = {circle: self.curr_image_points()[self.find_circle_coord_ind(circle.XY)][self.coord_keys.index('guess')] for circle in self.selections.keys()}
            guess_text = '\n Guess status: '
            for circle in list(selection_dict.keys()):
                guess_text += '{0}: {1}'.format(self.make_face_label(circle), selection_dict[circle])
            self.selectionText.SetLabel(selection_text + guess_text)
        else:
            self.selectionText.SetLabel('No Selections')

    # Allows one to change the color of a circle set by pressing CTRL and scrolling
    def on_cmd_scroll(self, event):
        if self.pressedKeys[wx.WXK_CONTROL]:
            coord_circ = self.curr_image_points()[self.find_circle_coord_ind(self.scrollingCircle.XY)]
            part = list(self.faceParts.keys())[int(coord_circ[5])]
            curr_color = self.faceParts[part][2]
            hsv_color = colorsys.rgb_to_hsv(curr_color.Red(), curr_color.Green(), curr_color.Blue())
            if event.GetWheelRotation() > 0:
                delta = .1
            else:
                delta = -.1
            new_color = colorsys.hsv_to_rgb(hsv_color[0] + delta, hsv_color[1], hsv_color[2])
            self.color_db.AddColour(part, wx.Colour(new_color[0], new_color[1], new_color[2], alpha=wx.ALPHA_OPAQUE))
            self.display_image(zoom=False)

    def on_key_press(self, event):
        self.pressedKeys[event.GetKeyCode()] = True
        if event.GetKeyCode() == wx.WXK_DELETE:
            self.del_selections()

    def select_part(self, index):
        self.clear_all_selections()
        for circle in self.curr_image_points():
            if not self.circ_is_null(circle) and circle[5] == index:
                self.add_to_selections(self.find_circle(circle[0:2]))

    def on_key_release(self, event):
        self.pressedKeys[event.GetKeyCode()] = False

    def multi_select(self, event):
        is_left_down = wx.GetMouseState().LeftIsDown()
        if self.pressedKeys[wx.WXK_CONTROL] and is_left_down:
            if self.rectangleStart is None:
                self.rectangleStart = event.Coords
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
                    LineColor='Purple',
                    LineStyle='Dot',
                    LineWidth=1,
                    FillColor='Gold',
                    FillStyle='BiDiagonalHatch'
                ))
            self.Canvas.Draw()
            self.Canvas.Bind(FloatCanvas.EVT_LEFT_UP, self.fin_select)

    def del_selections(self):
        for circle in list(self.selections.keys()):
            index = self.find_circle_coord_ind(circle.XY)
            self.curr_image_points()[index] = self.default_array
        self.Canvas.RemoveObjects(self.selections)
        self.clear_all_selections()
        self.assign_part_nums()
        self.display_image(zoom=False)

    def fin_select(self, event):
        # self.selections.clear()
        if self.select_rectangle:
            for circle in self.Canvas._ForeDrawList:
                if self.check_if_contained(circle):
                    self.add_to_selections(circle)
            self.Canvas.RemoveObject(self.select_rectangle, ResetBB=False)
            self.select_rectangle = None
            self.rectangleStart = None
        self.bind_all_mouse_events()
        self.display_selections()
        self.Canvas.Draw()

    def add_to_selections(self, circle):
        self.selections[circle] = circle.XY
        circle.SetLineStyle('Dot')

    def check_if_contained(self, circle):
        c_x, c_y = circle.XY
        if self.select_rectangle:
            r_x, r_y = self.select_rectangle.XY
            r_x2, r_y2 = self.select_rectangle.XY + self.select_rectangle.WH
            if self.select_rectangle:
                if r_x2 < r_x:
                    r_x2, r_x = r_x, r_x2
                if r_y2 < r_y:
                    r_y2, r_y = r_y, r_y2
            return r_x <= c_x <= r_x2 and r_y <= c_y <= r_y2
        else:
            return False

    def on_right_click(self, event):
        if wx.GetKeyState(wx.WXK_CONTROL):
            self.pre_rotate_mouse_coords = event.Coords
            self.pre_rotate_coords = []
            for circle in self.selections:
                self.half = self.find_bb_half(self.faceBB)
                self.pre_rotate_coords.append(circle.XY - self.half)
            self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.rotate)
        else:  # clears selections
            if event:
                self.are_selecting_multiple = False
            if not self.are_selecting_multiple:
                self.clear_all_selections()

    def mark_guess(self, object):
        self.curr_image_points(self.find_circle_coord_ind(object.XY))[self.coord_keys.index('guess')] = 1.0

    def mark_not_guess(self, object):
        self.curr_image_points(self.find_circle_coord_ind(object.XY))[self.coord_keys.index('guess')] = 0.0

    def clear_all_selections(self):
        for circle in self.Canvas._ForeDrawList:
            circle.SetLineStyle('Solid')
        if self.select_rectangle:
            self.rectangleStart = None
            self.Canvas.RemoveObject(self.select_rectangle)
            self.select_rectangle = None
        self.selections.clear()
        self.display_selections()
        self.Canvas.Draw()

    def rotate(self, event):
        is_right_down = wx.GetMouseState().RightIsDown()
        if is_right_down:
            for index, circle in enumerate(self.selections):
                diff_coords = event.Coords - self.pre_rotate_mouse_coords
                diff_coord_x = diff_coords[0] / 100
                diff_coord_y = diff_coords[1] / 100
                coord_matrix = np.array([[self.pre_rotate_coords[index][0]], [self.pre_rotate_coords[index][1]]])
                new_coords = self.rotate_mat(diff_coord_y, coord_matrix)
                self.set_coords(circle, np.array([new_coords[0, 0] +
                                                  diff_coord_x * self.pre_rotate_coords[index][0],
                                                  new_coords[1, 0] + diff_coord_x *
                                                  self.pre_rotate_coords[index][1]]) + self.half)
            self.Canvas.Draw()
        else:
            self.bind_all_mouse_events()

    # Resets selection text
    def selection_reset(self, object):
        is_left_down = wx.GetMouseState().LeftIsDown()
        is_right_down = wx.GetMouseState().RightIsDown()
        if not (is_right_down or is_left_down or self.are_selecting_multiple):
            self.are_selecting_multiple = False
            self.bind_all_mouse_events()
            self.selections.clear()
            self.selectionText.SetLabel('No Selections')

    # Shows face numbers for each circle
    def show_labels(self, event):
        if len(self.shownLabels) == 0:
            part_dict = self.part_dict()
            for index in part_dict.keys():
                face_part_circles = part_dict[index]
                face_part_bb = Utilities.BBox.fromPoints([circle[0:2] for circle in face_part_circles])
                face_part_bb_center = self.find_bb_half(face_part_bb)
                for circle in face_part_circles:
                    circle = self.find_circle(circle[0:2])
                    coord_diff = circle.XY - face_part_bb_center
                    mag_coord_diff = np.sqrt(np.square(coord_diff[0]) + np.square(coord_diff[1]))
                    unit_coord_diff = np.array(
                        [np.divide(coord_diff[0], mag_coord_diff), np.divide(coord_diff[1], mag_coord_diff)])
                    theta = self.find_theta(unit_coord_diff[1], unit_coord_diff[0])
                    w_h_in_dir_of_diff = self.rotate_mat(theta, 2 * circle.WH.transpose())
                    t = FloatCanvas.ScaledText(String=self.make_face_label(circle),
                                               XY=(circle.XY + w_h_in_dir_of_diff.transpose()),
                                               Size=circle.WH[0] * .85,
                                               Color=circle.LineColor)
                    t = self.Canvas.AddObject(t)
                    self.shownLabels.append(t)
                    # l = FloatCanvas.Line(np.array([circle.XY, face_part_bb_center]))
                    # l = self.Canvas.AddObject(l)
            self.Canvas.Draw()
            self.labelButton.SetLabel('Hide Labels')
        else:
            self.remove_labels()

    def find_theta(self, y, x):
        theta = math.atan(np.divide(y, x))
        if y < 0 < x:
            theta += math.pi / 2
        elif x < 0 and y < 0:
            theta += math.pi
        elif x < 0 < y:
            theta += 1.5 * math.pi
        return theta

    def make_face_label_list(self):
        dl = self.draw_list()
        for index in sorted(dl.keys()):
            circle = dl[index]
            face_part = self.dict_index(self.faceParts, int(circle[5]))
            face_part[0] += 1
            circle = self.find_circle(dl[index][0:2])
            if circle is not None:
                circle.SetColor(face_part[2].GetAsString())
                circle.SetLineStyle('Solid')
        self.counterList.Clear()
        self.reset_face_labels()
        self.counterList.Set(self.faceLabels)


    # Triggers on opening of CSV file
    def on_open(self, event):
        print("Open")
        fd = wx.FileDialog(self, style=wx.FD_OPEN)
        fd.ShowModal()
        self.filename = fd.GetPath()
        print("On Open...", self.filename)

        self.open_csv_file(self.filename)

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
    def remove_array(self, l, arr):
        ind = 0
        size = len(l)
        while ind != size and not np.array_equal(l[ind][0:2], arr):
            ind += 1
        if ind != size:
            l[ind] = self.nullArray
            return ind
        else:
            raise ValueError('array not found in list.')

    # Returns face number for a given circle
    def make_face_label(self, circle):
        if not np.array_equal(circle.XY, self.nullArray[0:2]):
            return str(self.faceNums[self.find_circle_coord_ind(circle.XY)])

    def remove_labels(self):
        self.Canvas.RemoveObjects(self.shownLabels)
        self.shownLabels.clear()
        self.Canvas.Draw()
        self.labelButton.SetLabel('Show Labels')

    def next_part(self, event):
        ind = self.find_first_free_pos() - 1
        part_index = self.curr_image_points()[ind, 5]
        next_circ = self.curr_image_points()[ind + 1]
        while next_circ[5] == part_index:
            next_circ[4] = 1.0
            ind += 1
            next_circ = self.curr_image_points()[ind + 1]

    def set_coords(self, circle, x_y, im_ind=None):
        if im_ind is None:
            im_ind = self.imageIndex
        index = self.find_circle_coord_ind(circle.XY)
        circle.XY = x_y
        self.coordMatrix[im_ind, index, 0:2] = x_y

    def find_circle(self, x_y):
        # x_y_list = [circle.XY for circle in self.Canvas._ForeDrawList]
        for index, circle in enumerate(self.Canvas._ForeDrawList):
            if np.array_equal(circle.XY, x_y):
                return self.Canvas._ForeDrawList[index]
        raise ValueError

    def find_circle_coord_ind(self, x_y, ind=None):
        if ind is None:
            ind = self.imageIndex
        x_y_list = [tuple(circle[0:2]) for circle in self.coordMatrix[ind,] if not self.circ_is_null(circle)]
        for index, circle in enumerate(x_y_list):
            if np.array_equal(circle, x_y):
                return index
        raise ValueError

    def find_first_free_pos(self):
        for i in range(self.totalDotNum):
            circ = self.coordMatrix[self.imageIndex, i, ]
            if self.circ_is_null(circ) and circ[4] != 1:
                return i

    def draw_list(self, ind=None):
        if ind is None:
            ind = self.imageIndex
        return {index: circ for index, circ in enumerate(self.curr_image_points(ind)) if
                (not self.circ_is_null(circ) and not self.is_occluded(circ))}

    def part_dict(self):
        return {k: v for k, v in {
            index: [circle for circle in self.curr_image_points() if
                    (circle[5] == index and not self.circ_is_null(circle))]
            for index in range(len(self.faceParts.keys()))}.items() if len(v) > 1}

    @staticmethod
    def find_bb_half(bbox):
        min_x = bbox[0][0]
        max_x = bbox[1][0]
        min_y = bbox[0][1]
        max_y = bbox[1][1]
        return np.array([max_x + min_x, max_y + min_y]) / 2

    @staticmethod
    def rotate_mat(theta, mat):
        rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
        return np.dot(rotation_matrix, mat)

    @staticmethod
    def is_occluded(circ_array):
        return circ_array[4] == 1.0

    def is_null(self, circ_array, property):
        num = circ_array[self.coord_keys.index(property)]
        bool = num == -1.0
        return bool

    def curr_image_points(self, ind=None):
        if ind is None:
            ind = self.imageIndex
        return self.coordMatrix[ind, ]

    @staticmethod
    def distance(p1, p2):
        return math.sqrt(math.pow(p2[1] - p1[1], 2) + math.pow(p2[0] - p1[0], 2))

    @staticmethod
    def dict_index(diction, index):
        return diction[list(diction.keys())[int(index)]]

    def circ_is_null(self, circle):
        return np.array_equal(circle[0:2], self.nullArray)

    def circ_array_is_null(self, arr):
        for circle in arr:
            if not self.circ_is_null(circle):
                return False
        return True

    def no_dots(self, index):
        for circle in self.coordMatrix[index, ]:
            if not self.circ_is_null(circle):
                return False
        return True

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

            frame = FaceMapperFrame(None, wx.ID_ANY, "FaceMapper", image_dir,
                                    n_points=None, scale=scale, is_video=False)
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
