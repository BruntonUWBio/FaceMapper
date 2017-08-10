import colorsys
import csv
import glob
import math
import os
import subprocess
import sys
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import wx
import wx.lib.agw.cubecolourdialog as ccd
from FaceMapperModel import FaceMapperModel
from skimage.measure import compare_ssim as ssim
from wx.lib import colourdb
from wx.lib.floatcanvas import NavCanvas, Utilities, FloatCanvas

IMAGE_FORMATS = [".jpg", ".png", ".ppm", ".pgm", ".gif", ".tif", ".tiff", ".jpe"]


class FaceMapperFrame(wx.Frame):
    def __init__(self, parent, id, name, image_dir, n_points=None, scale=1.0, is_video=False, csv_path=None):
        """
        Default constructor.
        :param parent: Inherited from wx.Frame.
        :param id: Inherited from wx.Frame.
        :param name: Name of window.
        :param image_dir: Parameter indicating either location of image sequence or video.
        :param n_points: Maximum number of points for annotation.
        :param scale: Unused.
        :param is_video: If image_dir is the location of a video.
        :param csv_path: Path to csv to store annotations
        """
        wx.Frame.__init__(self, parent, id, name)
        self.smart_dlg = None
        self.fps_frac = '30'
        if not csv_path:
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
                                subprocess.Popen(
                                    'ffmpeg -i "{0}" -vf fps={1} "{2}"'.format(image_dir, frames_dlg.GetValue(),
                                                                               self.image_dir + '/' + os.path.basename(
                                                                                   image_dir)
                                                                               + '_out%04d.png'), shell=True).wait()
                        else:
                            self.smart_dlg = True
                            subprocess.Popen(
                                'ffmpeg -i "{0}" -vf fps={1} "{2}"'.format(image_dir, self.fps_frac,
                                                                           self.image_dir + '/'
                                                                           + os.path.basename(
                                                                               image_dir) + '_out%04d.png'),
                                shell=True).wait()
            # ---------------- Basic Data -------------------
            else:
                self.image_dir = image_dir
        else:
            self.image_dir = os.path.dirname(csv_path)
        self.n_points = n_points
        self.ssim_threshold = .9
        self.image_names = []
        self.imageIndex = 0
        self.current_image = None
        self.image_name = None
        self.prev_image_name = None
        self.compare_image_name = None
        self.paused = None
        self.scale = scale
        for files in IMAGE_FORMATS:
            self.image_names.extend([os.path.basename(x) for x in glob.glob(self.image_dir + '/*{0}'.format(files))])

        self.image_names.sort()

        imgDlg = wx.MessageDialog(None, message="Please wait, images processing...", style=wx.CENTER)
        imgDlg.Show(show=True)

        if self.smart_dlg:
            self.image_reads = {image: cv2.imread(os.path.join(self.image_dir, image)) for image in self.image_names}

        imgDlg.Destroy()

        self.faceBB = None

        self.filename = None

        self.faceLabels = []

        self.firstColors = True

        self.shownLabels = []

        self.selections = OrderedDict()
        self.emotion_dict = defaultdict()

        # ---------- Colors ----
        # Set default colors

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
        self.totalDotNum = 68
        self.dotDiam = 1.1
        self.part_counter = 0
        self.colourData = wx.ColourData()

        #for facePart in self.model.faceParts:
        #    self.totalDotNum += self.model.faceParts[facePart][0]

        self.model = FaceMapperModel(self.totalDotNum)
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
        self.make_face_labels()
        frame_dir_name = os.path.dirname(sys.argv[0])
        # self.sample_image_canvas = FloatCanvas.FloatCanvas(self.counterBox, Debug=0, BackgroundColor="Black")
        self.sampleImage = wx.Image(os.path.join(frame_dir_name, 'sample_face.png'), wx.BITMAP_TYPE_ANY)
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
            'Surprised',
            'None (Face not Visible/ Emotion Unclear)'
        ]

        self.emotionList = wx.ListBox(self, wx.NewId(), style=wx.LB_MULTIPLE, choices=self.emotion_choices)

        self.saveButton = wx.Button(self, wx.NewId(), label='Save and Continue')
        self.labelButton = wx.Button(self, wx.NewId(), label='Show Labels')
        self.re_mirror_button = wx.Button(self, wx.NewId(), label='Re-Mirror')
        self.play_button = wx.Button(self, wx.NewId(), label='Play')
        self.pause_button = wx.Button(self, wx.NewId(), label='Pause')
        self.leftBox.Add(self.emotionList, 1, wx.EXPAND)
        self.rightBox.Add(self.counterList, 1, wx.EXPAND)
        self.rightBox.Add(self.sample_image_bitmap, 4, wx.EXPAND)
        self.rightBox.Add(self.nextButton, .5, wx.EXPAND)
        self.rightBox.Add(self.re_mirror_button, .5, wx.EXPAND)
        self.rightBox.Add(self.labelButton, .5, wx.EXPAND)
        self.rightBox.Add(self.saveButton, .5, wx.EXPAND)
        self.rightBox.Add(self.play_button, .5, wx.EXPAND)
        self.rightBox.Add(self.pause_button, .5, wx.EXPAND)

        # ----------------- Window Layout  -------------
        self.mainBox = wx.BoxSizer()
        self.mainBox.Add(self.leftBox, 1, wx.EXPAND)
        self.mainBox.Add(n_c, 3, wx.EXPAND)
        self.mainBox.Add(self.rightBox, 1, wx.EXPAND)

        self.box = wx.BoxSizer(wx.VERTICAL)
        self.box.Add(self.mainBox, 5, wx.EXPAND)

        self.selectionText = wx.StaticText(self, wx.NewId(), label='Nothing currently selected')
        self.box.Add(self.selectionText, 1, wx.EXPAND)

        # self.SetAutoLayout(True)
        self.SetSizer(self.box)
        self.Layout()

        # -------------- Event Handling ----------------
        self.Bind(wx.EVT_LISTBOX, self.on_select, id=self.list.GetId())
        self.Bind(wx.EVT_LISTBOX, self.color_select, id=self.counterList.GetId())
        self.Bind(wx.EVT_LISTBOX, self.emotion_select, id=self.emotionList.GetId())
        self.Bind(wx.EVT_BUTTON, self.on_button_save, id=self.saveButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.show_labels, id=self.labelButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.next_part, id=self.nextButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.re_mirror, id=self.re_mirror_button.GetId())
        self.Bind(wx.EVT_BUTTON, self.play, id=self.play_button.GetId())
        self.Bind(wx.EVT_BUTTON, self.pause, id=self.pause_button.GetId())
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
        for facePart in self.model.faceParts:
            split = facePart.split()
            abbr = ''
            for i in range(len(split)):
                abbr += split[i].strip()[0]
            for index in range(self.model.faceParts[facePart][0]):
                self.default_face_nums.append(abbr + str(index + 1))

        # mirror if opening
        if csv_path:
            self.open_csv_file(csv_path)

    # Resets mouse events, only tracks left down, right down, and multiple select
    def bind_all_mouse_events(self):
        self.Canvas.Unbind(FloatCanvas.EVT_MOUSEWHEEL)
        self.Canvas.Unbind(FloatCanvas.EVT_LEFT_UP)
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.multi_select)
        self.Canvas.Bind(FloatCanvas.EVT_LEFT_DOWN, self.on_left_down)
        self.Canvas.Bind(FloatCanvas.EVT_RIGHT_DOWN, self.on_right_click)
        self.Canvas.Bind(FloatCanvas.EVT_RIGHT_DCLICK, self.mark_multi_guess)

        # Triggers on left mouse click

    def on_left_down(self, event):
        if not self.pressedKeys[wx.WXK_CONTROL]:
            self.add_coords(event.Coords)

    # Adds location of event to coordinate matrix
    def add_coords(self, coords: np.ndarray):
        length = self.model.index_first_none(self.imageIndex)
        if length < self.totalDotNum:
            self.model.add_point(self.imageIndex, length, coords)
            self.display_image(zoom=False)

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

    def rotate(self, event):
        is_right_down = wx.GetMouseState().RightIsDown()
        if is_right_down:
            for index, circle in enumerate(self.selections):
                diff_coords = event.Coords - self.pre_rotate_mouse_coords
                diff_coord_x = diff_coords[0] / 100
                diff_coord_y = diff_coords[1] / 100
                coord_matrix = np.array(
                    [[self.pre_rotate_coords[index][0]], [self.pre_rotate_coords[index][1]]])
                new_coords = self.rotate_mat(diff_coord_y, coord_matrix)
                self.set_coords(circle, np.array([new_coords[0, 0] +
                                                  diff_coord_x * self.pre_rotate_coords[index][0],
                                                  new_coords[1, 0] + diff_coord_x *
                                                  self.pre_rotate_coords[index][1]]) + self.half)
            self.Canvas.Draw()
        else:
            self.bind_all_mouse_events()

    def set_coords(self, circle, x_y, im_ind=None):
        if im_ind is None:
            im_ind = self.imageIndex
        self.model.set_coords(circle, x_y, im_ind)

    def mark_multi_guess(self, event):
        for circle in self.selections:
            self.model.mark_guess(circle, self.imageIndex)
        self.display_image()


    # Triggers on event selection
    def on_select(self, event):
        self.image_name = event.GetString()
        self.select_im(index=self.image_names.index(self.image_name))

    def select_im(self, index):
        self.update_index(index)
        self.mirror_im(event=None, should_save=False, check_ssim_if_smart=False)
        #else:
        #    self.display_image(zoom=True)

    def update_index(self, index):
        self.imageIndex = index
        self.image_name = self.image_names[self.imageIndex]

        # Mirrors coordinates from previous image, if previous image exists

    def mirror_im(self, event, should_save, check_ssim_if_smart, show=True):
        self.model.mirror_im(self.imageIndex)
        filename = os.path.join(self.image_dir, self.image_name)
        self.current_image = wx.Image(filename)
        self.list.SetSelection(self.imageIndex)
        self.remove_labels()

        if self.smart_dlg and check_ssim_if_smart:
            if self.prev_image_name:
                cv_prev_image = self.image_reads[self.prev_image_name]
                self.prev_image_name = None

                cv_curr_image = self.image_reads[self.image_name]

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
                        return True
                    else:
                        self.prev_image_name = self.image_name
                        self.display_image(zoom=True)
                        return False
            else:
                return True
        else:
            self.display_image(zoom=False, re_show=True)

    def re_mirror(self, event=None):
        if self.imageIndex >= 1:
            index = self.imageIndex
            while True:
                if self.model.draw_list(index):
                    self.iter_mirror(index)
                    index += 1
                else:
                    break
            self.Canvas.Draw()

    def iter_mirror(self, index):
        dl = self.model.coord_list(index)
        for ind in range(len(dl)):
            # circ_ind = self.find_circle_coord_ind(dl[ind][0:2], ind=index)
            prev_circ = self.model.coord_list(index - 1)[ind]
            if prev_circ:
                self.set_coords(self.model.draw_list(self.imageIndex), prev_circ[0:2],
                                im_ind=index)


    def next_part(self, event):
        self.model.next_part(self.imageIndex)

    def remove_labels(self):
        self.Canvas.RemoveObjects(self.shownLabels)
        self.shownLabels.clear()
        self.Canvas.Draw()
        self.labelButton.SetLabel('Show Labels')

    # Triggers on selecting a face part
    def color_select(self, event):
        choices = ['Reset Num', 'Choose Color']
        list_dlg = wx.SingleChoiceDialog(self, message="Choose an option", caption="list option",
                                         choices=choices)
        if list_dlg.ShowModal() == wx.ID_OK:
            select_string = choices[list_dlg.GetSelection()]
            if select_string == 'Choose Color':
                name = event.GetString()
                self.colourData.SetColour(self.model.curr_face_part_vals(name)[2])
                color_dlg = ccd.CubeColourDialog(self, self.colourData)
                if color_dlg.ShowModal() == wx.ID_OK:
                    self.firstColors = False
                    self.colourData = color_dlg.GetColourData()
                    self.model.color_db.AddColour(name, self.colourData.GetColour())
                    self.display_image(zoom=False)
            elif select_string == 'Reset Num':
                self.model.reset_default_face_parts()
                self.model.reset_face_num()
                self.remove_occluded()
                self.make_face_label_list()

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

    def fin_select(self, event):
        # self.selections.clear()
        if self.select_rectangle:
            for circle in self.model.not_none_draw_list(self.imageIndex):
                if self.check_if_contained(circle):
                    self.add_to_selections(circle)
            self.Canvas.RemoveObject(self.select_rectangle, ResetBB=False)
            self.select_rectangle = None
            self.rectangleStart = None
        self.bind_all_mouse_events()
        self.display_selections()
        self.Canvas.Draw()

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

    def add_to_selections(self, circle):
        self.selections[circle] = circle.XY
        circle.SetLineStyle('Dot')

    def display_selections(self):
        if len(self.selections) >= 1:
            selection_text = 'Current selections: ' + self.model.make_face_label(list(self.selections.keys())[0],
                                                                                 self.imageIndex)
            for i in range(1, len(self.selections.keys())):
                selection_text += ', ' + self.model.make_face_label(list(self.selections.keys())[i], self.imageIndex)
            self.selectionText.SetLabel(selection_text)
        else:
            self.selectionText.SetLabel('No Selections')

    def remove_occluded(self):
        self.model.remove_occluded(self.imageIndex)

    def make_face_label_list(self):
        self.model.make_face_label_list(self.imageIndex)
        self.make_face_labels()

    def make_face_labels(self):
        self.counterList.Clear()
        self.counterList.Set(self.model.faceLabels)

    def emotion_select(self, event=None, index=None):
        if index is None:
            index = self.imageIndex
        try:
            self.emotion_dict[self.image_names[index]] = self.emotionList.GetSelections()
            if self.emotionList.GetSelections() is None:
                self.emotion_dict[self.image_names[index]] = ['None selected']
        except:
            pass

    # Triggers on pressing "save and continue"
    def on_button_save(self, event):
        while self.mirror_im(event, should_save=True, check_ssim_if_smart=True):
            self.emotion_select()
            self.emotion_select(index=self.imageIndex + 1)
            i = self.image_names.index(self.image_name)
            if len(self.image_names) > i + 1:
                if self.image_name:
                    self.prev_image_name = self.image_name
                self.image_name = self.image_names[i + 1]
                self.imageIndex = self.image_names.index(self.image_name)

            else:
                print('You\'re Done!')
                break

                # Triggers on opening of CSV file

    def on_open(self, event):
        print("Open")
        fd = wx.FileDialog(self, style=wx.FD_OPEN)
        fd.ShowModal()
        self.filename = fd.GetPath()
        print("On Open...", self.filename)

        self.open_csv_file(self.filename)

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
        for image in self.image_names:
            print('Removing ' + image)
            if os.path.exists(os.path.join(self.image_dir, image)):
                os.remove(os.path.join(self.image_dir, image))
        print('Done!')
        sys.exit(0)

    def open_csv_file(self, path):
        """
        Sets coordinates based on csv.

        :param path: Path to csv.
        :return: None
        """
        with open(path, 'rt') as csvfile:
            file_names = []
            reader = csv.reader(csvfile)
            numRows = 0
            for index, row in enumerate(reader):
                filename = row[0]
                if filename in self.image_names:
                    file_names.append(filename)
                    file_index = self.image_names.index(filename)
                    points = []
                    self.emotion_dict[filename] = row[1].split(',')
                    for i in range(2, len(row), 3):
                        points.append(np.array([float(row[i]), float(row[i + 1])]))
                    for ind, point in enumerate(points):
                        self.model.add_point(self.imageIndex, point)

                numRows += 1
            self.select_im(self.image_names.index(file_names.pop()))

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

    # Save coordinates to a csv file
    def save(self, path):
        writer = csv.writer(open(path if path[len(path) - 4:len(path)] == '.csv' else path + '.csv', 'w'))
        first_row = [' ', ' ']
        for faceNum in self.default_face_nums:
            first_row.append(faceNum)
            first_row.append('')
            first_row.append('guess')
        writer.writerow(first_row)
        for index, image in enumerate(self.image_names):
            row = [image, ', '.join([self.emotion_choices[i] for i in self.emotion_dict[image]])]
            for point in self.model.coord_list(index):
                if point:
                    row.append(point[0])
                    row.append(point[1])
                    row.append(point[self.model.coord_keys.index('guess')])
                else:
                    row.append('')
            writer.writerow(row)

    def play(self, event):
        self.paused = True
        while self.paused:
            self.emotion_select()
            self.emotion_select(index=self.imageIndex + 1)
            i = self.image_names.index(self.image_name)
            if len(self.image_names) > i + 1:
                if self.image_name:
                    self.prev_image_name = self.image_name
                self.image_name = self.image_names[i + 1]
                self.imageIndex = self.image_names.index(self.image_name)
                self.mirror_im(event, should_save=True, check_ssim_if_smart=False)
            else:
                print('You\'re Done!')
                break
            # self.display_image(zoom=False)
            wx.Yield()

    def pause(self, event):
        self.paused = False

    def on_key_press(self, event):
        self.pressedKeys[event.GetKeyCode()] = True
        if event.GetKeyCode() == wx.WXK_DELETE:
            self.del_selections()

    def on_key_release(self, event):
        self.pressedKeys[event.GetKeyCode()] = False


    def del_selections(self):
        for circle in list(self.selections.keys()):
            self.model.delete_circle(self.imageIndex, circle)
        self.Canvas.RemoveObjects(self.selections)
        self.clear_all_selections()

        #TODO: Why is this here
        self.assign_part_nums()

        self.display_image(zoom=False)

    def clear_all_selections(self):
        for circle in self.model.not_none_draw_list(self.imageIndex):
            circle.SetLineStyle('Solid')
        if self.select_rectangle:
            self.rectangleStart = None
            self.Canvas.RemoveObject(self.select_rectangle)
            self.select_rectangle = None
        self.selections.clear()
        self.display_selections()
        self.Canvas.Draw()

    # Assign part numbers to current entry in coordinate matrix
    def assign_part_nums(self):
        part_num = 0
        counter = 0
        for index, circle in enumerate(self.model.coord_list(self.imageIndex)):
            if circle:
                circle[self.model.coord_keys.index('face_part')] = float(part_num)
                val = self.model.face_part_values[self.model.face_part_list[part_num]]
                if index - counter + 1 == val:
                    part_num += 1
                    counter += val


    # TODO: Fix
    def show_labels(self, event):
        if len(self.shownLabels) == 0:
            part_dict = self.part_dict()
            for index in part_dict.keys():
                face_part_circles = part_dict[index]
                face_part_bb = Utilities.BBox.fromPoints([circle[0:2] for circle in face_part_circles])
                face_part_bb_center = self.find_bb_half(face_part_bb)
                for circle in face_part_circles:
                    circle = self.model.draw_list(self.imageIndex)[self.model.coord_list(self.imageIndex).index(circle)]
                    coord_diff = circle.XY - face_part_bb_center
                    mag_coord_diff = np.sqrt(np.square(coord_diff[0]) + np.square(coord_diff[1]))
                    unit_coord_diff = np.array(
                        [np.divide(coord_diff[0], mag_coord_diff), np.divide(coord_diff[1], mag_coord_diff)])
                    theta = math.atan2(unit_coord_diff[1], unit_coord_diff[0])
                    w = np.array([circle.WH[0], 0.0])
                    w_h_in_dir_of_diff = self.rotate_mat(theta, 2 * w.transpose())
                    t = FloatCanvas.ScaledText(String=self.model.make_face_label(circle, self.imageIndex),
                                               XY=(circle.XY + w_h_in_dir_of_diff.transpose()),
                                               Size=circle.WH[0] * .85,
                                               Color=circle.LineColor)
                    t = self.Canvas.AddObject(t)
                    self.shownLabels.append(t)
            self.Canvas.Draw()
            self.labelButton.SetLabel('Hide Labels')
        else:
            self.remove_labels()

    # TODO: Fix
    def part_dict(self):
        return {k: v for k, v in {
            index: [circle for circle in self.model.coord_list(self.imageIndex) if
                    (circle[self.model.coord_keys.index('guess')] == index and circle)]
            for index in range(len(self.model.faceParts.keys()))}.items() if len(v) > 1}

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


    def display_image(self, zoom, re_show = False):
        if re_show:
            self.Canvas.InitAll()
            if self.current_image:
                im = self.current_image.Copy()
                self.imHeight = im.GetHeight()
                self.imWidth = im.GetWidth()
                bm = im.ConvertToBitmap()
                self.Canvas.AddScaledBitmap(bm, XY=(0, 0), Height= self.Canvas.GetSize().GetHeight(), Position='cc')
                self.dotDiam = self.imHeight/100
        self.model.zero_face_parts()
        dl = self.model.not_none_draw_list(self.imageIndex)
        cl = self.model.not_none_coord_list(self.imageIndex)
        for index, circle in enumerate(dl):
            coord_circle = cl[index]
            if not coord_circle[self.model.coord_keys.index('drawn')]:
                part = self.model.face_part_list[coord_circle[self.model.coord_keys.index('face_part')]]
                circle.SetColor(self.model.curr_face_part_vals(part)[2].GetAsString())
                circle.SetDiameter(coord_circle[self.model.coord_keys.index('diameter')])
                if coord_circle[self.model.coord_keys.index('guess')] == 1:
                    circle.SetFillStyle('CrossHatch')
                circ = self.Canvas.AddObject(circle)
                circ.Bind(FloatCanvas.EVT_FC_LEFT_DOWN, self.circle_left_down)
                circ.Bind(FloatCanvas.EVT_FC_RIGHT_DOWN, self.circle_resize)
                circ.Bind(FloatCanvas.EVT_FC_ENTER_OBJECT, self.circle_hover)
                circ.Bind(FloatCanvas.EVT_FC_LEAVE_OBJECT, self.selection_reset)
                circ.Bind(FloatCanvas.EVT_FC_LEFT_DCLICK, self.model.mark_guess)
                coord_circle[self.model.coord_keys.index('drawn')] = 1

        self.make_face_label_list()

        self.Canvas.Draw()
        if len(dl) >= 1:
            self.faceBB = Utilities.BBox.fromPoints([circ.XY for circ in dl])

        if zoom:
            if len(dl) >= 1 and distance(p1=self.faceBB[0], p2=self.faceBB[1]) >= self.imHeight / 5:
                self.Canvas.ZoomToBB(self.faceBB)

    # Triggers when clicking inside of a circle
    def circle_left_down(self, circle):
        if wx.GetKeyState(wx.WXK_CONTROL):
            self.select_part(self.model.coord_list(self.imageIndex)[self.model.draw_list(self.imageIndex).index(circle)][5])
        else:
            self.draggingCircle = circle
            self.draggingCircleIndex = self.model.draw_list(self.imageIndex).index(self.draggingCircle)
            self.pre_drag_coords = self.draggingCircle.XY
            self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.drag)
            self.Canvas.Draw()

    def select_part(self, index):
        self.clear_all_selections()
        for circInd, circle in self.model.coord_list(self.imageIndex):
            if circle and circle[self.model.coord_keys.index('face_part')] == index:
                self.add_to_selections(self.model.draw_list(self.imageIndex)[circInd])

    # Triggers when right-clicking a circle
    def circle_resize(self, circle):
        self.pre_size_coords = circle.XY
        self.resizing_circle = circle
        self.resizing_circle_index = self.model.draw_list(self.imageIndex).index(self.resizing_circle)
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.resize)

        # Redraws circle at size based on dragging mouse

    def resize(self, event):
        is_right_down = wx.GetMouseState().RightIsDown()
        curr_y_coord = event.Coords[1]
        if is_right_down:
            if wx.GetKeyState(wx.WXK_CONTROL):  # If CTRL key is pressed, resize all circles in selections
                for circle in self.selections:
                    self.gen_resize(circle, self.resizing_circle, curr_y_coord)
            else:
                self.gen_resize(self.resizing_circle, self.resizing_circle, curr_y_coord)
            self.pre_size_coords = event.Coords
            self.Canvas.Draw()

        else:
            self.bind_all_mouse_events()

    def gen_resize(self, circle, hit_circle, new_coord):
        pre_size_coords = hit_circle.XY
        pre_circ_coords = circle.XY
        diff_coords = (new_coord - pre_size_coords[1]) * .1 + abs(circle.WH)
        diff = diff_coords[0] + diff_coords[1]
        if diff > 0:
            self.model.coord_list(self.imageIndex)[self.model.draw_list(self.imageIndex).index(hit_circle)][self.model.coord_keys.index('diameter')] = diff
            circle.SetDiameter(diff)

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
            else:
                self.draggingCircle.SetLineStyle('Solid')
                self.bind_all_mouse_events()

            self.Canvas.Draw()
# Triggers when hovering over circle
    def circle_hover(self, circle):
        if not self.are_selecting_multiple:
            self.selectionText.SetLabel('Hovering Over ' + str(self.model.make_face_label(circle, self.imageIndex)))
            self.Canvas.Bind(FloatCanvas.EVT_MOUSEWHEEL, self.on_cmd_scroll)
            self.scrollingCircle = circle

            # Allows one to change the color of a circle set by pressing CTRL and scrolling

    def on_cmd_scroll(self, event):
        if self.pressedKeys[wx.WXK_CONTROL]:
            coord_circ = self.model.coord_list(self.imageIndex)[self.model.draw_list(self.imageIndex).index(self.scrollingCircle)]
            colorList = wx.lib.colourdb.getColourList()
            part = list(self.model.face_part_list)[int(coord_circ[self.model.coord_keys.index('face_part')])]
            curr_color = self.model.curr_face_part_vals(part)[2]
            hsv_color = colorsys.rgb_to_hsv(curr_color.Red(), curr_color.Green(), curr_color.Blue())
            if event.GetWheelRotation() > 0:
                delta = .05
            else:
                delta = -.05
            new_color = colorsys.hsv_to_rgb(hsv_color[0] + delta, hsv_color[1], hsv_color[2])
            self.model.color_db.AddColour(part, wx.Colour(new_color[0], new_color[1], new_color[2], alpha=wx.ALPHA_OPAQUE))
            self.model.curr_face_part_vals(part)[2] = wx.Colour(new_color[0], new_color[1], new_color[2], alpha=wx.ALPHA_OPAQUE)
            self.display_image(zoom=False)


    # Resets selection text
    def selection_reset(self, object):
        is_left_down = wx.GetMouseState().LeftIsDown()
        is_right_down = wx.GetMouseState().RightIsDown()
        if not (is_right_down or is_left_down or self.are_selecting_multiple):
            self.are_selecting_multiple = False
            self.bind_all_mouse_events()
            self.selections.clear()
            self.selectionText.SetLabel('No Selections')


def distance(p1, p2):
    """
    Returns the mathematical distance between two point arrays containing their x coordinate in their first index
    and their y coordinate in their second index

    :param p1: First point
    :param p2: Second point
    :return: Distance between p1 and p2
    """
    return math.sqrt(math.pow(p2[1] - p1[1], 2) + math.pow(p2[0] - p1[0], 2))


if __name__ == '__main__':
    args = sys.argv
    csv_file_path = None
    if 'csv' in args:
        csv_file_path = args[args.index('csv') + 1]
    app = wx.App(False)
    scale = 1.0
    frame = None
    if not csv_file_path:
        type_dialog = wx.SingleChoiceDialog(None, message="Options", caption="Select either",
                                            choices=["video", "image directory"])
        choice = type_dialog.ShowModal()

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

                frame = FaceMapperFrame(None, wx.ID_ANY, "FaceMapper", image_dir, scale=scale, is_video=False)
            else:
                file_dialog = wx.FileDialog(None, message="Please select a video file.")
                err = file_dialog.ShowModal()
                video = ''
                if err == wx.ID_OK:
                    video = file_dialog.GetPath()
                else:
                    print("Error getting video file")
                print("Video: " + video)

                frame = FaceMapperFrame(None, wx.ID_ANY, "FaceMapper", video, scale=scale, is_video=True)
    else:
        frame = FaceMapperFrame(None, wx.ID_ANY, "FaceMapper", None, scale=scale, is_video=True, csv_path=csv_file_path)
    frame.Show(True)
    app.MainLoop()
