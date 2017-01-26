# FaceMapper License
#
# Copyright (c) Gautham Velchuru
# Inspired by and shares code with David S. Bolme's EyePicker, with the following copyright:
# !/usr/bin/env python
# PyVision License
#
# Copyright (c) 2006-2008 David S. Bolme
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import copy
import csv
import os
import os.path
import random
from collections import defaultdict

import wx
from wx.lib.floatcanvas import NavCanvas, FloatCanvas

IMAGE_FORMATS = [".JPG", ".PNG", ".PPM", ".PGM", ".GIF", ".TIF", ".TIFF", ]


class FaceMapperFrame(wx.Frame):
    def __init__(self, parent, id, name, image_dir, n_points=None, randomize=False, scale=1.0):
        wx.Frame.__init__(self, parent, id, name)

        # ---------------- Basic Data -------------------
        self.image_dir = image_dir
        self.n_points = n_points
        self.image_names = []
        self.current_image = None
        self.image_name = None
        self.prev_image_name = None
        self.scale = scale
        for name in os.listdir(image_dir):
            for format in IMAGE_FORMATS:
                if name.upper().endswith(format):
                    self.image_names.append(name)
        if randomize:
            random.shuffle(self.image_names)
        self.image_names.sort()
        self.filename = None
        self.coords = defaultdict()
        self.circles = {}

        self.leftEyeMax = 10
        self.rightEyeMax = 10
        self.mouthMax = 10
        self.leftEyeNum = 0
        self.rightEyeNum = 0
        self.mouthNum = 0
        self.faceLabels = []
        self.faceLabels.append("Left Eye: {0} out of {1}".format(self.leftEyeNum, self.leftEyeMax))
        self.faceLabels.append("Right Eye: {0} out of {1}".format(self.rightEyeNum, self.rightEyeMax))
        self.faceLabels.append("Mouth: {0} out of {1}".format(self.mouthNum, self.mouthMax))
        self.faceNums = {"LE": self.leftEyeMax, "ME": self.mouthMax, "RE": self.rightEyeMax}
        self.first_click = True

        # ------------- Other Components ----------------
        self.CreateStatusBar()
        # ------------------- Menu ----------------------
        filemenu = wx.Menu()
        id_about = wx.NewId()
        id_open = wx.NewId()
        id_save = wx.NewId()
        id_save_as = wx.NewId()
        id_exit = wx.NewId()
        # File Menu
        filemenu.Append(wx.ID_ABOUT, wx.EmptyString)
        filemenu.AppendSeparator()
        filemenu.Append(wx.ID_OPEN, wx.EmptyString)
        filemenu.Append(wx.ID_SAVE, wx.EmptyString)
        filemenu.Append(wx.ID_SAVEAS, wx.EmptyString)
        filemenu.AppendSeparator()
        filemenu.Append(wx.ID_EXIT, wx.EmptyString)

        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")  # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.

        # ----------------- Image List ------------------
        self.list = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER,
                               choices=sorted(self.image_names))
        self.list.Show(True)

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
        self.box = wx.BoxSizer(wx.HORIZONTAL)
        self.box.Add(self.list, 1, wx.EXPAND)
        self.box.Add(NC, 3, wx.EXPAND)
        self.box.Add(self.counterBox, 1, wx.EXPAND)

        self.SetAutoLayout(True)
        self.SetSizer(self.box)
        self.Layout()

        # -------------- Event Handling ----------------
        wx.EVT_LISTBOX(self, self.list.GetId(), self.onSelect)
        wx.EVT_BUTTON(self, self.saveButton.GetId(), self.onButtonSave)
        self.EventsAreBound = False
        self.BindAllMouseEvents()
        wx.EVT_MENU(self, wx.ID_OPEN, self.onOpen)
        wx.EVT_MENU(self, wx.ID_SAVE, self.onSave)
        wx.EVT_MENU(self, wx.ID_SAVEAS, self.onSaveAs)

    def BindAllMouseEvents(self):
        if not self.EventsAreBound:
            ## Here is how you catch FloatCanvas mouse events
            self.Canvas.Bind(FloatCanvas.EVT_LEFT_DOWN, self.OnLeftDown)

        self.EventsAreBound = True

    def openCSVFile(self, path):

        reader = csv.reader(open(path, "rb"))
        first = True
        eyes = False
        coords = {}
        for row in reader:
            filename = row[0]
            row = row[1:]

            if len(row) % 2 != 0:
                print "Error Loading File"
                raise TypeError("Odd number of values in this row")

            points = []
            for i in range(0, len(row), 2):
                point = (float(row[i]), float(row[i + 1]))
                points.append(point)

            coords[filename] = points
        print "CSV File Data: ", coords
        self.coords = coords

    def onButtonSave(self, event):
        i = self.image_names.index(self.image_name)
        self.image_names.remove(self.image_name)
        self.list.Set(self.image_names)
        self.list.Update()
        if len(self.image_names) > 1:
            if self.image_name:
                self.prev_image_name = self.image_name
                if self.n_points != None and len(self.coords[self.image_name]) != self.n_points:
                    print "ERROR: incorrect number of points."

            self.image_name = self.image_names[i]
            self.mirrorImage(event, shouldSave=True)
        else:
            print('You\'re Done!')

    def mirrorImage(self, event, shouldSave):
        if not self.coords.has_key(self.image_name):
            if self.prev_image_name:
                self.coords[self.image_name] = copy.copy(self.coords[self.prev_image_name])
                self.circles[self.image_name] = self.circles[self.prev_image_name]
            else:
                self.coords[self.image_name] = []
                self.circles[self.image_name] = []
        filename = os.path.join(self.image_dir, self.image_name)
        self.current_image = wx.Image(filename)

        if not self.prev_image_name:
            self.first_click = True
        self.DisplayImage(Zoom=True)
        print(self.coords)
        if shouldSave:
            self.onSave(event)


    def save(self, path):
        ''' Save the coords to a csv file. '''
        writer = csv.writer(open(path, 'wb'))

        firstRow = [' ']
        for facePart in sorted(self.faceNums.keys()):
            for j in range(1, self.faceNums[facePart] + 1):
                firstRow.append(facePart + str(j))
                firstRow.append('')
        writer.writerow(firstRow)
        keys = self.coords.keys()
        keys.sort()
        for key in keys:
            row = [key]
            for point in self.coords[key]:
                row.append(point[0])
                row.append(point[1])
            writer.writerow(row)

    def onSelect(self, event):
        if self.image_name:
            self.prev_image_name = self.image_name
            if self.n_points != None and len(self.coords[self.image_name]) != self.n_points:
                print "ERROR: incorrect number of points."

        self.image_name = event.GetString()
        self.mirrorImage(event, shouldSave=False)

    def DisplayImage(self, Zoom):
        if Zoom:
            self.Canvas.InitAll()
            if self.current_image:
                im = self.current_image.Copy()
                self.Canvas.AddScaledBitmap(im.ConvertToBitmap(), (0, 0), Height=self.imHeight, Position="tl")
                self.Canvas.ZoomToBB()
        self.Canvas._ForeDrawList = []
        i = 1
        self.leftEyeNum = 0
        self.mouthNum = 0
        self.rightEyeNum = 0
        for circle in self.coords[self.image_name]:
            if i <= self.leftEyeMax:
                self.faceNum = 'LE' + str(i)
                self.leftEyeNum += 1
            elif i <= self.mouthMax + self.leftEyeMax:
                self.faceNum = 'M' + str(i - self.leftEyeMax)
                self.mouthNum += 1
            else:
                self.faceNum = 'RE' + str(i - self.leftEyeMax - self.mouthMax)
                self.rightEyeNum += 1
            i += 1
            C = self.Canvas.AddCircle(circle, self.imHeight / 50, LineWidth=1, LineColor='RED',
                                      FillStyle='TRANSPARENT', InForeground=True)
            C.Bind(FloatCanvas.EVT_FC_LEFT_DOWN, self.move)
            self.circles[self.image_name].append(C)
        self.counterList.Clear()
        self.counterList.Append("Left Eye: {0} out of {1}".format(self.leftEyeNum, self.leftEyeMax))
        self.counterList.Append("Right Eye: {0} out of {1}".format(self.rightEyeNum, self.rightEyeMax))
        self.counterList.Append("Mouth: {0} out of {1}".format(self.mouthNum, self.mouthMax))
        self.Canvas.Draw()

    def OnLeftDown(self, event):
        self.AddCoords(event, isMoving=False)

    def AddCoords(self, event, isMoving):
        if len(self.coords[self.image_name]) < self.n_points or self.n_points == None:
            self.moving = len(self.coords[self.image_name]) - 1
            if not isMoving:
                print self.image_name
                print self.coords[self.image_name]
                self.coords[self.image_name].append(event.Coords)
            else:
                self.coords[self.image_name][self.moving] = event.Coords
        self.DisplayImage(Zoom=False)

    def move(self, object):
        self.MovingObject = object
        self.Canvas.Bind(FloatCanvas.EVT_LEFT_UP, self.movingRelease)
        self.Canvas.RemoveObject(self.MovingObject)

    def movingRelease(self, event):
        self.AddCoords(event, isMoving=True)

    def onOpen(self, event):
        print "Open"
        fd = wx.FileDialog(self, style=wx.FD_OPEN)
        fd.ShowModal()
        self.filename = fd.GetPath()
        print "On Open...", self.filename

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
            print "Saving..."
            self.onSave(event)
        else:
            print "Discarding changes..."

        # Pass this on to the default handler.
        event.Skip()


if __name__ == '__main__':
    app = wx.App(False)

    dir_dialog = wx.DirDialog(None, message="Please select a directory that contains images.")
    err = dir_dialog.ShowModal()
    image_dir = '.'
    if (err == wx.ID_OK):
        image_dir = dir_dialog.GetPath()
    else:
        print "Error getting path:", err

    print "Image Dir", image_dir
    scale = 1.0

    frame = FaceMapperFrame(None, wx.ID_ANY, "Eye Selector", image_dir, n_points=None, randomize=True,
                            scale=scale)
    frame.Show(True)
    app.MainLoop()
