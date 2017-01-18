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



import os
import os.path
import random

import wx
from wx.lib.floatcanvas import NavCanvas

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
        self.filename = None
        self.coords = {}

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
        self.list = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER | wx.LB_SORT,
                               choices=self.image_names)
        self.list.Show(True)

        # ----------------- Image Display ---------------
        NC = NavCanvas.NavCanvas(self, Debug=0, BackgroundColor="BLACK")
        self.Canvas = NC.Canvas
        self.Canvas.MinScale = 14
        self.Canvas.MaxScale = 500

        # ----------------- Counter Display ------------
        self.counterList = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER | wx.LB_SORT,
                                      choices=self.faceLabels)

        # ----------------- Window Layout  -------------
        self.box = wx.BoxSizer(wx.HORIZONTAL)
        self.box.Add(self.list, 1, wx.EXPAND)
        self.box.Add(NC, 3, wx.EXPAND)
        self.box.Add(self.counterList, 1, wx.EXPAND)

        self.SetAutoLayout(True)
        self.SetSizer(self.box)
        self.Layout()

        # -------------- Event Handling ----------------
        wx.EVT_LISTBOX(self, self.list.GetId(), self.onSelect)

    def onSelect(self, event):
        if self.image_name:
            self.prev_image_name = self.image_name
            if self.n_points != None and len(self.coords[self.image_name]) != self.n_points:
                print "ERROR: incorrect number of points."

        self.image_name = event.GetString()

        if not self.coords.has_key(self.image_name):
            if self.prev_image_name:
                self.coords[self.image_name] = self.coords[self.prev_image_name]
            else:
                self.coords[self.image_name] = []
        filename = os.path.join(self.image_dir, self.image_name)
        self.current_image = wx.Image(filename)

        if not self.prev_image_name:
            self.first_click = True
        self.DisplayImage()

    def DisplayImage(self):
        self.Canvas.InitAll()
        if self.current_image:
            im = self.current_image.Copy()
            self.Canvas.AddScaledBitmap(im.ConvertToBitmap(), (0, 0), Height=50, Position="tl")
        self.Canvas.ZoomToBB()


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
