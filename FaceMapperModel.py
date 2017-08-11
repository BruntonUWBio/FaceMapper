from collections import OrderedDict

import numpy
import wx
from wx.lib import colourdb
from wx.lib.floatcanvas import FloatCanvas


class FaceMapperModel:
    def __init__(self, dotNum):
        self.frame_dict = {}
        # Format of frame_dict:
        # {index: [[circ_list], [coord_list]]}
        self.face_part_values = OrderedDict()

        self.face_part_list = [
            "Left Eye",
            "Right Eye",
            "Mouth",
            "Jaw",
            "Eyebrows",
            "Nose"
        ]

        self.faceParts = OrderedDict()
        self.default_face_values = [6, 6, 20, 17, 10, 9]

        for i in range(len(self.face_part_list)):
            self.faceParts[self.face_part_list[i]] = [self.default_face_values[i], []]

        colourdb.updateColourDB()
        self.color_db = wx.ColourDatabase()
        default_colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Violet']
        for index, facePart in enumerate(self.faceParts):
            self.color_db.AddColour(facePart, wx.TheColourDatabase.Find(default_colors[index]))

        self.reset_default_face_parts()

        self.totalDotNum = 0
        for facePart in self.faceParts.keys():
            self.totalDotNum += self.faceParts[facePart][1][1]

        self.coord_keys = [
            'x',
            'y',
            'drawn',
            'diameter',
            'occluded',
            'face_part',
            'guess'
        ]

        self.faceNums = []
        self.faceLabels = []
        self.reset_face_num()

        self.default_face_nums = []
        for facePart in self.faceParts:
            split = facePart.split()
            abbr = ''
            for i in range(len(split)):
                abbr += split[i].strip()[0]
            for index in range(self.faceParts[facePart][0]):
                self.default_face_nums.append(abbr + str(index + 1))

    def draw_list(self, index):
        if index in self.frame_dict:
            return self.frame_dict[index][0]
        else:
            return []

    def not_none_draw_list(self, index):
        return [x for x in self.draw_list(index) if x]

    def index_first_none(self, index):
        try:
             return max([self.draw_list(index).index(x) for x in [a for a in self.draw_list(index) if a]]) + 1
        except ValueError as e:
            return 0

    def coord_list(self, index):
        if index in self.frame_dict:
            return self.frame_dict[index][1]
        else:
            return []

    def not_none_coord_list(self, index):
        return [x for x in self.coord_list(index) if x]

    def get_default_face_part_val(self, facePart):
        return self.faceParts[facePart][0]

    # Sets the face parts to their default values, with default colors
    def reset_default_face_parts(self):
        for facePart in self.faceParts:
            self.faceParts[facePart][1] = [0, self.faceParts[facePart][0], self.color_db.Find(facePart)]

    # Makes the numbers based on face parts
    def reset_face_num(self):
        self.faceNums.clear()
        for facePart in self.faceParts:
            split = facePart.split()
            abbr = ''
            for i in range(len(split)):
                abbr += split[i].strip()[0]
            for index in range(self.faceParts[facePart][1][1]):
                self.faceNums.append(abbr + str(index + 1))

    def curr_face_part_vals(self, facePart):
        return self.faceParts[facePart][1]

    def mirror_im(self, index):
        if index not in self.frame_dict:
            if index - 1 in self.frame_dict:
                self.frame_dict[index] = self.frame_dict[index - 1]
            else:
                self.frame_dict[index] = [[None] * 68, [None] * 68]

    def remove_occluded(self, index):
        for point in self.frame_dict[index][2]:
            if point and point[4] == 1:
                point[4] = 0

    def make_face_label_list(self, index: int):
        dl = self.coord_list(index)
        for circ_ind, circle in enumerate(dl):
            if circle:
                face_part = self.curr_face_part_vals(
                    self.face_part_list[int(circle[self.coord_keys.index('face_part')])])
                face_part[0] += 1
                draw_circle = self.draw_list(index)[circ_ind]
                self.set_color(draw_circle, face_part[2].GetAsString())
                draw_circle.SetLineStyle('Solid')

        self.make_face_labels()

    # Makes face labels based on the face parts
    def make_face_labels(self):
        self.faceLabels = []
        part_index = 1
        for facePart in self.face_part_list:
            self.faceLabels.append(
                "{0}. {1}: {2} out of {3}".format(part_index, facePart, self.faceParts[facePart][1][0],
                                                  self.faceParts[facePart][1][1]))
            part_index += 1

    @staticmethod
    def set_color(circle: FloatCanvas.Circle, color: str):
        circle.SetColor(color)
        circle.SetFillColor(color)

    def next_part(self, index: int):
        ind = self.index_first_none(index) - 1
        if ind > -1:
            part_index = self.coord_list(index)[ind][self.coord_keys.index('face_part')]
            currPart = self.face_part_list[int(part_index)]
            difference = self.get_default_face_part_val(facePart=currPart) - self.curr_face_part_vals(currPart)[0]
            self.curr_face_part_vals(currPart)[1] = self.curr_face_part_vals(currPart)[0]
            for i in range(difference):
                self.draw_list(index).append(None)
                self.coord_list(index).append(None)
            self.make_face_labels()

    # Returns face number for a given circle
    def make_face_label(self, circle: FloatCanvas.Circle, index: int):
        return str(self.faceNums[self.draw_list(index).index(circle)])

    def add_point(self, imageIndex : int, index: int, point: numpy.ndarray):
        self.coord_list(imageIndex)[index] = [0] * len(self.coord_keys)
        coord_circle = self.coord_list(imageIndex)[index]
        coord_circle[self.coord_keys.index('x')] = point[0]
        coord_circle[self.coord_keys.index('y')] = point[1]
        coord_circle[self.coord_keys.index('diameter')] = 5
        coord_circle[self.coord_keys.index('face_part')] = self.get_part_index(index)
        self.draw_list(imageIndex)[index] = FloatCanvas.Circle(XY=coord_circle[0:2], Diameter=coord_circle[3],
                                                               LineWidth=.7, LineColor='Red', FillColor='Red',
                                                               FillStyle='Transparent', InForeground=True)
        self.add_index = index + 1

    def get_part_index(self, index):
        for partIndex, part in enumerate(self.face_part_list):
            if index >= self.curr_face_part_vals(part)[1]:
                index -= self.curr_face_part_vals(part)[1]
            else:
                return partIndex

    def delete_circle(self, imageIndex, circle: FloatCanvas.Circle):
        index = self.draw_list(imageIndex).index(circle)
        self.draw_list(imageIndex)[index] = None
        self.coord_list(imageIndex)[index] = None

    def set_coords(self, circle: FloatCanvas.Circle, x_y: numpy.ndarray, im_ind: int):
        index = self.draw_list(im_ind).index(circle)
        self.draw_list(im_ind)[index].XY = x_y
        self.coord_list(im_ind)[index][0:2] = x_y

    def mark_guess(self, object, index):
        coord_point = self.coord_list(index)[self.draw_list(index).index(object)]
        val = coord_point[self.coord_keys.index('guess')]

        if val == 1.0:
            coord_point[self.coord_keys.index('guess')] = 0.0
            object.SetFillStyle('Transparent')
        else:
            coord_point[self.coord_keys.index('guess')] = 1.0
            object.SetFillStyle('CrossHatch')

    def zero_face_parts(self):
        for facePart in self.faceParts:
            self.faceParts[facePart][1][0] = 0
