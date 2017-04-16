# Script for taking output from FaceMapperFrame (as a csv) and turning it into an xml for use to train Dlib


import csv
import sys
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import glob
import random

import copy
from wx.lib.floatcanvas import NavCanvas, FloatCanvas, Utilities


class XmlTransformer:  # CSV File in Disguise
    def __init__(self):
        if len(sys.argv) != 2:
            print(
                'First argument is path')
            sys.exit()
        arg_list = sys.argv[1:]
        path = arg_list[0]
        # output_path = arg_list[1]
        first_row = []
        self.landmark_map = defaultdict()
        self.make_landmark_map()
        self.data = ET.Element('dataset')
        self.images = ET.SubElement(self.data, 'images')
        self.append_data(path)
        fake_tree = ET.Element(None)
        pi = ET.PI("xml-stylesheet", "type='text/xsl' href='image_metadata_stylesheet.xsl'")
        pi.tail = "\n"
        fake_tree.append(pi)
        fake_tree.append(self.data)
        self.indent(fake_tree)
        tree = ET.ElementTree(fake_tree)

        test_data = ET.Element('dataset')
        test_images = ET.SubElement(test_data, 'images')

        tree.write(path + 'training_with_face_landmarks.xml', encoding='ISO-8859-1', xml_declaration=True)
        rand = random.randrange(10)
        for index, image in enumerate(self.data):
            for file in list(image):
                if index % 10 == 0:
                    rand = random.randrange(10)
                if index % 10 == rand:
                    self.images.remove(file)
                    test_images.append(file)
                for box in list(file):
                    for part in list(box):
                        box.remove(part)
        tree.write(path + 'training.xml', encoding='ISO-8859-1', xml_declaration=True)

        pi_fake = copy.deepcopy(pi)
        test_fake_tree = ET.Element(None)
        test_fake_tree.append(pi)
        test_fake_tree.append(test_data)
        self.indent(test_fake_tree)
        test_tree = ET.ElementTree(test_fake_tree)

        test_tree.write(path + 'testing_with_face_landmarks.xml', encoding='ISO-8859-1', xml_declaration=True)
        self.remove_parts(test_data)
        test_tree.write(path + 'testing.xml', encoding='ISO-8859-1', xml_declaration=True)

    def remove_parts(self, data):
        for index, image in enumerate(data):
            for file in list(image):
                for box in list(file):
                    for part in list(box):
                        box.remove(part)

    def append_data(self, path):
        for file in glob.iglob(path + '/**/*.csv', recursive=True):
            for image in self.csv_to_xml(file):
                self.images.append(image)
        for file in glob.iglob(path + '/**/*.pts', recursive=True):
            for image in self.pts_to_xml(file):
                self.images.append(image)

    @staticmethod
    def bb(points):
        return Utilities.BBox.fromPoints([(points[i], points[i + 1]) for i in range(0, len(points), 2)])

    def csv_to_xml(self, csv_path):
        image_map = defaultdict()
        first_row = None
        split_path = os.path.dirname(csv_path)
        with open(csv_path, 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for index, row in enumerate(reader):
                if index == 0:
                    first_row = row
                else:
                    filename = split_path + '/' + row[0]
                    image_map[filename] = []
                    j = 1
                    for i in range(2, len(row), 3):
                        ind = first_row.index(self.landmark_map[j])
                        x = str(abs(int(float(row[ind]))))
                        y = str(abs(int(float(row[ind + 1]))))
                        image_map[filename].append(x)
                        image_map[filename].append(y)
                        j += 1
                    image_map[filename].insert(0, self.bb(image_map[filename]))
        return self.make_image_list(image_map)

    def pts_to_xml(self, pts_path):
        print(pts_path)
        pt_file = open(pts_path, 'r+')
        s = pt_file.read()
        split_lines = s.splitlines()[3:]
        split_lines = split_lines[0:len(split_lines) - 1]
        if '}' in split_lines:
            split_lines.remove('}')
        if 'version: 1' in split_lines:
            split_lines.remove('version: 1')
        if 'n_points: 68' in split_lines:
            split_lines.remove('n_points: 68')
        if '{' in split_lines:
            split_lines.remove('{')
        split_path = os.path.dirname(pts_path)
        image_map = defaultdict()
        just_file = os.path.basename(pts_path)
        just_file = just_file[0: len(just_file) - 4]
        filename = split_path + '/' + just_file + '.png'
        image_map[filename] = []
        for i in range(len(split_lines)):
            pts = split_lines[i].split(' ')
            x = str(abs(int(float(pts[0]))))
            y = str(abs(int(float(pts[1]))))
            image_map[filename].append(x)
            image_map[filename].append(y)
        image_map[filename].insert(0, self.bb(image_map[filename]))
        return self.make_image_list(image_map)

    @staticmethod
    def make_image_list(image_map):
        image_list = defaultdict()
        images = ET.Element('images')
        for index, file in enumerate(image_map.keys()):
            e = ET.SubElement(images, 'image', {'file': '{0}'.format(file)})
            image_list[e] = {}
            coord_list = image_map[file]
            bb = coord_list[0].astype(int)
            bbox = ET.SubElement(e,
                                 'box',
                                 {
                                     'top': '{0}'.format(bb[0][1]),
                                     'left': '{0}'.format(bb[0][0]),
                                     'width': '{0}'.format(bb[1][0] - bb[0][0]),
                                     'height': '{0}'.format(bb[1][1] - bb[0][1])
                                 })
            image_list[e][bbox] = []
            j = 0
            for i in range(1, len(coord_list), 2):
                name = ''
                if j < 10:
                    name = str('0' + str(j))
                else:
                    name = str(j)
                p = ET.SubElement(bbox, 'part',
                                  {'name': '{0}'.format(name),
                                   'x': '{0}'.format(coord_list[i]),
                                   'y': '{0}'.format(coord_list[i + 1])})
                image_list[e][bbox].append(p)
                j += 1
        return images

    def indent(self, elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def make_landmark_map(self):
        for i in range(1, 18):
            self.landmark_map[i] = 'J' + str(i)
        for i in range(18, 28):
            self.landmark_map[i] = 'E' + str(i - 17)
        for i in range(28, 37):
            self.landmark_map[i] = 'N' + str(i - 27)
        for i in range(37, 40):
            self.landmark_map[i] = 'RE' + str(i - 36)
        self.landmark_map[40] = 'RE6'
        self.landmark_map[41] = 'RE5'
        self.landmark_map[42] = 'RE4'
        for i in range(43, 46):
            self.landmark_map[i] = 'LE' + str(i - 42)
        self.landmark_map[46] = 'LE6'
        self.landmark_map[47] = 'LE5'
        self.landmark_map[48] = 'LE4'
        for i in range(49, 56):
            self.landmark_map[i] = 'M' + str(i - 48)
        self.landmark_map[56] = 'M10'
        self.landmark_map[57] = 'M9'
        self.landmark_map[58] = 'M8'
        self.landmark_map[59] = 'M7'
        self.landmark_map[60] = 'M6'
        self.landmark_map[61] = 'M11'
        self.landmark_map[62] = 'M12'
        self.landmark_map[63] = 'M13'
        self.landmark_map[64] = 'M14'
        self.landmark_map[65] = 'M20'
        self.landmark_map[66] = 'M19'
        self.landmark_map[67] = 'M18'
        self.landmark_map[68] = 'M16'


if __name__ == '__main__':
    transform = XmlTransformer()
