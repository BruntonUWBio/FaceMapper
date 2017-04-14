# Script for taking output from FaceMapperFrame (as a csv) and turning it into an xml for use to train Dlib


import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

from wx.lib.floatcanvas import Utilities


def BB(points):
    return Utilities.BBox.fromPoints([(points[i], points[i + 1]) for i in range(0, len(points), 2)])


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(
            'First argument is path to csv file')
        sys.exit()
    arg_list = sys.argv[1:]
    csv_path = arg_list[0]
    # output_path = arg_list[1]
    first_row = []
    landmark_map = defaultdict()

    for i in range(1, 18):
        landmark_map[i] = 'J' + str(i)
    for i in range(18, 28):
        landmark_map[i] = 'E' + str(i - 17)
    for i in range(28, 37):
        landmark_map[i] = 'N' + str(i - 27)
    for i in range(37, 40):
        landmark_map[i] = 'RE' + str(i - 36)
    landmark_map[40] = 'RE6'
    landmark_map[41] = 'RE5'
    landmark_map[42] = 'RE4'
    for i in range(43, 46):
        landmark_map[i] = 'LE' + str(i - 42)
    landmark_map[46] = 'LE6'
    landmark_map[47] = 'LE5'
    landmark_map[48] = 'LE4'
    for i in range(49, 56):
        landmark_map[i] = 'M' + str(i - 48)
    landmark_map[56] = 'M10'
    landmark_map[57] = 'M9'
    landmark_map[58] = 'M8'
    landmark_map[59] = 'M7'
    landmark_map[60] = 'M6'
    landmark_map[61] = 'M11'
    landmark_map[62] = 'M12'
    landmark_map[63] = 'M13'
    landmark_map[64] = 'M14'
    landmark_map[65] = 'M20'
    landmark_map[66] = 'M19'
    landmark_map[67] = 'M18'
    landmark_map[68] = 'M16'

    image_map = defaultdict()

    with open(csv_path, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index == 0:
                first_row = row
            else:
                image_map[row[0]] = []
                j = 1
                for i in range(2, len(row), 3):
                    ind = first_row.index(landmark_map[j])
                    x = str(abs(int(float(row[ind]))))
                    y = str(abs(int(float(row[ind + 1]))))
                    image_map[row[0]].append(x)
                    image_map[row[0]].append(y)
                    j += 1
                image_map[row[0]].insert(0, BB(image_map[row[0]]))
    image_list = defaultdict()
    data = ET.Element('dataset')
    images = ET.SubElement(data, 'images')
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
                              {'name': '{0}'.format(name), 'x': '{0}'.format(coord_list[i]),
                               'y': '{0}'.format(coord_list[i + 1])})
            image_list[e][bbox].append(p)
            j += 1
    fake_tree = ET.Element(None)
    pi = ET.PI("xml-stylesheet", "type='text/xsl' href='image_metadata_stylesheet.xsl'")
    pi.tail = "\n"
    fake_tree.append(pi)
    fake_tree.append(data)
    indent(fake_tree)
    tree = ET.ElementTree(fake_tree)
    tree.write(csv_path + '_with_landmarks.xml', encoding='ISO-8859-1', xml_declaration=True)
    for image in list(data):
        for file in list(image):
            for box in list(file):
                for part in list(box):
                    box.remove(part)
    tree.write(csv_path + '_without_landmarks.xml')
