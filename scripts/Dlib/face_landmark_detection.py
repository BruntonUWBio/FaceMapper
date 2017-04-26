#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import sys
import os
from os.path import join
import dlib
import glob
from time import sleep
from skimage import io
import scipy
from collections import defaultdict
from scipy import misc
import cv2
import numpy as np


def overlay(win, shape, d):
    win.add_overlay(shape)
    win.add_overlay(d)


def find_crop_path(self, file, crop_path):
    parts = file.split('.')
    pid = parts[0]
    out_num = int(''.join(parts[1][parts[1].index('out') + 3: parts[1].index('out') + 6]))
    out_file = None
    for crop_file in glob.iglob(os.path.join(crop_path + '/**/*.txt'), recursive=True):
        crop_file_name = os.path.splitext(os.path.basename(crop_file))[0]
        if crop_file_name == pid:
            out_file = crop_file
            break
    return out_file, out_num


def crop_predictor(img, d, crop_path, name):
    print('Name: {0}'.format(name))
    dir_name = os.path.dirname(name)
    base_name = os.path.basename(name)
    split_name = os.path.splitext(base_name)
    crop_file_path, file_num = find_crop_path(base_name, crop_path)
    print('Crop file: {0}'.format(crop_file_path))
    if crop_file_path is not None:
        f = open(crop_file_path)
        readArr = f.readlines()
        readArr = [readArr[i].split(',')[0:3] for i in range(0, len(readArr), 30)]
        for index, num in enumerate(readArr):
            for val_index, val in enumerate(num):
                readArr[index][val_index] = val.replace('(', '')
                val = readArr[index][val_index]
                readArr[index][val_index] = val.replace(')', '')
        readArr = [[float(k) for k in i] for i in readArr]

        i = file_num - 1
        if len(readArr) > i:
            confidence = readArr[i][2]
            print('Confidence: {0}'.format(confidence))
            if confidence > .25:
                x_center = readArr[i][0] * 640 / 256
                y_center = readArr[i][1] * 480 / 256
                bb_size = 150
                xmin = int(x_center - bb_size)
                ymin = int(y_center - bb_size)
                xmax = int(x_center + bb_size)
                ymax = int(y_center + bb_size)
                im = misc.imread(name, mode='RGB')
                x_coords = np.clip(np.array([xmin, xmax]), 0, im.shape[1])
                y_coords = np.clip(np.array([ymin, ymax]), 0, im.shape[0])
                xmin = x_coords[0]
                xmax = x_coords[1]
                ymin = y_coords[0]
                ymax = y_coords[1]
                crop_im = im[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]].copy()

if __name__ == '__main__':
    predictor_path = sys.argv[1]
    faces_folder_path = sys.argv[2]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()
    threshold = -1
    all = False
    pause = False
    crop = False

    if '-th' in sys.argv:
        threshold = float(sys.argv[sys.argv.index('-th') + 1])
    arg_dict = {
        '-a': all,
        '-p': pause,
        '-c': crop
    }
    for arg in list(arg_dict.keys()):
        if arg in sys.argv:
            arg_dict[arg] = True
    if crop:
        crop_path = sys.argv[sys.argv.index('-c') + 1]

    file_types = ['*.jpg', '*.png']
    files = []
    for ext in file_types:
        files.extend(glob.glob(join(faces_folder_path + '/**/', ext), recursive=True))

    files = sorted(files)

    for f in files:
        print("Processing file: {}".format(f))
        img = misc.imread(f, mode='RGB')
        detected = False
        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        # dets = detector(img, 1)

        dets, scores, idx = detector.run(img, 1, -1)
        scores_dict = defaultdict()
        print("Number of faces detected: {}".format(len(dets)))

        for i, d in enumerate(dets):

            score = scores[i]
            scores_dict[score] = [d, i, idx[i]]
            if all:
                if score > threshold:
                    print("Detection: {}, score: {}, face_type:{}".format(d, score, idx[i]))
                    if crop:
                        shape = crop_predictor(img, d, crop_path, f)
                    else:
                        shape = predictor(img, d)
                    print("Left: {} Top: {} Right: {} Bottom: {}".format(d.left(), d.top(), d.right(), d.bottom()))
                    overlay(win, shape, d)
                    detected = True
        if detected and pause:
            dlib.hit_enter_to_continue()

        if not all:
            max_score = max(list(scores_dict.keys()))
            if max_score > threshold:
                max_d = scores_dict[max_score][0]
                max_i = scores_dict[max_score][1]
                face_type = scores_dict[max_score][2]
                print("Detection {}, score: {}, face_type:{}".format(
                    max_d, max_score, face_type))
                if crop:
                    shape = crop_predictor(img, max_d, crop_path, f)
                else:
                    shape = predictor(img, d)
                print("Left: {} Top: {} Right: {} Bottom: {}".format(d.left(), d.top(), d.right(), d.bottom()))
                # Draw the face landmarks on the screen.
                overlay(win, shape, max_d)
                if pause:
                    dlib.hit_enter_to_continue()
