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
import re
from PIL import Image


def overlay(win, shape, d):
    win.add_overlay(shape)
    win.add_overlay(d)


def find_crop_path(file, crop_path, crop_txt_files):
    parts = file.split('.')
    pid = parts[0]
    try:
        back_half_name = ''.join(parts[1][parts[1].index('out') + 3: len(parts[1])])
        out_num = int(re.sub("[^0-9]", "", back_half_name))
    except ValueError:
        return None, None
    out_file = None
    if pid in list(crop_txt_files.keys()):
        out_file = crop_txt_files[pid]
    return out_file, out_num


def crop_predictor(img, crop_path, name, crop_txt_files):
    print('Name: {0}'.format(name))
    dir_name = os.path.dirname(name)
    base_name = os.path.basename(name)
    split_name = os.path.splitext(base_name)
    crop_file_path, file_num = find_crop_path(base_name, crop_path, crop_txt_files)
    print('Crop file: {0}'.format(crop_file_path))
    if crop_file_path is not None:
        f = open(crop_file_path)
        readArr = f.readlines()
        readArr = [readArr[i].split(',')[0:3] for i in range(0, len(readArr), fps_frac)]
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
                im = img
                x_coords = np.clip(np.array([xmin, xmax]), 0, im.shape[1])
                y_coords = np.clip(np.array([ymin, ymax]), 0, im.shape[0])
                xmin = x_coords[0]
                xmax = x_coords[1]
                ymin = y_coords[0]
                ymax = y_coords[1]
                # pil_im = Image.open(name)
                # pil_im = pil_im.crop((xmin, ymin, xmax, ymax))
                # new_file = os.path.join(dir_name, split_name[0] + '_cropped' + split_name[1])
                # pil_im.save(new_file)
                crop_im = im[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]].copy()
                # crop_im = misc.imread(new_file, mode='RGB')
                return [crop_im, xmin, ymin]

if __name__ == '__main__':
    predictor_path = sys.argv[1]
    faces_folder_path = sys.argv[2]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()
    threshold = -1

    if '-th' in sys.argv:
        threshold = float(sys.argv[sys.argv.index('-th') + 1])
    arg_dict = {
        '-a': False,
        '-p': False,
        '-c': False,
        '-f': 1
    }
    for arg in list(arg_dict.keys()):
        if arg in sys.argv:
            arg_dict[arg] = True
    all = arg_dict['-a']
    pause = arg_dict['-p']
    crop = arg_dict['-c']
    fps_frac = arg_dict['-f']
    if crop:
        crop_path = sys.argv[sys.argv.index('-c') + 1]
        crop_txt_files = {os.path.splitext(os.path.basename(v))[0]: v for v in
                          glob.iglob(os.path.join(crop_path + '/**/*.txt'), recursive=True)}

    file_types = ['*.jpg', '*.png']
    files = []
    for ext in file_types:
        files.extend(glob.glob(join(faces_folder_path + '/**/', ext), recursive=True))

    files = sorted(files)

    for f in files:
        dir_name = os.path.dirname(f)
        base_name = os.path.basename(f)
        split_name = os.path.splitext(base_name)
        img = misc.imread(f, mode='RGB')
        xmin = 0
        ymin = 0
        if crop:
            cv_im = misc.imread(f, mode='RGB')
            crop_im_arr = crop_predictor(cv_im, crop_path, f, crop_txt_files)
            crop_im = crop_im_arr[0]
            xmin = crop_im_arr[1]
            ymin = crop_im_arr[2]
        detected = False
        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        # dets = detector(img, 1)

        dets, scores, idx = detector.run(crop_im, 1, -1)
        if len(scores) >= 1:
            scores_dict = defaultdict()
            print("Number of faces detected: {}".format(len(dets)))

            for i, d in enumerate(dets):
                old_top = d.top()
                old_left = d.left()
                old_right = d.right()
                old_bottom = d.bottom()
                new_top = old_top + ymin
                new_left = old_left + xmin
                new_right = old_right + xmin
                new_bottom = old_bottom + ymin
                c = cv2.rectangle(img, (new_left, new_top), (new_right, new_bottom), (255, 0, 255))
                score = scores[i]
                scores_dict[score] = [d, i, idx[i]]
                if all:
                    if score > threshold:
                        print("Detection: {}, score: {}, face_type:{}".format(d, score, idx[i]))
                        shape = predictor(img, d)
                        print("Left: {} Top: {} Right: {} Bottom: {}".format(new_left, new_top, new_right, new_bottom))
                        new_name = os.path.join(dir_name, split_name[0] + '_with_rectangles' + split_name[1])
                        cv2.imwrite(new_name, img)
                        img = misc.imread(new_name)
                        win.set_image(img)
                        detected = True
            if detected and pause:
                dlib.hit_enter_to_continue()

            if not all:
                max_score = max(list(scores_dict.keys()))
                if max_score > threshold:
                    max_d = scores_dict[max_score][0]
                    max_i = scores_dict[max_score][1]
                    old_top = max_d.top()
                    old_left = max_d.left()
                    old_right = max_d.right()
                    old_bottom = max_d.bottom()
                    new_top = old_top + ymin
                    new_left = old_left + xmin
                    new_right = old_right + xmin
                    new_bottom = old_bottom + ymin
                    c = cv2.rectangle(img, (new_left, new_top), (new_right, new_bottom), (255, 0, 255))
                    face_type = scores_dict[max_score][2]
                    print("Detection {}, score: {}, face_type:{}".format(
                        max_d, max_score, face_type))
                    shape = predictor(img, max_d)
                    print("Left: {} Top: {} Right: {} Bottom: {}".format(new_left, new_top, new_right, new_bottom))
                    new_name = os.path.join(dir_name, split_name[0] + '_with_rectangles' + split_name[1])
                    cv2.imwrite(new_name, img)
                    img = misc.imread(new_name)
                    win.set_image(img)
                    # Draw the face landmarks on the screen.
                    if pause:
                        dlib.hit_enter_to_continue()
