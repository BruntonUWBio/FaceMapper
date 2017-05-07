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
from collections import defaultdict
from scipy import misc
import numpy as np
import re
import cv2


class Detector:
    def __init__(self):
        predictor_path = sys.argv[1]
        faces_folder_path = sys.argv[2]

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.win = dlib.image_window()
        self.threshold = -1

        if '-th' in sys.argv:
            self.threshold = float(sys.argv[sys.argv.index('-th') + 1])
        arg_dict = {
            '-a': False,
            '-p': False,
            '-c': False,
            '-n': False,
            '-f': 1,
            '-s': False,
        }
        for arg in list(arg_dict.keys()):
            if arg in sys.argv:
                arg_dict[arg] = True
        self.all = arg_dict['-a']
        self.pause = arg_dict['-p']
        self.crop = arg_dict['-c']
        self.nose = arg_dict['-n']
        self.fps_frac = arg_dict['-f']
        self.save = arg_dict['-s']

        crop_path = None
        nose_txt_files = None
        nose_path = None
        crop_txt_files = None

        if self.nose:
            nose_path = sys.argv[sys.argv.index('-n') + 1]
            nose_txt_files = self.find_txt_files(nose_path)

        if self.crop:
            crop_path = sys.argv[sys.argv.index('-c') + 1]
            crop_txt_files = self.find_txt_files(crop_path)

        file_types = ['*.jpg', '*.png']
        files = []
        for ext in file_types:
            files.extend(glob.glob(join(faces_folder_path + '/**/', ext), recursive=True))

        files = sorted(files)
        for f in files:
            if '_detected' in f:
                files.remove(f)

        for f in files:
            print("Processing file: {}".format(f))
            img = misc.imread(f, mode='RGB')
            img = misc.imresize(img, (960, 1280))
            scaled_width = img.shape[1]
            scaled_height = img.shape[0]
            detected = False
            self.win.clear_overlay()
            self.win.set_image(img)
            if self.nose and self.crop:
                crop_im_arr = self.crop_predictor(img, f, crop_txt_files, nose_path, nose_txt_files,
                                                  scaled_height=scaled_height, scaled_width=scaled_width)
                if crop_im_arr is not None:
                    crop_im = crop_im_arr[0]
                    x_min = crop_im_arr[1]
                    y_min = crop_im_arr[2]
                    x_max = crop_im_arr[3]
                    y_max = crop_im_arr[4]
                    scores_dict = self.show_face(f, crop_im, detected, show=False)

                    if not self.all:
                        max_score, max_d = self.find_maxes(scores_dict)
                        if max_score is not None:
                            old_top = max_d.top()
                            old_left = max_d.left()
                            old_right = max_d.right()
                            old_bottom = max_d.bottom()
                            self.win.set_image(img)
                            new_top = int(old_top + y_min)
                            new_left = int(old_left + x_min)
                            new_right = int(old_right + x_min)
                            new_bottom = int(old_bottom + y_min)
                            new_d = dlib.rectangle(left=new_left, top=new_top, right=new_right, bottom=new_bottom)
                            self.show_best_face(name=f, scores_dict=scores_dict, img=img, show=True,
                                                max_score=max_score, max_d=new_d, save=True)


                else:
                    dir_name, base_name, split_name = self.splitname(f)
                    new_name = self.new_file_name(os.path.join(dir_name, 'detected/'), split_name, '_detected')
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(new_name, img)
            else:
                self.show_face(f, img, detected)

    def overlay(self, shape, d):
        self.win.add_overlay(shape)
        self.win.add_overlay(d)

    @staticmethod
    def splitname(name):
        dir_name = os.path.dirname(name)
        base_name = os.path.basename(name)
        split_name = os.path.splitext(base_name)
        return dir_name, base_name, split_name

    @staticmethod
    def find_crop_path(file, crop_txt_files):
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

    def make_read_arr(self, f, num_constraint=None):
        readArr = f.readlines()
        if num_constraint is not None:
            readArr = [readArr[i].split(',')[0:num_constraint] for i in range(0, len(readArr), self.fps_frac)]
        else:
            readArr = [readArr[i].split(',') for i in range(0, len(readArr), self.fps_frac)]
        for index, num in enumerate(readArr):
            for val_index, val in enumerate(num):
                readArr[index][val_index] = val.replace('(', '')
                val = readArr[index][val_index]
                readArr[index][val_index] = val.replace(')', '')
        readArr = [[float(k) for k in i] for i in readArr]
        return readArr

    def crop_predictor(self, img, name, crop_txt_files, nose_path, nose_txt_files, scaled_width, scaled_height):
        print('Name: {0}'.format(name))
        dir_name = os.path.dirname(name)
        base_name = os.path.basename(name)
        split_name = os.path.splitext(base_name)
        crop_file_path, file_num = self.find_crop_path(base_name, crop_txt_files)
        print('Crop file: {0}'.format(crop_file_path))
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0
        if crop_file_path is not None:
            f = open(crop_file_path)
            readArr = self.make_read_arr(f)
            i = file_num - 1
            if len(readArr) > i:
                curr_im_coords = readArr[i]
                x_min = curr_im_coords[0] * scaled_width / 640
                y_min = curr_im_coords[2] * scaled_height / 480
                x_max = curr_im_coords[1] * scaled_width / 640
                y_max = curr_im_coords[3] * scaled_height / 480

        nose_file_path, file_num = self.find_crop_path(base_name, nose_txt_files)
        print('Nose file: {0}'.format(nose_file_path))
        if nose_file_path is not None:
            f = open(nose_file_path)
            readArr = self.make_read_arr(f, 3)

            i = file_num - 1
            if len(readArr) > i:
                confidence = readArr[i][2]
                print('Crop Confidence: {0}'.format(confidence))
                if confidence > .25:
                    x_center = readArr[i][0]
                    y_center = readArr[i][1]
                    norm_coords = self.normalize_to_camera([(x_center, y_center)], [x_min, x_max, y_min, y_max],
                                                           scaled_width=scaled_width, scaled_height=scaled_height)
                    x_center = norm_coords[0][0]
                    y_center = norm_coords[0][1]
                    bb_size = 100
                    x_min = int(x_center - bb_size)
                    y_min = int(y_center - bb_size)
                    x_max = int(x_center + bb_size)
                    y_max = int(y_center + bb_size)
                    im = img
                    x_coords = np.clip(np.array([x_min, x_max]), 0, im.shape[0])
                    y_coords = np.clip(np.array([y_min, y_max]), 0, im.shape[1])
                    x_min = x_coords[0]
                    x_max = x_coords[1]
                    y_min = y_coords[0]
                    y_max = y_coords[1]
                    crop_im = im[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]].copy()
                    return [crop_im, x_min, y_min, x_max, y_max]

    def normalize_to_camera(self, coords, crop_coord, scaled_width, scaled_height):
        if sum(crop_coord) <= 0:
            rescale_factor = (scaled_width / 256, scaled_height / 256)  # Original size was 256
        else:
            rescale_factor = ((crop_coord[1] - crop_coord[0]) / 256.0, (crop_coord[3] - crop_coord[2]) / 256.0)
        norm_coords = [
            np.array((coord[0] * rescale_factor[0] + crop_coord[0], coord[1] * rescale_factor[1] + crop_coord[2]))
            for coord in coords]
        return np.array(norm_coords)

    @staticmethod
    def new_file_name(dir_name, split_name, addition):
        return os.path.join(dir_name, split_name[0] + addition + split_name[1])

    def show_best_face(self, name, scores_dict, img, show=True, max_score=None, max_d=None, save=False):
        if max_score is None and max_d is None:
            max_score, max_d = self.find_maxes(scores_dict)
        if max_score is not None:
            max_i = scores_dict[max_score][1]
            face_type = scores_dict[max_score][2]
            shape = None
            print("Detection {}, score: {}, face_type:{}".format(
                max_d, max_score, max_i))
            if self.threshold is not None and max_score is not None and max_score > self.threshold:
                shape = self.predictor(img, max_d)
                print("Left: {} Top: {} Right: {} Bottom: {}".format(max_d.left(), max_d.top(), max_d.right(),
                                                                     max_d.bottom()))
                # Draw the face landmarks on the screen.
                if show:
                    self.overlay(shape, max_d)
                if self.pause:
                    dlib.hit_enter_to_continue()
                if save and self.save:
                    self.save_im(name, shape, max_d, scale=True)

    def save_im(self, name, shape, d, scale):
        img = cv2.imread(name)
        img = misc.imresize(img, (960, 1280))
        dir_name, basename, split_name = self.splitname(name)
        new_name = self.new_file_name(os.path.join(dir_name, 'detected/'), split_name, '_detected')
        box = cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), color=(255, 0, 0))
        for i in range(shape.num_parts):
            dot = shape.part(i)
            cv_dot = cv2.circle(img, (dot.x, dot.y), 3, (0, 0, 255))
        cv2.imwrite(new_name, img)

    def show_face(self, name, img, detected, show=True):
        dets, scores, idx = self.detector.run(img, 1, -1)
        scores_dict = defaultdict()
        print("Number of faces detected: {}".format(len(dets)))

        for i, d in enumerate(dets):
            score = scores[i]
            scores_dict[score] = [d, i, idx[i]]
            if show:
                if self.all:
                    if score > self.threshold:
                        print("Detection: {}, score: {}, face_type:{}".format(d, score, idx[i]))
                        shape = self.predictor(img, d)
                        print("Left: {} Top: {} Right: {} Bottom: {}".format(d.left(), d.top(), d.right(), d.bottom()))
                        self.overlay(shape, d)
                        detected = True
        if detected and self.pause:
            dlib.hit_enter_to_continue()

        if show and not self.all:
            self.show_best_face(name, scores_dict, img=img)

        return scores_dict

    def find_maxes(self, scores_dict):
        try:
            max_score = max(list(scores_dict.keys()))
        except:
            return None, None
        if max_score > self.threshold:
            max_d = scores_dict[max_score][0]
            return max_score, max_d
        else:
            return None, None

    @staticmethod
    def find_txt_files(path):
        return {os.path.splitext(os.path.basename(v))[0]: v for v in
                glob.iglob(os.path.join(path + '/**/*.txt'), recursive=True)}


if __name__ == '__main__':
    det = Detector()
