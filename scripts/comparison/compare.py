import os
import numpy as np

from OpenFaceScripts import AUGui

my_dir = '/home/gvelchuru/Desktop/gauth_annotations'
maya_dir = '/data2/OpenFaceTests'
differences = []
total_num = 0
for dir in os.listdir(my_dir):
    my_csv = os.path.join(my_dir, dir, dir + '_emotions.csv')
    my_dict = AUGui.csv_emotion_reader(my_csv)
    maya_csv = os.path.join(maya_dir, dir + '_cropped', dir + '_emotions.csv')
    maya_dict = AUGui.csv_emotion_reader(maya_csv)
    difference_num = 0
    for num in my_dict:
        if num in maya_dict and maya_dict[num] != my_dict[num]:
            difference_num += 1
    differences.append(difference_num / len(my_dict))
differences = np.array(differences)
print('Mean: {0}'.format(np.mean(differences)))
print('SD: {0}'.format(np.std(differences)))
# print('Total: {0}'.format(np.sum(differences)))
# print('Total Num: {0}'.format(total_num))
