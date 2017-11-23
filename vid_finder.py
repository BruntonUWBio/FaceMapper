import glob
import os

for folder in sorted(os.listdir('/data2/OpenFaceTests')):
    if 'cb46' in folder and os.path.isdir(os.path.join('/data2/OpenFaceTests', folder)):
        if glob.glob(os.path.join(os.path.join('/data2/OpenFaceTests', folder), '*.csv')) and folder.replace('_cropped',
                                                                                                             '') not in os.listdir(
                '/home/gvelchuru/Desktop/gauth_annotations'):
            print(folder)
