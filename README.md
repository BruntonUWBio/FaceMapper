# Description
Use FaceMapperFrame.py to annotate a sequence of faces from a directory or video with dots - the coordinates can be saved to a CSV

# Features
Right-click and drag on a dot to change size<br />
Click on one of the categories on the right to change a color or reset numbers<br />
Hold CTRL while scrolling on a dot to change the color for all of the dots in that category <br />
CTRL click and drag to select multiple dots (selected dots have dashed outlines)<br />
CTRL + right-click + drag on canvas will rotate all selections around their center <br />
CTRL + right-click + drag on a dot will resize all selections <br />
Right click to remove selections  <br />
Press DEL while dots are selected to remove them <br />
CTRL + Click on a dot to select all dots of that part <br />
Double-click on a dot to mark/unmark as guess <br />
Right double click on canvas to mark all selections as guess <br />

# Command Line Arguments
- "csv %path/to/csv" will open from a pre-existing file inside of an image directory

# Requirements
- ffmpeg (for video processing)

 
# Scripts
## XmlTransformer
XMLTransformer turns a CSV (outputted by FaceMapperFrame) or a .pts file into an xml format usable by Dlib
### Command Line Arguments
- -g flag for including dots marked as guess in FaceMapper tool
## face_landmark_detection
Contains some tweaks to Dlib's packaged face landmark detection script.
### Features
 - Increased file types supported
### Command line arguments
- "th %threshold_value" will show all faces with a greater confidence than the specified threshold value


