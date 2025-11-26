# Welcome to the yolo_model repository!

This package is for creating, training, and applying a convolutional neural network defined using the yolo model framework ([Redmand et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)). The hallmark feature of this model framework is that input images can have any number of channels, and the network will create a representation of the input image containg exactly 3 channels. This simplifies downstream operations and allows the user to apply this model to images with a variable number of channels.

There are 3 major steps in this pipeline:
(1) Preprocess the input image
(2) Apply the model to the pre-processed image
(3) Postprocess the output

Preprocessing
=============
Normalize the pixel/voxel values with respect to all these values in a given image.

(Optional) Apply a gamma correction (gamma = 0.5) to the input image, to adjust for abnormally large variations in brightness across the image

(Optional) Upsample the input image by a factor of 2. Useful if target objects only occupy a few pixels (i.e. cell nuclei segmentation)

