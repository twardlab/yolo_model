# Welcome to the yolo_model repository!

This package is for creating, training, and applying a convolutional neural network defined using the yolo model framework (CITE). The hallmark feature of this model framework is that input images can have any number of channels, and the network will create a representation of the input image containg exactly 3 channels. This simplifies downstream operations and allows the user to apply this model to images with a variable number of channels.

The outputs from this model can then be filtered using non-maximum suppression. The performance of this model can then be quantified by computing the mean average precision across a set of 10 PR curves describing the quality of our model's predicted bounding boxes over a simulated datset consisting of 100 images which resemble micsocopy slides.

