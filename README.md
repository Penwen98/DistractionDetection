# DistractionDetection

## Introduction

This project aims to classify the distraction of a driver into one of **five categories**, using deep convolutional neural networks. The model is trained on a custom dataset given by the researchers found on this site https://dmd.vicomtech.org/.

## Dependencies

Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)

## Algorithm

* First, the **haar cascade** method is used to detect the upper body of the driver in each frame of the webcam feed.

* The region of image containing the driver is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the five classes of distractions.

* The distraction with maximum score is displayed on the screen.
