# Lane-and-Object-Detection-for-Self-Driving-Cars-with-Road-Signs-Recognition
This project is about deep learning solution for lane , road signs and object detection in self driving cars which i with my team prepared for our final year project.
## What is Object Detection?
Object Detection is the process of finding real-world object instances like cars, bikes, TVs, flowers, and humans in still images or videos. It allows for the recognition, localization, and detection of multiple objects within an image, which provides us with a much better understanding of an image as a whole.It is commonly used in applications such as image retrieval, security, surveillance, and advanced driver assistance systems (ADAS).
Object detection can be done in multiple ways: Feature-based object detection Viola-Jones object detection SVM classifications with HOG features Deep learning object detection.
## A General Framework for Object Detection
Typically, we follow three steps when building an object detection framework:
First, a deep learning model or algorithm is used to generate a large set of bounding boxes spanning the full image (that is, an object localization component)
![image](https://user-images.githubusercontent.com/58527014/121800971-693e3080-cc52-11eb-8f70-09f95cda19d9.png)

Next, visual features are extracted for each of the bounding boxes. They are evaluated and it is determined whether and which objects are present in the boxes based on visual features (i.e. an object classification component)
![image](https://user-images.githubusercontent.com/58527014/121800954-54fa3380-cc52-11eb-9017-16d578e3f819.png)

In the final post-processing step, overlapping boxes are combined into a single bounding box (that is, non-maximum suppression)
![image](https://user-images.githubusercontent.com/58527014/121800981-74915c00-cc52-11eb-9dd3-afedaf56e015.png)

## What is MobileNet SSD?
### SSD
The SSD architecture is a single convolution network that learns to predict bounding box locations and classify these locations in one pass. Hence, SSD can be trained end-to-end. The SSD network consists of base architecture (MobileNet in this case) followed by several convolution layers:
![image](https://user-images.githubusercontent.com/58527014/121801030-cdf98b00-cc52-11eb-90f2-7876c52b9022.png)

## Lane Detection
To detect and draw a polygon that takes the shape of the lane the car is currently in, we build a pipeline consisting of the following steps:

1) Computation of camera calibration matrix and distortion coefficients from a set of chessboard images
2) Distortion removal on images
3) Application of color and gradient thresholds to focus on lane lines
4) Production of a bird’s eye view image via perspective transform
5) Use of sliding windows to find hot lane line pixels
6) Fitting of second degree polynomials to identify left and right lines composing the lane
7) Computation of lane curvature and deviation from lane center
8) Warping and drawing of lane boundaries on image as well as lane curvature information

![image](https://user-images.githubusercontent.com/58527014/121801113-3f393e00-cc53-11eb-918c-e6503dbd6e74.png)

## A General Framework for Road Signs Detection:
### ResNet-50:
ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. This model was the winner of ImageNet challenge in 2015. The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks with 150+layers successfully. Prior to ResNet training very deep neural networks was difficult due to the problem of vanishing gradients.
![image](https://user-images.githubusercontent.com/58527014/121801132-6132c080-cc53-11eb-80e8-7699865a2f8d.png)

### Steps for Loading the Road Sign Detection Model:
1) First of all, I segregated classes and then selected the one which we thought of to go with. We selected 23 classes which includes all the basic classes which are found on the roads.
2) Secondly, we selected around 23 thousand and 400 images. And trained it into a pre-trained ResNet 50 model with the inputs of the images as the classes specified by us.
3) ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pre-trained version of the network trained on more than a million images from the Image-Net database. The pre-trained network has input as our classes which we have specified and as an output it forms out the tensorflow lite file. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.
4) We take the tflite file as an output to our module, which we further input it into the application. In the input part, we first have designed the application according to us and have taken that trained model as an input to our application. It uses the camera on your phone in order to detect the objects.
5) Then we specified the percentage of detection and which of the three traffic signs matches the live detected sign, as it is detecting, app also includes the feature of the detection percentage (%), i.e. what is the chances that is gets detected and how accurate it is.
6) As soon as we bring the camera near to a sign, it detects it out and displays the percentage as an indicator to how much it is detected.

#### Requirements for the development of the App
1) Android Studio 3.2 (installed on a Linux, Mac or Windows machine)
2) Android device in developer mode with USB debugging enabled
3) USB cable (to connect Android device to your computer)






