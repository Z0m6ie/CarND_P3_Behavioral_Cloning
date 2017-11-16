# **Behavioural Cloning**

## David Peabody

#### Using a deep convolutional neural networks to clone a human driving input to autonomously drive a car around a simulator track.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./project_img/NVIDIA.png "NVIDIA Base Architecture"
[image2]: ./project_img/center.jpg "center lane driving"
[image3]: ./project_img/uncropped.jpg "Uncropped"
[image4]: ./project_img/cropped.jpg "Cropped"
[image5]: ./figure_1.png "Graph"



## Rubric Points
###### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model01.h5, model03.h5, model04.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* Model01.mp4, Model03.mp4, Model04.mp4 video files showing successful laps

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file along with the model.h5 file, the car can be driven autonomously around the track by executing
```sh
python drive.py model01.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolution layers 1 flatten layer, 3 dense layers and an output node.
```sh
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # Crop image
model.add(Conv2D(24, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(36, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(48, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001)))
model.add(Flatten())
model.add(Dense(100, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(50, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(10, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(1))
```

The model includes RELU activation functions to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 80).

In addition on line 83 of the clone.py file we have used keras cropping layer to crop the input image down to just the road, getting rid of the sky and trees (possible confounding factors). This also speeds up training.

#### 2. Attempts to reduce overfitting in the model

The model contains L2 regularization on all layers in order to reduce overfitting. During my various training phases I also used dropouts and gobal pooling. dropouts worked just as well but I decided to go with L2 as i hadn't really used it before. Pooling did not harm results but did not improve them either and it increased training time.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 23). I used the keras model checkpointing system to ensure all models were saved which allowed the model to be tested at every epoch. In addition I was able to plot the training and validations results to check for overfitting or underfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 103). As further work I may experiment with the different optimizers. I have used others in the past but adam usually provides the best results, so it was my default choice on this project.

#### 4. Appropriate training data

The model was trained with the sample data as a starting point and then with my own created data to a total of about 26,000 frames.

The data I created myself involved several laps of clean driving in both directions around the test course. The reverse direction was used to counter and left turn bias. I also recorded each corner multiple times taking different angles through the course.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to just get a simple working network, so initially i built a 2 layer dense network just to ensure all the supporting files and framework was working. The results of this model were... lack luster to say the least. The car managed about 5 yards before going off the road.

After this I implemented the LeNet network, famed for its character recognition. This provided better results but the car still failed at the first corner.

Seeing a general improvement I then implemented a deeper network. The NVIDIA self driving car network. This provided much more consistent results however I was still often failing on the end of the first corner.

It was at this point I decided I needed more data beyond the provided images. So I recorded a bunch of data.... However my results actually got worse!

After reading up on possible issues I found 2 main problems.
1) opencv was reading images as BRG rather than RGB. So in training and validation the NN was seeing some sort of alien world with weird colors but then when it came to test on the track its was being fed normal images. To fix this I implemented
```sh
center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
```
2) On closer inspection my model had a much lower mean squared error on the testing than the validation set. Leading me to believe on the data I had my model was over fitting and not generalizing well. To combat this I implemented a dropout of 50% on the first dense layer.

After these improvements I was able to make is successfully around the course.... Or at least I thought I did. Apparently you are not allowed to slightly touch a kerb (I was taking the racing line!)

After this I played around with the model architecture itself ultimately opting for L2 regularization on every layer. The larger breakthrough came from feeding in the left and right camera views along with a corresponding +/- 0.25 steering angle correction.

This immediately led to a successful run where no yellow lines or kerbs were harmed.

#### 2. Final Model Architecture

The final model architecture is as mentioned above basically the NVIDIA architecture with a image cropping layer and L2 regularization.

```sh
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # Crop image
model.add(Conv2D(24, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(36, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(48, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001)))
model.add(Flatten())
model.add(Dense(100, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(50, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(10, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(1))
```
##### The architecture below was the basis of my final architecture.
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded multiple lap in both directions following roughly the center line. Here is an example image of center lane driving:

![alt text][image2]

I cropped the original image in the model to reduce the amount of inconsequential information:

![alt text][image3]
![alt text][image4]

In addition I used the left and right camera data along with a corresponding corrected steering angle, +/-0.25. I used trial and error to arrive at 0.25 for the correction. my initial guesses of 0.1 and 0.4 provided rather hilarious results.
This proved to be the breakthrough I needed to make it around the course. 


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

I also recorded the training and validation graphs for each model to assess overfitting and underfitting:

##### Training & validation Graph:
![alt text][image5]


What we see in this results is a typical training & validation graph and shows a good decrease and levelling out of both the training and validation loss. This shows a model that is neither overfitting or underfitting
