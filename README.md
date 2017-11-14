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
[image3]: ./project_img/uncropped.jpg "Recovery Image"
[image4]: ./project_img/cropped.jpg "Recovery Image"
[image5]: ./Working_model_provided_data/figure_1.png "Provided Data"
[image6]: ./Working_model_self_data/figure_1.png "Self Created Data"


## Rubric Points
###### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run1.mp4 video file showing a successful lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file along with the model.h5 file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolution layers 1 flatten layer, 3 dense layers (including one dropout layer) and an output node.
```sh
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # Crop image
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
```

The model includes RELU activation functions to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 61).

In addition on line 64 of the clone.py file we have used keras cropping layer to crop the input image down to just the road, getting rid of the sky and trees (possible confounding factors). This also speeds up training.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer on line 72 in order to reduce overfitting. During my various training phases I tried several different drop outs from none, 25, 35 & 50%. 50% gave the best response when testing on the track.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 23). This validation set enabled me to use the keras model checkpointing system to ensure only the best models were saved. In addition I was able to plot the training and validations results to check for overfitting or underfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 77). As further work I may experiment with the different optimizers. I have used others in the past but adam usually provides the best results, so it was my default choice on this project.

#### 4. Appropriate training data

I have provided two models in my submission one trained purely from the sample data provided and the other trained with the sample data as a starting point and then with my own created data to a total of about 26,000 frames.

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

After these improvements I was able to make is successfully around the course with both sets of data, hence why I have included both models and videos.

#### 2. Final Model Architecture

The final model architecture is as mentioned above basically the NVIDIA architecture with an additional convolution layer, an added dropout layer and an image cropping layer.

```sh
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # Crop image
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
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

At the end of the the data recording I had roughly 26,000 frames and stearing angles.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used model checkpointing to save the model with the lowest validation score.

I also recorded the training and validation graphs for each model to assess overfitting and underfitting:

##### Provided data:
![alt text][image5]

##### Self recorded data:
![alt text][image6]

What we see in these results is a pattern of mostly slight overfitting. However it must be kept in mind that the loss numbers are very small. In fact the model used on the provided data (model used was the one with the lowest validation score) actually shows some underfitting. This possible let it generalize better with a small set of data.
