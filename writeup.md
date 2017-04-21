## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[img0]: ./plots/example_images.png "Example Images"
[img1]: ./plots/data_sets.png "Data Sets"
[img2]: ./plots/training_vs_validation.png "Training vs. Validation"
[img3]: ./plots/validation_vs_test.png "Validation vs. Test"

[img4]: ./plots/grayscale.png "Grayscale Conversion"
[img5]: ./plots/luma.png "Luma Conversion"
[img6]: ./plots/resized.png "Resize Transformation"
[img7]: ./plots/rotated.png "Rotate Transformation"
[img8]: ./plots/translated.png "Translate Transformation"

[img9]: ./internet-traffic-signs/8_speed_limit_120km.jpg "Speed Limit 120km/h"
[img10]: ./internet-traffic-signs/9_no_passing.jpg "No passing"
[img11]: ./internet-traffic-signs/13_yield.jpg "Yield"
[img12]: ./internet-traffic-signs/17_no_entry.jpg "No Entry"
[img13]: ./internet-traffic-signs/21_double_curve.jpg "Double Curve"

[img14]: ./plots/visualisation_image.png "Yield Input"
[img15]: ./plots/conv_layer_1.png "Convolution Layer 1"
[img16]: ./plots/activation_layer_1.png "Activation Layer 1"
[img17]: ./plots/max_pool_layer_1.png "Max Pool Layer 1"
[img18]: ./plots/conv_layer_2.png "Convolution Layer 2"
[img19]: ./plots/activation_layer_2.png "Activation Layer 2"
[img20]: ./plots/max_pool_layer_2.png "Max Pool Layer 2"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Firstly I wanted to plot a couple of the training images to get an idea for the data set.

![example images][img0]

Next I looked at the relative sizes of the training, validation and test sets. As we  can see most of the data is used for training with a small data set for validation and around 25% of the data held back to test the final network against. This should provide a reasonable measure of the accuracy of the network.

![data sets][img1]

Then I investigated the distribution of the data sets to see whether they matched. If the test data contained lots of 120km/h signs, for example, then I would want to make sure we are training the model using lots of them too. Likewise we want our validation set to match our training set. Looking at the histograms of the sign types in the data sets we can see that the split between training, validation and test sets is good.

![training vs validation][img2]
![validation vs test][img3]

What we do notice is that the distribution of the different sign types is by no means uniform. Therefore training the model using this data I would expect it to be better at identifying certain signs. The table below shows the number of images for each type of traffic sign in the training data set. So I would expect the model to struggle more with signs at the top of the table than the bottom.

| Traffic Sign | Samples |
| --- | --- |
| Speed limit (20km/h) | 180 |
| Dangerous curve to the left | 180 |
| Go straight or left | 180 |
| Pedestrians | 210 |
| End of all speed and passing limits | 210 |
| End of no passing | 210 |
| End of no passing by vehicles over 3.5 metric tons | 210 |
| Road narrows on the right | 240 |
| Bicycles crossing | 240 |
| Double curve | 270 |
| Keep left | 270 |
| Dangerous curve to the right | 300 |
| Roundabout mandatory | 300 |
| Bumpy road | 330 |
| Go straight or right | 330 |
| End of speed limit (80km/h) | 360 |
| Vehicles over 3.5 metric tons prohibited | 360 |
| Turn left ahead | 360 |
| Beware of ice/snow | 390 |
| Slippery road | 450 |
| Children crossing | 480 |
| No vehicles | 540 |
| Traffic signals | 540 |
| Turn right ahead | 599 |
| Stop | 690 |
| Wild animals crossing | 690 |
| No entry | 990 |
| General caution | 1080 |
| Ahead only | 1080 |
| Right-of-way at the next intersection | 1170 |
| Speed limit (60km/h) | 1260 |
| Speed limit (120km/h) | 1260 |
| Speed limit (100km/h) | 1290 |
| No passing | 1320 |
| Road work | 1350 |
| Speed limit (80km/h) | 1650 |
| Speed limit (70km/h) | 1770 |
| No passing for vehicles over 3.5 metric tons | 1800 |
| Keep right | 1860 |
| Priority road | 1890 |
| Yield | 1920 |
| Speed limit (30km/h) | 1980 |
| Speed limit (50km/h) | 2010 |

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Initially I decided to convert the images to grayscale because it tends to reduce the effect of different lighting conditions on the image. 

Here is an example of a traffic sign image before and after grayscaling.

![grayscale conversion][img4]

However, having read the baseline model article [http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf "baseline model") I decided to use a YUV mapping taking the Y (or luma) component. The thought behind this move was that it would provide better contrast than the grayscale images.

![luma conversion][img5]

Having converted the images to a single colour channel (Y), I normalized the image data to zero mean and unit variance using the `sklearn.preprocessing.scale` function. This helps to keep the weights throughout the model similar which is important during back-propagation of error gradients to prevent the optimisation from becoming numerically unstable and therefore struggling to converge.

I decided to generate additional data because the original data set contains images where all of the signs are reasonably well centred and orientated in the image. This won't lead to a model that is invariant to affine transformations of the signs which is what you would be likely to get in reality. Therefore I wrote functions to resize, rotate and translate the images randomly within specified limits to generated jittered versions of the original training data set.

Here is an example of the transformed images:

![resize transformation][img6]

![rotate transformation][img7]

![translate transformation][img8]

Using these functions I created 5 jittered versions of each of the training images (rotated, translated and resized). Adding these to the training set gave the following new training set sizes:

| Training Set | Shape |
| --- | --- |
| Original Images | (34799, 32, 32, 3) |
| Original + Jittered Images | (208794, 32, 32, 3) |
| Original Labels | (34799,) |
| Original + Jittered Labels | (208794,) |

Note that I added 5 new versions of all signs. Given more time I would have liked to investigate the effect of evening out the number of instances of each type of sign in the training set to see if that produced a better result. This could be achieved by adding more jittered data for signs that have less instances in the original training set.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on the LeNet architecture and consisted of the following layers:

| Layer | Description | 
|:---:|:---:| 
| Input | 32x32x1 Y-component of YUV image | 
| Convolution 5x5 | 1x1 stride, 'valid' padding, outputs 28x28x6 	|
| RELU |	activation function |
| Max pooling | 2x2 stride,  outputs 14x14x6 |
| Convolution 5x5 | 1x1 stride, 'valid' padding, outputs 10x10x16 |
| RELU |	activation function |
| Max pooling | 2x2 stride,  outputs 10x10x16 |
| Flatten | outputs 400 |
| Fully connected	| outputs 120 |
| Dropout | keep probability of 0.75 |
| RELU |	activation function |
| Fully connected	| outputs 84 |
| Dropout | keep probability of 0.75 |
| RELU |	activation function |
| Fully connected	| outputs 43 (number of classes) |
| Softmax | converts output values to probabilities |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a `tf.train.AdamOptimizer` and minimised the mean cross entropy of my model output. I found that the following hyperparameters gave me the best performance:

```python
# Training iterations and batch size
EPOCHS = 8               # Default = 10
BATCH_SIZE = 256          # Default = 128

# Learing rate
LEARING_RATE = 0.0015     # Default = 0.001

# Weight initialisation mean and standard deviation
MU = 0                   # Default = 0
SIGMA = 0.1              # Default = 0.1

# Dropout parameters
DROPOUT_KEEP_PROB = 0.75 # Default = 0.75
```

To train the model I experimented with different values for each of the parameters and then looked at how they affected the training and validation accuracy. 

* If the training and validation accuracy were both low then I knew that the model was underfitting. In that instance I might increase the learning rate. 
* If the training accuracy was high but the validation accuracy was low then I knew the model was overfitting. In that instance I might decrease my dropout keep probability to help regulate the model and prevent overfitting.
* If the validation accuracy started to drop off during training (over each epoch) then I knew the model was starting to overfit. In that instance I might decrease the number of epochs used to train the model.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97.6%
* validation set accuracy of 97.2% 
* test set accuracy of 95.3%

I feel like the similarity between the accuracy on all data sets provides evidence that the model is behaving reasonably well.

The first architecture I chose was the LeNet-5 architecture. This architecture was originally developed to recognise hand/machine written characters and uses convolutional layers to reduce the number of parameters required to train the model. This has some cross over with traffic signs as a number of them include alpha-numerial characters. In addition the complexity level of the features in traffic signs is quite similar to that of written characters. This made it a good starting point for model.

The first issue encountered (once the data had been converted to luma and normalised) was overfitting. To combat this I added two dropout layers in the fully connected part of the model. This helped significantly to improve the accuracy against the validation set relative to the training set.

Having done this I found I could then increase my learning rate slightly which improved the fit of the model.

Then having read up on other techniques used to tackle this problem I decided to augment my training set with a jittered version of the data. This helps to provide robustness to changes in scale and orientation of the images in the test set and provided around a 2% gain in validation accuracy.

At this point I looked at the change in my training/validation accuracy over each epoch and found that I was slightly overfitting past epoch 8 and so reduced the number of epochs to this value.

Given more time I would probably experiment more with the size of the convolutional layers, and max pooling to see if I could get a better result.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![speed limit 120km/h][img9] ![no passing][img10] ![yield][img11] ![no entry][img12] ![double curve][img13]

Overall these are a pretty straightforward set of images to classify. They are all reasonably straight on to the camera and in good lighting conditions. The slightly difficult parts might be the slight shadowing over the no entry sign and the fact that there are additional signs underneath the 120km/h speed limit and double curve signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image | Prediction	| 
|:---:|:---:| 
| Speed limit (120km/h)	| Speed limit (120km/h)	| 
| No passing	| No passing	|
| Yield	| Yield	|
| No entry	| No entry	|
| Double curve	| Double curve	|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.3% but this is a very small data set with clearly visible signs in all images. Given more time it would be interesting to try the model against some tougher images where the signs are obscured or only visible under headlights for example.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the softmax probabilities for each of my web images:

##### Image 1 (Speed limit (120km/h))

| Probability | Prediction | 
|:---:|:---:| 
| 99.9% | Speed limit (120km/h) |
| 0.0% | Speed limit (100km/h) |
| 0.0% | Speed limit (20km/h) |
| 0.0% | Roundabout mandatory |
| 0.0% | Speed limit (70km/h) |

##### Image 2 (No passing)

| Probability | Prediction | 
|:---:|:---:| 
| 100.0% | No passing |
| 0.0% | End of no passing |
| 0.0% | No passing for vehicles over 3.5 metric tons |
| 0.0% | End of all speed and passing limits |
| 0.0% | Vehicles over 3.5 metric tons prohibited |

##### Image 3 (Yield)

| Probability | Prediction | 
|:---:|:---:| 
| 100.0% | Yield |
| 0.0% | No vehicles |
| 0.0% | Speed limit (50km/h) |
| 0.0% | Keep right |
| 0.0% | Stop |

##### Image 4 (No entry)

| Probability | Prediction | 
|:---:|:---:| 
| 100.0% | No entry |
| 0.0% | Stop |
| 0.0% | No passing |
| 0.0% | Turn left ahead |
| 0.0% | Yield |

##### Image 5 (Double curve)

| Probability | Prediction | 
|:---:|:---:| 
| 100.0% | Double curve |
| 0.0% | Wild animals crossing |
| 0.0% | Slippery road |
| 0.0% | Right-of-way at the next intersection |
| 0.0% | Bicycles crossing |

Clearly the model is very happy with its predictions for all of my web images. The confidence in the 120km/h sign is slightly lower which is understandable givent the similarly between it and the 100km/h sign. It is reassuring to see that the signs in the top five for this speed limit sign contains 4 speed limit signs as you would expect them to appear in the most likely list.

Again given more time I would try some different (more challenging) images so that these numbers were more intersting and I could then look into the test set to see whether signs that are misclassified in my web images were well classified by the model. This would then prvide a way to discover how the training set might be augmented to improve accuracy of certain sign types.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is a visualisation of my nerual network convolutional layers based on my Yield sign as an input. To me it looks like for the Yield sign the network is responding to the edges of the sign (top and two diagonals) as well as the large white space in the centre of the sign.

![input image][img14]
![convolution layer 1][img15]
![activation layer 1][img16]
![max pool layer 1][img17]
![convolution layer 2][img18]
![activation layer 2][img19]
![max pool layer 2][img20]
