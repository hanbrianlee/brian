# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/image1.png "histogram"
[image2]: ./writeup_images/image2.png "example image"
[image3]: ./writeup_images/image3.jpg "web1"
[image4]: ./writeup_images/image4.jpg "web2"
[image5]: ./writeup_images/image5.jpg "web3"
[image6]: ./writeup_images/image6.jpg "web4"
[image7]: ./writeup_images/image7.jpg "web5"
[image8]: ./writeup_images/image8.png "sampleResult"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my Traffic_sign_classifier
(https://github.com/hanbrianlee/brian/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python native command .shape for arrays to get the shape of one of the images. As for unique classes, I used numpy concatenate to put all the y_train, t_valid, y_test into one array and then used np.unique commands followed by len command to find the number of unique classes.

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
To start off, I plotted a histrogram of y_train to see what the distribution of 43 unique classes are in terms of examples. This is important to see because if the data are skewed (i.e. for some classes there are tons of examples but for other classes scarce), then the CNN will learn&pick out more features from those that are more abundant while trying to reduce the overall losses when optimization&gradient descent is done. In this case, as shown on the histogram below, there are way more examples of classes 1(speed 30) and 2(speed 50) compared to class 19(dangerous curve to the left)

![alt text][image1]

Below is an example image of index 16758 (just a number I chose) which I kept running every time my code is run to visually check if my preprocessing does actually well and not harm the data. The X_train is shuffled every time the entire code is run, so that the index 16758 will pick out a different image every time. 

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For pre-processing, I played around with a couple of methods. On my very first try, a simple grayscale (i.e. np.sum(X_train/3,axis=3,keepdims=True)) and simple normalization (subtracting by 128 and dividing by 128) were the only pre-processings I applied. Surprisingly, this provided validation accuracy of 91%. Trying non-gray-scaled images (with 3 channels) provided worse results so I stuck with 1 channel images. In order to go above the minimum 93% requirement, I had to try other methods. 

First of all, I used the cv2.cvtColor command (Y = 0.299 R + 0.587 G + 0.114 B instead of simple numpy average scheme presented above) to change to grayscale; but this method I believe had almost no effect on the accuracy. The improvement, if any, was so miniscule. 

Next, I tried different normalizations, such as changing the ranges or using zscale. I settled on using -0.5 to 0.5 range because it was the simplest and also the random initilization done via tf.truncatednormal method seems to output value ranges that are quite small (smaller than -0.5 to 0.5 range). I just wanted my image values and the weights to be on a somewhat similar range so that excessive loss during float computations do not harm my learning model. 

However, changing normalization values seemed to provide only miniscule benefits, if it all, again. So I went on to try using cv2.equalizeHist to improve some of the images that were so dark that nothing could be made out of them. While this method was able to salvage a lot of the dark photos, for some of the images, the results weren't that satisfactory; some images were over-adjusted and some of the good images were excessively brightened in some areas that numbers such as 100 or 50 for speed limits were being effaced. Hence, in looking for a better method, I discovered this CLAHE method on one of the cv2 documentation websites and saw that some of their sample results were very promising. CLAHE helped the images achieve better contrast and avoid over contrast adjustment (thus avoiding over-brightening effects). With this I was satisfied with how the images were adjusted and the accuracy of my learning model improved by about 2%, resulting in validation accuracy of around 93~94%. However, when run on the test set, it achieved 93% once but most of the time was sitting at around 92%.

Still not being able to meet the minimum requirements, I tweaked some of hyperparameters and model structure, which will be described in the sections below. In the end, I realized that no matter how much effort I put into finetuning the hyperparameters and grayscaling/normalization/image adjustments, I could not get significant improvement unless the skewed data problem was fixed and more data were available. So I went onto utilize keras module to generate more data that are rotated, zoomed, sheared, and shifted. Horizontal flips and and vertical flips were not applied as these won't happen in real life anyway. I generated more augmented data so that X_train will have equal number (3000) of images for each unique class. I tried 1000 at first, but it only provided 1~2% improvement on validation accuracy so I went on to increase the amount.

In summary, the final pre-processing was done in the order of grayscaling, augmenting, and finally normalizing.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I developed my model architecture based on LeNet. I increased the filter depth in both of my convolution layers as it approved my accuracy by about 1%.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, depth 12 outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 1x1 stride, valid padding, depth 32 outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Flatten	      | outputs 800      									|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Fully connected		| outputs 43        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the optimizer, I just went with the Adam optimizer which seems to be the popular choice. Unfortunately I did not have time to play around with different optimizers. As for learning rate, I kept 0.001 which was the value provided as an example for the LeNet exercise. As for batch size, I tried using 50, 86, 128, and it seemd like 50 was providing me the best result when the data were not augmented. However, after augmenting data, the choice 50 and 128 did not make much of a difference. I just went with 128 as it processed a bit faster. The accuracy constantly improved after every epoch except for one epoch where it dipped a little bit so it was a good indication that my model is not being overfitted. As for number of epochs, I was running this model on surface pro 1 model, which has a sub-par processor at best, so in order to reduce the time it takes to train, I just went with 10. I believe my model can reach higher percentages perhaps 98.5% on validation accuracy if I were to jack up the epochs to 50.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

As described in the previous sections, finding the best solution involved running the models many many times while trying different pre-processing techniques, tweaking hyperparameters, and changing depth of the LeNet convolution layers. The reason I went with the LeNet is because its code base was already provided and it required only simple tweaking. For other more advanced architectures, I did have a read on others codes as well as papers, but it seemed like, while they do provide better accuracy, none did the justice as much as "more augmented data" did. I took the coursera machine learning course and from there I learned that the easiest solution to solving accuracy problems is really "more data". And it seemed to be the case for this project as well. I was only able to achieve ~4% (biggest improvement) by going through data augmentation. I believe involving a more complicated architecture that involves some feed-forwards as well could improve the accuracy but only to a magnitude of about 1%.

My final model results were:
* training set accuracy of ? 99.1%
* validation set accuracy of ? 97.4%
* test set accuracy of ? 95%

* What architecture was chosen? 
LeNet

* Why did you believe it would be relevant to the traffic sign application? 
It was successfully used to identify digits from 0~9 successfully in one of the training exercises in the course. Digits contain lines and curves as do traffic signs, so I thought it should be able to identify traffic signs as well.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? 
With 95% accuracy on the test set, it is not be sufficient for an autonomous driving car. However, it did meet the requirement of this project. I did not spend too much time on getting to higher percentages as I believe that further teachings from this course will provide me with better tools to achieve what I believe is acceptable for actual autonomous cars.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because it is a relatively small sign compared to the whole frame and it is shifted a lot to the left. Unless there are lots of images provided in the training set as an example of this setting for this class, the model will likely misclassify this.

The second image should be classifiable since its zoom and the way it's facing the camera is pretty typical of the sample images in the training set.

The third image would be very difficult to classify because it is very small and shifted to the left.

The fourth one should be classifiable but there are some background noises which might hinder correct classification

The fifth one's sign has way too much background noises located below the sign after pre-processing that it will likely not be classified correctly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![alt text][image8]

The accuracy was a disappointing 20%. But these images had a lot of background noises that even after pre-processing, streaks of high intensity pixels were present aside from the sign's details. The test set accuracy was a satisfactory 95% and this 20% result against samples were definitely not satisfactory. More work will have to be put in later into pre-processing to remove background noises or perhaps utilizing hough transforms in combination to better extract features would help.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for computing top 5 softmax probabilities is located at the very bottom right before the optional exercise. The results are as follows, and my model makes an astoundingly confident assessment of each image, as evidently shown by the first choice's probability being significantly greater than the others (in fact, the other 4 for each image are so small that they are shown as 0.). The exact top 5 classes they predict seem quite questionable. I would have to investigate further why the 2nd~4th predictions are all identically 0,1,2 but it might have to do with the fact that speed signs were the most common out of all the training sets, and the CNN naturally applied weights/biases corresponding to features extracted from the speed signs, and forward propagating the images result in predictions skewed towards speed signs. Any feedback on what could be happening would be appreciated.

Prediction [False  True False False False]
Accuracy 0.20000000298023224
TOP 5 TopKV2(values=array([[ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.]], dtype=float32), indices=array([[23,  0,  1,  2,  3],
       [12,  0,  1,  2,  3],
       [23,  0,  1,  2,  3],
       [ 3,  0,  1,  2,  4],
       [23,  0,  1,  2,  3]]))

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I will try this optional exercise later after finishing all the projects for term1. I was already delayed on this project for about a month due to increase in workload at work.

