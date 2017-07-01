#**Traffic Sign Recognition** 

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images/0.jpg "Traffic Sign 1"
[image5]: ./images/1.jpg "Traffic Sign 2"
[image6]: ./images/2.jpg "Traffic Sign 3"
[image7]: ./images/3.jpg "Traffic Sign 4"
[image8]: ./images/4.jpg "Traffic Sign 5"
[image9]: ./images/5.jpg "Traffic Sign 6"
[image10]: ./images/6.jpg "Traffic Sign 7"
[image11]: ./images/7.jpg "Traffic Sign 8"
[image12]: ./images/8.jpg "Traffic Sign 9"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.
Please refer to the notebook to see images

1) one image per class
2) class distribution in all data sets - with histogram

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Pre Process Steps:
1) I decided to convert the images to grayscale after viewing the dataset and training with color and without, the color in general didn't help and the model size is much smaller without it.

2) I normalized the image data to zero mean in order to improve numerical precision, it makes the algorithm converge faster.

3) I decided to generate additional data for the model to generalize better since we need large dataset and most traffic sign in the dataset are positioned the same(scale, centered, no rotation).
I added randomly augmented data within some boundaries that my dataset will be more scale/rotation/translation invarient.
Limits: rotation - +-15 Degrees, Scale - up to 15% scale up/down, translation according to scale.
I tried to scale more, but it didn't manage to predict small traffic signs correctly and it destroyed my accuracy on the dataset, the images have low resolution, which doesn't let me scale too much.

Please refer to the notebook to see the augmentation effect

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

7 Layers - 3 Convlutional Layers, 4 Fully Connected, Xavier Initializer + Batch Normalization + Relu Activation for each layer, Dropout regularization of 50% for fully connected layers.

1) Input Layer - 32x32x1(grayscale)

2) Convlutional Layer -
    Convolution -
        Input - 32x32x1,
        Weights Initializaion - Xavier,
        Filter - 5x5,
        Stride - 1x1,
        Padding - Same,
        Output - 32x32x64

    Normalization - Batch Normalization

    Activation - Relu
    
3) Convlutional Layer -
    Convolution -
        Input - 32x32x64,
        Weights Initializaion - Xavier,
        Filter - 5x5,
        Stride - 1x1,
        Padding - Valid,
        Output - 28x28x64

    Normalization - Batch Normalization

    Activation - Relu

    Max Pooling -
        Input - 28x28x64,
        Stride - 2x2,
        Padding - Valid,
        Output - 14x14x64
        
4) Convlutional Layer -
    Convolution -
        Input - 14x14x64
        Weights Initializaion - Xavier,
        Filter - 5x5,
        Stride - 1x1,
        Padding - Valid,
        Output - 14x14x100

    Normalization - Batch Normalization

    Activation - Relu

    Max Pooling -
        Input - 14x14x100
        Stride - 2x2,
        Padding - Valid,
        Output - 5x5x100
    
5) Flatten -
    Input - 5x5x100,
    Output - 2500
    
6) Fully Connected Layer -
    Input - 2500,
    Weights Initializaion - Xavier,
    Normalization - Batch Normalization,
    Activation - Relu,
    Dropout - 0.5,
    Output - 500
    
7) Fully Connected Layer -
    Input - 500,
    Weights Initializaion - Xavier,
    Normalization - Batch Normalization,
    Activation - Relu,
    Dropout - 0.5,
    Output - 250
    
8) Fully Connected Layer -
    Input - 250,
    Weights Initializaion - Xavier,
    Normalization - Batch Normalization,
    Activation - Relu,
    Dropout - 0.5,
    Output - 100
    
9) Fully Connected Layer -
    Input - 100,
    Weights Initializaion - Xavier,
    Output - n_classes
    
Softmax With Cross Entropy

Stochastic Gradient Descent With Adam Optimizer -
    Batch Size - 512,
    Learning Rate - 0.0001


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, my first priority was to minimze hyperparameters.

Xavier Initializer - initialize weights so they won't be too small or too large for backpropagation,
it works realy good in practice and it elimanated 2 hyper parameters for me.

Learning Rate - trail and error - needed a good learning rate so I can converge low(not fast).
Followed this advice: http://cs231n.github.io/assets/nn3/learningrates.jpeg

Batch Size - 512 - trail and error - higher numbers made the learning more stable, 512 was the memory limit.

Number Of Epchos - trail and error, increased it when I decreased learning rate, converge slower but lower.

Stochastic Gradient Descent Optimizer - Adam Optimizer - it is the recommended one and it worked better in practice.
Also, it decays the learning rate automatically so I didn't have to decay by myself which includes more tuning.

Relu activation - recommended non linear activation function.

Batch Normalization - makes learning faster and acts as a regulizer, it gave me better results on the validation set, not by much though.

Dropout - Used it only on fully connected layer, acts as a regulizer to prevent overfitting, does a good job with the recommended 50% drop for all layers.

be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 98.6% 
* test set accuracy of 96.9%

Convlutional networks are the best method for learning invariant and absract feature hirerachies from labeled image data. 

I started with the lenet 5 architecture, as it was running on 32x32 images, and it's a general model for convolutional networks which work well on image classifcation.

I started with some recommended general configuration - learning rate of 0.001, 128 batch size, image normalization, recommended weights initilization hyper parameters, relu activation after each layer, they were all a good starting point.

The model was underfitting at first, since the covolutional layers depth was too low.
My first adjusment to it was to increase the depth of the 2 convolution layers(and fully connected layers as well ofcourse) which yielded much better results for training and validation set, this had the largest improvment by far in terms of accuracy on all data sets.

Then I added another fully connected layer which had good increase in accuracy as well.

After doing that I still didn't get the results I wanted on the validation set and the training set was much higher, so I was overfitting now.
I didn't get the results I wanted on the training loss as well(probablities accuracy) which influences the accuracy indirectly.

To address the overfitting, I started with adding synthetic data to the model, so the model will be more scale/translation/rotation invarient and more data is mostly always a blessing, it gave me a good boost in validation accuracy and training loss.

Then I added dropout with 50% dropout on the fully connected layers to address the overfitting, it really helped.
You can see the effect of the dropout in the low training accuracy and the difference from the validation accuracy at the begininig of the learning process.

With only these changes I started to get really good results, but ofcourse I still had some room for improvement.

Adjusted the learning rate, implemented step decay, but then I read the adam optimizer is doing it for me, so I stopped it, the best learning rate was 0.0001, gave a good learning curve as you can see in the notebook.

Adjusted Batch size to be 512, couldn't go higher as I was exhausting memory, it made the learning process more stable.

In my efforts to have less hyper parameters, I came acrros xavier initilizar which had good results and didn't have to tune anything.

I read about batch normalization, implemented it after each layer before the activation, and it did show increased performance on the learning process.

I was trying to have the best trade off with respect to performance with adding convolution layers without down sampling, after a few tests I decided to add one such layer for the first layer with same padding to keep the original image size at first layer.

The training, validation, test sets all have high accuracy, the model is very certain(high probability on predicatd class) for most images, the recall and precision are very high for most classes, it means the model is succesful at predicting german traffic signs(at least centerd with specific scale ones).

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 9 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]

The only image that should be hard to classify is the last one, since the sign is not centered and scaled like our dataset.
The others may suffer from high probablities on similiar classes, like "No passing" and no "No passing for vehicles over 3.5 metric tons" which share resemblance.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88.88%. The test set accuracy is higher.

My last image is not a normal image in comparison to the dataset, it is not centered and scaled like the rest.
Since my images are only a sample, there is no point in comparing accuracies since each image has high influence on the final accuracy.

The model has high precision and recall on the wrong predication class, thus it's not a problem of the specific class.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

As you can see in the notebook, all the correct predictions have high certainty, predicated classes have really high probability, close to 1. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Mostly we can see the network learns the sign boundaries(shape), the sign inner details and ignores noise.
You can see it clearly on the first convolutional layer.
