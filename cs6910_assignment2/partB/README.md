# Part B: Fine-tune a pre-trained model

### Link to Colab of Assignment-2 Part-B
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashwanth10/Deep_Learning/blob/master/cs6910_assignment2/partB/assignment_2_part_b.ipynb)

----------------------------------------------------

<br>

# Description
Using the existing pre-trained models like InceptionV3, InceptionResNetV2, ResNet50 and Xception we can fine-tune the naturalist data. This helps us to learn model quickly without training the model from scratch.

<br>

# Libraries #
1. We have used numpy for most of the mathematical operations while playing around with the data.
2. Tensorflow and keras for training the models and getting the pre-trained models.
3. Wandb for making inferences of our models.
4. matplotlib for plotting the plots.
5. sklearn for splitting the training data and validation data.

<br>

# Run the Colab #
1. Open the colab by clicking the above button "Open in Colab"
2. Click on RunTime > Run All to run the notebook

# How to USE? #
The code has been well written in functions to reuse the code as much as possible.
The notebook file `assignment-2-part-b.ipynb` contains the `build_model()` which takes in a config file as argument and constructs a base model using a pre-trained model. 
The paremeters in the config file are:<br/>
arg1 : pretrained_model  : We can pass values like 'resnet50','inceptionresnetv2', 'inceptionV3', 'xception' <br />
arg2 : freeze_before : Number of layers before freeze. We can pass values like [50,70,100] <br />
arg3 : no_of_neurons_in_dense : It tells the number of layers in the dense layer. We can pass values like [64, 128, 256, 512].

The model is then trained using the method `model.fit()`. with WandBCallback to log the results.</br>

</br>
We are resizing the images to a size of 400 x 400 and then rescaling the image by a factor of 1/255 during the loading of the data. This helps us to stay in line with the input arguments of the pre-trained model.

<br/>

# Link to Report #
The wandb report is available [here](https://wandb.ai/cs21m010-cs21m041/DL_Assignment_2/reports/Assignment-2--VmlldzoxNzc3OTU0).

<br/>

# Acknowledgements #
1. For the most of project we referred the lecture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
2. https://github.com/ 
3. https://wandb.ai
4. http://www.cse.iitm.ac.in/~miteshk/CS6910/Slides/Lecture11.pdf
5. http://www.cse.iitm.ac.in/~miteshk/CS6910/Slides/Lecture12.pdf
