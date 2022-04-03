# Implementation of Neural Network

### Link to Colab of Assignment-1  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashwanth10/DL_Assignment_1/blob/master/assignment_1.ipynb)

----------------------------------------------------

<br>

# Description
This repository is done as part of learning **CS6910: Fundamentals of Deep Learning** course offered by Indian Institute of Technology, Madras. The course is taught by Prof. Mitesh Khapra. 

This repository consists of code (Jupiter Notebook) of Assignment-1. The objective of this assignment was to train a neural network to classify Fashion-MNIST images across 10 class labels. To get the insights of our model development and draw some inferences we used wandb. Wandb helped us to get better visualization of data, compare different models along with their accuracies for a large number of models. This enabled us to make good inferences and pick the best models.

Below are the algorithms we implemented:

1. Feedforward Neural Network
2. Backpropagation Algorithm
3. Gradient Descent's varients like:

    ```
    a) Stochastic Gradient Descent
    b) Momentum Based Gradient Descent
    c) Nesterov Accelerated Gradient Descent
    d) RMSProp
    e) Adam
    f) Nadam
    ```
<br>

# Libraries #
1. We have used numpy for most of the mathematical operations in Forward, Back propagation algorithm and loss function computation.
2. Tensorflow and keras for getting the Fashion-MNIST and MNIST datasets.
3. Wandb for making inferences of our models.
4. Pandas for plotting confusion matrix.

<br>

# Run the Colab #
1. Open the colab by clicking the above button "Open in Colab"
2. Click on RunTime > Run All to run the notebook

<br>

# How to train model? #
The code has been written using functions to easily add new optimizers and reuse the code as much as possible. To train the model, the project makes a call to `train()` with a config file. The parameters used in the config are the following:

```
neurons_hidden_layer    - Number of neurons in each hidden layer
number_of_epochs        - Number of iterations trained over the entire data
activation              - The activation function used for activation.
                          E.g: sigmoid, tanh, relu.
no_of_hidden_layer      - Number of hidden layers in the network.
                          E.g: 2, 3.
batch_size              - Number of data points for a gradient update. 
                          E.g: 16, 32, 64.
optimiser               - Variant of Gradient Descent used for 
                          optimization.
                          E.g: adam, rmsprop, nadam, sgd,momentum, nesterov.
initialization          - Weights initialization during network initialization.
                          E.g: "random" or "xavier"
loss                    - Loss function used for optimization. 
                          E.g: "crossEntropy" or "squaredError".
learning_rate           - Learning rate of the optimizer.
                          E.g: 0.1
```

In order to train a model, we define the above config and create a wandb sweep and call the method `train()`.

<br>

# How to initialize network structure? #
The method `initialize_network()` gives the modularity to create a neural network based on our requirement. The parameters of the method are:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **n_inputs** - Dimension of the input layer <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **n_hidden** - Number of hidden layers <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **n_outputs** - Number of neurons in output layer <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **n_layers** - Number of layers <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **config** - Config parameters for weight initialization and setting activation function.<br/>

The method `initialize_network()` on invoking returns a network object with the above configuration. 

<br>

# Optimisers #

As mentioned in the above config parameters, optimiser is a varient of Gradient Descent. Below are the optimisers and their hyper-parameters used internally:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **optimiser_sgd** - learning_rate<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **optimiser_mgd** - learning_rate, gamma<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **optimiser_nagd** - learning_rate, gamma<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **optimiser_rmsprop** - learning_rate, beta1, eps<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **optimiser_adam** - learning_rate, beta1, beta2, eps<br/>

Also, each of the above methods have the function parameters:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **network** - Network object created during network initialization.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **config** - Config object to customize hyper-parameters.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **x_train** - training datapoints<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **targs** - one-hot encoding of classification of training data.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **x_val** - validation datapoints<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **targs_val** - one-hot encoding of classification of validation data.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **x_test** - testing datapoints<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **targs_test** - one-hot encoding of classification of testing data.<br/>

Optimiser function function trains the model with training data and prints the accuracy of the validation and testing data. Each optimiser has internally three main methods i.e. `forward propagation, backward propagation and update` to train the model.

<br/>

# Forward Propagation #
Performs forward propagation on the data vector through the network. `forward_propagate` or `forward_propagate_nagd` is the method used across all the optimisers to perform forward propagation.

Function parameters are: <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **network** - Network in which to perform forward propagation<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **X** - Single normalized data vector (28 x 28) image is converted to a vector of (784 x 1).<br/>

<br/>

# Back Propagation #
Performs back propagation in the network and calculates the gradients. `backward_propagation_*` are the backward propagation methods, where * depends on the optimiser used.

Function parameters are: <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **network** - Network in which to perform back propagation<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **actual** - Actual classification of an image (one-hot encoding)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **predicted** - Probability of the predicted class of an image (vector of probabilities).<br/>

<br/>

# Updating network weights and bias #

After performing the back propagation, depending on the batch size given, the code updates the weights and bias. `update_*` are the update methods, where * depends on the optimiser used.

<br/>

# Flow of the Model Development #

The given dataset is converted to right format before training. We converted each 28 x 28 pixel image to 784 x 1 vector and normalized the vector. The classification of each image is also converted to one-hot encoding.

For training the model, we call `train()` along with the config. This method initiates the wandb sweep and creates a run for keeping track of the model performance. The run starts by initializing the network with the parameters. Depending on the optimiser selected, the model is trained using training data by calling forward propagation, backward propagation and update methods. Once the epochs are completed, the model calculates the accuracy with validation data as well as with the test data. After running many sweeps, we filtered the best model based on accuracy. Finally, the parameters of best model is used to build a model and classify the test data. The confusion matrix is also plotted for the same test data with the best model.

<br/>

# How to add new optimiser in future? #

The code has been well written in functions to reuse the code as much as possible. To add a new optimiser, we just need to define backpropagation, update methods of the new optimiser. `config` parameters (dictionary) can be tuned with different hyper-parameters and try out the new optimiser as required. The `train()` functional creates a network on the fly based on `config` values and can also call the new optimiser. After the model is trained with new optimiser, we can test the accuracy of the model by calling the function `calculate_loss_and_accuracy()`.

Many functions like `forward_propagation()`, `initialize_network()`, etc., can be used for the new optimiser without any changes.

<br/>

# Link to Report #
The wandb report is available [here](https://wandb.ai/cs21m010-cs21m041/DL_Assignment_1/reports/Assignment-1--VmlldzoxNTk0Njg1).

<br/>

# Acknowledgements #
1. For the most of project we referred the lecture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
2. https://github.com/ 
3. https://wandb.ai
4. http://www.cse.iitm.ac.in/~miteshk/CS6910/Slides/Lecture4.pdf
5. http://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Handout/Lecture5.pdf