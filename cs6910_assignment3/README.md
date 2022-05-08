# Assignment - 3

## Sequence to Sequence model (transliteration)

In this assignment, we transliterated Telugu words written in English script to Telugu script by implementing in both vanilla seq2seq transliteration and attention transliteration.

<br/>

### Libraries used:

1. Tensorflow and keras for training the models.
2. Pandas for csv data handling. 
3. matplotlib and mpl_toolkits library was used for plotting the grid for the predicted sentences and visualise them.
4. We have used numpy for most of the mathematical operations while playing around with the data.
5. wandb to find insightful observations and the best hyperparameter configuration and the plots.

<br/>

# Installations: #
1. pip package manager is used to install the required libraries

<br/>

# Dataset Used #
We downloaded the Dakshina dataset present in google drive using the below command:

```
!wget "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
```

The Dakshina dataset is a collection of text in both Latin and native scripts for 12 South Asian languages. For each language, the dataset includes a large collection of native script Wikipedia text, a romanization lexicon which consists of words in the native script with attested romanizations, and some full sentence parallel data in both a native script of the language and the basic Latin alphabet.

<br/>

# Code #

The code can be run either from the command line or as a colab notebook. For running in google colab ensure that the file `nirmala.ttf` is uploaded to the location `/content/drive/MyDrive/fonts/nirmala.ttf`

For running through command prompt (update the location of nirmala.ttf appropriately):
```
python assignment_3_no_attention.py
python assignment_3_attention.py
```

### Link to Colab of Assignment-3 Vanilla Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashwanth10/Deep_Learning/blob/master/cs6910_assignment3/assignment_3_no_attention.ipynb)


### Link to Colab of Assignment-3 Attention Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashwanth10/Deep_Learning/blob/master/cs6910_assignment3/assignment_3_attention.ipynb)

### Link to Colab of Transformers
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashwanth10/Deep_Learning/blob/master/cs6910_assignment3/transformers/transformers-gpt2.ipynb)

<br/>

# How to USE? #
The entire project has been modularised using functions to make it as scalable as possible for future developments and extensions.

The main class for the model is `CustomRNN` which has the necessary functions to train and test the model. The parameters for the constructor of this class are the following:
input_embedding_size, cell_type='GRU', hidden_layer_size=32,
                 num_encoder_layers=2, num_decoder_layers = 2, dropout=0.1, 
                 batch_size=32, epochs=25, is_test_model=False

arg1 : input_embedding_size  :  size of the input embedding<br/>
arg2 : cell_type  : To specify whether the RNN used is LSTM or GRU or SimpleRNN <br/>
arg3 : hidden_layer_size  : Number of neurons in the hidden dense layer <br/>
arg4 : num_encoder_layers  : number of encoder layers used in model <br/>
arg5 : num_decoder_layers  : number of decoder layers used in model<br/>
arg6 : dropout  :  Dropout of the model <br/>
arg7 : batch_size  : Batch size used for training the model <br/>
arg8 : epochs  : number of epochs used during model training<br/>
arg9 : is_test_model  : whether the model is being used for computing validation accuracy or test accuracy.<br/>

<br/>

## How to train the model? ##
We added a method called `training(self)` in the class which will train the model using the dakshina training dataset along with the parameters specified in the constructor. This function internally has `model.compile()` and `model.fit()` as below:

```
model.fit([encoder_input_data, decoder_input_data],
           decoder_target_data,
           batch_size=self.batch_size,
           epochs=self.epochs,
           callbacks=[WandbCallback()])
```
This trains the model and logs the training accuracy in wandb.

<br/>

## How to inference the model? ##

Once the model is trained we call the method `inference_model(self)` which takes model, encoder layers and decoder layers as input and returns the encoder_model and decoder_model. 

```
model_rnn_obj.inference_model(model, encoder_layers, decoder_layers)
```

<br/>

## Decode sequence and Calculating Accuracy ##

Using the encoder_model and decoder_model for an input sequence either in validation data or test data we call the method `decode_sequence()` which returns the decode sequence in target language. 

In `test_and_calculate_accuraccy()` by iterating over all validation/test data samples we invoke the method `decode_sequence()` and compare the obtained predicted sequence with the target sequence at the word level. If both matches then we consider that the model predicted the sequence correct. Using this, we compute the accuracy and log it to wandb.

<br/>

# Attention Mechanism #

Majority of the code remains the same in both vanilla seq2seq and attention models except for a few changes. In `training(self)`, we added `AdditiveAttention` layer before the dense layer and then trained the model. Similarily, in `decode_sequence` we make note of attention_weights which will be later used for visualization of heat maps.

<br/>

# Visualizations # 

In no_attention model we represented 16 samples along with their English words, predicted telugu words and actual telugu words using word cloud. 

In summary table, for each character we showed a count of a letter occurance and their correct prediction percentage which  helps to see which letters were mostly wrong.

We also took around 20 samples and showed their predicted and actual Telugu values in a table. 

In attention model we showed the attention heat maps for a few samples. Later, we showed the visualization to see which is being decoded in target word based on the given input english word.

<br/>

# Transformers #

For the majority of the transformers code, we reffered to the blog. The code is present in transformers-gpt2.ipynb file in transformers folder. We used the dataset `gloom_index.xlsx` for generating song lyrics which is downloaded from the data world website. After running the above notebook, we get `generated_lyrics.txt` for the text 'I love Deep Learning'.

<br/>

## Note ##
While running the notebook or while running notebook, please add nirmala.ttf to your google drive in the location `/content/drive/MyDrive/fonts/nirmala.ttf` . This is used for font properties of the Telugu Language.

<br/>

# Link to Report #
The wandb report is available [here](https://wandb.ai/cs21m010-cs21m041/DL_Assignment_3_a/reports/Assignment-3--VmlldzoxOTY0MzUw).

<br/>

# Acknowledgements #
1. For the most of project we referred the lecture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
2. https://github.com/ 
3. https://wandb.ai
4. http://www.cse.iitm.ac.in/~miteshk/CS6910/Slides/Lecture13.pdf
5. http://www.cse.iitm.ac.in/~miteshk/CS6910/Slides/Lecture14.pdf
6. http://www.cse.iitm.ac.in/~miteshk/CS6910/Slides/Lecture15.pdf
7. https://github.com/google-research-datasets/dakshina
8. https://distill.pub/2019/memorization-in-rnns/#appendix-autocomplete
9. https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff
10. https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a
11. https://www.aclweb.org/anthology/2020.lrec-1.294
12. https://data.world/rcharlie/gloom-index-of-radiohead-songs/workspace/file?filename=gloom_index.csv