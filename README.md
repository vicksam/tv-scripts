# TV Script Generator

I built this deep learning model to create new realistic screenplays based on the real ones. You can use it to generate an entirely new script.

### Overview

I design a neural network with LSTM layers. Then, during training, it learns how to predict the next word by looking at a few last words in a text. At the end, I use it to produce a completely new script, word by word.

### Methods

* RNN (Recurrent Neural Network)

### Tech

* Python
* PyTorch
* NumPy

### How to use

To take a look at the notebook, just click on `tv_scripts.ipynb` and it should open automatically.

You can also download the project and open the file `tv_scripts.html` to see the results in your browser.

## Dataset

I am using scripts from [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) that can be found on kaggle.

#### Size

It contains 109,233 lines of text and about 21,000 unique words.
