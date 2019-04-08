
# coding: utf-8

# In[2]:


# set work directory 
import os
os.getcwd()
os.chdir("D:/Work_Place/Tend_LSTM_Model/")

# load file
import pandas as pd # for Data processing 
df = pd.read_csv('2015_Ted.csv',delimiter=',',encoding='latin-1') # read file form database
df.head() # To check fiest five records of data frame 


# In[3]:


# split data into train and test 
from sklearn.model_selection import train_test_split  # import standared train, test split library 
train_cat, test_cat = train_test_split(df['Cpv(main)'], test_size = 0.2)  # Spliting categories/ lables into train and test 
train_text, test_text = train_test_split(df['Description'], test_size = 0.2)  # Spliting text data into train and test 


# In[4]:


train_cat.shape , test_cat.shape, train_text.shape, test_text.shape  # inspect the Shape and size of train, test split data


# In[5]:


from sklearn import preprocessing # importing standared library for text preprocessing 

# encoding the target variable ie. category 
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(train_cat.astype(str))
y_test = encoder.fit_transform(test_cat.astype(str))


# In[10]:


# cleanig and tokenizing text with keras
from keras_preprocessing import text
max_words = 10000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ',char_level=False, oov_token=None)
tokenize.fit_on_texts(train_text.astype(str)) # fit tokenizer to our training text data


# In[9]:


# creatin text_to_matrix for calculating term-frequency-inverse document frequency for training
x_train = tokenize.texts_to_matrix(train_text.astype(str), mode = "tfidf") 
x_train.shape


# In[11]:


# creatin text_to_matrix for calculating term-frequency-inverse document frequency for testing
x_test = tokenize.texts_to_matrix(test_text.astype(str), mode = "tfidf")
x_test.shape


# In[12]:


import numpy as np #importing numpy for linear algebric operations

# # Converts the labels to a one-hot representation
num_classes = np.max(y_train.astype(int)) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train.shape , y_test.shape


# In[13]:


# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# In[19]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Embedding


# In[20]:


# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset

batch_size = 32
epochs = 2-3
drop_ratio = 0.5
max_words=10000


# In[21]:


def Tend_LSTM():
    model = Sequential()
    model.add(Embedding(max_words, 1))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# In[22]:


model = Tend_LSTM()
model.summary()


# In[24]:


# train models with 2 epoch 
model.fit(x_train, y_train, batch_size=16, epochs=2)
score = model.evaluate(x_test, y_test, batch_size=32)
print('Testing score Loss: {:0.3f}  Accuracy: {:0.3f}'.format(score[0],score[1]))


# In[27]:


from keras.models import load_model
model.save("Tend_LSTM_model.h5")
print("model saved")
get_ipython().run_line_magic('notebook', '"D:/Work_Place/Tend_LSTM_Model/Tender_2015_LSTM_Model.ipynb"')


# <b>Thank You!</b>
