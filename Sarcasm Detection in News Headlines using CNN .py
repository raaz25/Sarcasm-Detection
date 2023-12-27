#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk


# In[2]:


import random
# Read the JSON file into a Pandas DataFrame
data = pd.read_json(r"E:\Raj Aryan\Dataset - Sarcasm Detection In Newsheadlines\Sarcasm_Headlines_Dataset_v2.json", lines=True)
# Shuffle the DataFrame
suffled_set = data.sample(frac=1, random_state=42)  # Shuffle all rows
# Define the split size
SPLIT_SIZE = 0.9
# Calculate the length for training set
train_length = int(len(suffled_set) * SPLIT_SIZE)
# Split the shuffled dataset into training and test sets
train_set = suffled_set[:train_length]
test_set = suffled_set[train_length:]


# In[3]:


#Data Summary
print('Total length of the dataset: ',len(suffled_set))
print('Length of the Training dataset: ',len(train_set))
print('Length of the Test dataset: ',len(test_set))


# In[4]:


#Data Format
print(train_set.iloc[100])


# In[5]:


#This list is available at https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords=[ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


# In[6]:


#Function to remove stopwords
def remove_stopwords(sentence):
    for word in stopwords:
        token=' '+word+' '
        sentence=sentence.replace(token,' ')
        sentence=sentence.replace('  ',' ')
        return sentence


# In[7]:


#Function to get the plain text from the link of an article
def get_article(link):
    try:
        from bs4 import BeautifulSoup
        import urllib.request

        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

        url = link

        headers={'User-Agent':user_agent,} 

        request=urllib.request.Request(url,None,headers) #The assembled request
        response = urllib.request.urlopen(request)
        html= response.read()
        soup = BeautifulSoup(html)

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return ' '.join(text.split())
    except:
        return 'article not available'
        pass


# In[8]:


def extract(data):
    headlines = []
    labels = []
    for index, row in data.iterrows():
        labels.append(row['is_sarcastic'])
        headlines.append(remove_stopwords(row['headline']))
    return headlines, labels

train_headlines, train_labels = extract(train_set)
test_headlines, test_labels = extract(test_set)


# In[9]:


#Extracting articles, headlines and labels from the data
#train_articles,train_headlines,train_labels=extract(train_set)
train_headlines,train_labels=extract(train_set)
#test_articles,test_headlines,test_labels=extract(test_set)
test_headlines,test_labels=extract(test_set)


# In[10]:


import numpy as np

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


# In[11]:


#Preprocessing the training and test headlines for the word embedding
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_headlines)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_headlines)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_headlines)
test_padded = pad_sequences(test_sequences,maxlen=max_length)


# CONVOLUTIONAL NEURAL NETWORK (CNN)

# In[12]:


#Building the CNN Model
model_conv = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(16,3,activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[13]:


model_conv.summary()


# In[14]:


model_conv.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epochs = 20
history=model_conv.fit(padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels))


# In[16]:


#Evaluating Accuracy and Loss of the model
get_ipython().run_line_magic('matplotlib', 'inline')
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) #No. of epochs

#Plot training and validation accuracy per epoch
import matplotlib.pyplot as plt
plt.plot(epochs,acc,'r',label='Training Accuracy')
plt.plot(epochs,val_acc,'g',label='Testing Accuracy')
plt.legend()
plt.figure()

#Plot training and validation loss per epoch
plt.plot(epochs,loss,'r',label='Training Loss')
plt.plot(epochs,val_loss,'g',label='Testing Loss')
plt.legend()
plt.show()


# In[ ]:




