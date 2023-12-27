#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk


# ## Reading and cleaning the data

# In[2]:


data = pd.read_json(r"E:\Raj Aryan\Dataset - Sarcasm Detection In Newsheadlines\Sarcasm_Headlines_Dataset_v2.json", lines=True) #The lines=True argument indicates that the file contains a JSON object per line.
data.head()


# In[3]:


# shape of the data
data.shape


# In[4]:


#To check the total number of words present in the headline 
data_len = data['headline'].apply(lambda x: len(x.split(' '))).sum()
print(f'We have {data_len} words in the headline')


# In[5]:


# check the columns names
data.columns


# In[6]:


# check the data types in the columns
data.dtypes


# In[7]:


#checking the unique values in 'is_sarcastic' column
data.is_sarcastic.unique()


# In[8]:


import warnings 
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[9]:


#checking the value counts in 'is_sarcastic' column
labels = ['Sarcasm', 'Not Sarcasm']
values = [data['is_sarcastic'].sum(), len(data) - data['is_sarcastic'].sum()]

plt.ylabel('Number of articles')
plt.title('Number of Sarcastic Articles')

p1=plt.bar(labels,values,color=['g','b'])
for bar in p1:
    height = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2
    y = height * 1.01
    plt.annotate(str(height), (x, y), ha='center')
    
plt.show()


# In[10]:


# check the null values in data
data.isna().sum() 


# In[11]:


#drop 'article_link' column
data = data.drop('article_link', axis=1)


# In[12]:


#ckeck the data
data.head(10)


# In[13]:


#import necessary library
import re
from nltk.corpus import stopwords

set_stopwords = set(stopwords.words("english"))            #'is','the'


def clean_txt(text): # define the fuction with tokenization/string cleaning for all datasets 
                        
    text = re.sub(r"[^A-Za-z,!?]", " ", text)     
    text = re.sub(r'\[[^]]*\]'," ", text) 
    text = re.sub(r"\'s", "", text) 
    text = re.sub(r"\'t", "", text ) 
    text = re.sub(r"\'re", "",text) 
    text = re.sub(r"\'d", "", text) 
    text = re.sub(r"\'ll", " ",text) 
    text = re.sub(r",", " ", text) 
    text = re.sub(r"\(", " ", text) 
    text = re.sub(r"\)", " ", text) 
    text = re.sub(r"\'", " ", text)
    text = re.sub(r"aa", "", text)
    text = re.sub(r"zz", "", text)
    text = re.sub(r"[0-9]", ' ', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in set_stopwords)
    return text

data['headline'] = data['headline'].apply(clean_txt)


# ## Most common words

# In[14]:


from collections import Counter #import Counter for finding most common words
import seaborn as sns #import searbon for vizualization result

text = data['headline']
words = text.str.split(expand=True).unstack()
result_count = Counter(words).most_common()
result_df = pd.DataFrame(result_count).reset_index().drop(0) #converting to Dataframe and drop the Nones values
#result_df
#vizualize result
sns.set_theme(style="whitegrid") 
f, ax = plt.subplots(figsize=(15, 15)) 
sns.barplot(y=result_df[0][0:30], x=result_df[1][0:30], data=result_df, palette=None)
plt.ylabel('Words', color="blue")  # Add an x-label to the axes.
plt.xlabel('Count', color="blue")  # Add a y-label to the axes.
plt.title("Frequent Occuring words in Headlines", color="blue") 
plt.xticks(rotation=50)
ax.tick_params(axis='x', colors='black')
plt.show()


# In[15]:


# Finding most common words in 'is_sarcastic' column


# In[16]:


#create DataFrame for sarcastic words 
sarcastic = pd.DataFrame(data[data['is_sarcastic']==1]['headline'].str.split(expand=True).unstack().value_counts()).reset_index()


# In[17]:


#create DataFrame for non_sarcastic words 
non_sarcastic = pd.DataFrame(data[data['is_sarcastic']==0]['headline'].str.split(expand=True).unstack().value_counts()).reset_index()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

#Sarcastic words in Headlines
sns.set_theme(style="whitegrid") 
f, ax = plt.subplots(figsize=(15, 10)) 
sns.barplot(y=sarcastic['index'][:30], x=sarcastic['count'][:30], data=sarcastic[:30], palette=None)
plt.ylabel('Words', color="blue")  # Add a y-label to the axes.
plt.xlabel('Count', color="blue")  # Add an x-label to the axes.
plt.title("Frequently Occurring Sarcastic Words in Headlines", color="blue") 
plt.xticks(rotation=70)
plt.show()


# In[19]:


#Non-Sarcastic Words in Headlines
sns.set_theme(style="whitegrid") 
f, ax = plt.subplots(figsize=(15, 10)) 
sns.barplot(y=non_sarcastic['index'][:30], x=non_sarcastic['count'][:30], data=non_sarcastic[:30], palette=None)
plt.ylabel('Words', color="blue")  # Add an x-label to the axes.
plt.xlabel('Count', color="blue")  # Add a y-label to the axes.
plt.title("Frequently Occurring Non-Sarcastic Words in Headlines", color="blue") 
plt.xticks(rotation=70)
plt.show()


# ## WordCloud Vizualization with StopWords

# In[20]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')

sarcasctic_2 = [every_word.lower() for every_word in sarcastic['index']]

sarc_nonstop = [word for word in sarcasctic_2 if word not in stopwords]

non_sarcasctic_2 = [every_word.lower() for every_word in non_sarcastic['index']]

non_sarc_nonstop = [word for word in non_sarcasctic_2 if word not in stopwords]


# In[21]:


from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize=(15,10))
wordcloud = WordCloud(width=1000, height=500,
                      max_words=300, min_font_size = 10,
                      background_color="black",
                      stopwords = stopwords, 
                      ).generate(' ' .join(word for word in sarc_nonstop))

plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Wordcloud of Sarcactic Words', color="black")
plt.axis("off")
plt.show()


# In[22]:


from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize=(15,10))
wordcloud = WordCloud(width=1000, height=500,
                      max_words=300, min_font_size = 10,
                      background_color="black",
                      stopwords = stopwords,
                      ).generate(' ' .join(word for word in non_sarc_nonstop))

plt.imshow(wordcloud, interpolation='spline36')
plt.title('Wordcloud of Non_Sarcactic Words', color="black")
plt.axis("off")
plt.show()


# ## Split text to train and test

# In[23]:


from sklearn.model_selection import train_test_split # import library for train_test_split
X = text
y = data.is_sarcastic
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)


# In[24]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[25]:


# Vectorize the text data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Transform the count matrix to TF-IDF representation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


# In[ ]:





# In[26]:


'''
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize the text data with TF-IDF including N-grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf_ngram = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf_ngram = tfidf_vectorizer.transform(X_test)
'''


# ## Multinomial Naive Bayes Classifier

# In[27]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# In[28]:


# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)


# In[29]:


# Predict on the test data
y_pred = classifier.predict(X_test_tfidf)


# In[30]:


# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# ## Logistic Regression Classifier

# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# In[32]:


# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)


# In[33]:


# Predict on the test data
y_pred = classifier.predict(X_test_tfidf)


# In[34]:


# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# ## Decision Tree Classifier

# In[35]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# In[36]:


# Train a decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train_tfidf, y_train)


# In[37]:


# Predict on the test data
y_pred = classifier.predict(X_test_tfidf)


# In[38]:


# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# ## KNeighborsClassifier

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# In[40]:


# Train a KNN classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_tfidf, y_train)


# In[41]:


# Predict on the test data
y_pred = classifier.predict(X_test_tfidf)


# In[42]:


# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_json(r"E:\Raj Aryan\Dataset - Sarcasm Detection In Newsheadlines\Sarcasm_Headlines_Dataset_v2.json", lines=True) #The lines=True argument indicates that the file contains a JSON object per line.
data.head()


# In[ ]:


import json
import csv

# Load JSON data from a file
with open("E:\Raj Aryan\Dataset - Sarcasm Detection In Newsheadlines\Sarcasm_Headlines_Dataset_v2.json", 'r') as json_file:
    data = json.load(json_file)

# Specify the fields you want to extract
fields_to_extract = ['headline','is_sarcastic']  # Replace with your field names

# Open a TSV file for writing
with open('output.tsv', 'w', newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter='\t')

    # Write the header row with field names
    tsv_writer.writerow(fields_to_extract)

    # Extract and write data from the JSON to the TSV file
    for item in data:
        row = [str(item[field]) if field in item else '' for field in fields_to_extract]
        tsv_writer.writerow(row)


# In[4]:


import pandas as pd

# Load the JSON data
data = pd.read_json("E:/Raj Aryan/Dataset - Sarcasm Detection In Newsheadlines/Sarcasm_Headlines_Dataset_v2.json", lines=True)

# Create a new DataFrame with the required format
tsv_data = pd.DataFrame({'sentence': data['headline'], 'label': data['is_sarcastic']})

# Save the DataFrame as a TSV file
tsv_data.to_csv("output.tsv", sep='\t', index=False)


# In[7]:


import pandas as pd

# Load the JSON data
data = pd.read_json("E:/Raj Aryan/Dataset - Sarcasm Detection In Newsheadlines/Sarcasm_Headlines_Dataset_v2.json", lines=True)

# Create a new DataFrame with the required format
tsv_data = pd.DataFrame({'sentence': data['headline'], 'label': data['is_sarcastic']})

# Specify the full file path where you want to save the TSV file on your local machine
output_file_path = "E:/Projects/output.tsv"  # Specify the full file path

# Save the DataFrame as a TSV file to the specified path
tsv_data.to_csv(output_file_path, sep='\t', index=False)

print("TSV file saved to:", output_file_path)


# In[ ]:




