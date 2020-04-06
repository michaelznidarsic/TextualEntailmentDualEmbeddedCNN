# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:39:33 2020

@author: mznid
"""

import pandas as pd
import numpy as np
import os
import random
import math
import itertools


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import keras   
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, AveragePooling2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding,GRU, LSTM,Bidirectional,TimeDistributed,AveragePooling1D,GlobalMaxPool1D,Reshape,Input,Concatenate,concatenate,Attention,Permute, Lambda,RepeatVector,Add
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import seaborn as sb

import nltk
from gensim.models import Word2Vec
from string import punctuation

# UNLIST GPU OPTION FOR TENSORFLOW 2.1. GPU RUNS OUT OF MEMORY WITH THIS NN MODEL AND EVEN RUNS SLOWER THAN CPU 

tf.config.set_visible_devices([], 'GPU')


import tensorboard
import datetime

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# LOAD AND CLEAN

raw = pd.read_csv("D:\\stanford-natural-language-inference-corpus\\snli_1.0_train.csv")
rawtest = pd.read_csv("D:\\stanford-natural-language-inference-corpus\\snli_1.0_test.csv")
rawdev = pd.read_csv("D:\\stanford-natural-language-inference-corpus\\snli_1.0_dev.csv")




#for ind,each in enumerate(sentences2):
#    if type(each) == float:
#        print(ind)
nullset = {91479,91480,91481,311124,311125,311126}        
nonnullset = set(range(0,len(raw.iloc[:,0]))) - nullset
raw = raw.iloc[sorted(list(nonnullset)),:]

drawlist = []
for ind, each in enumerate(raw.iloc[:,0]):
    if each == '-':
        drawlist.append(ind)
        
raw = raw.iloc[sorted(list((set(range(0,len(raw.iloc[:,0]))) - set(drawlist)))),:]


drawlist = []
for ind, each in enumerate(rawtest.iloc[:,0]):
    if each == '-':
        drawlist.append(ind)
        
rawtest = rawtest.iloc[sorted(list((set(range(0,len(rawtest.iloc[:,0]))) - set(drawlist)))),:]


drawlist = []
for ind, each in enumerate(rawdev.iloc[:,0]):
    if each == '-':
        drawlist.append(ind)
        
rawdev = rawdev.iloc[sorted(list((set(range(0,len(rawdev.iloc[:,0]))) - set(drawlist)))),:]




#########################################################################

# DOWNSAMPLE

#sampleindex = random.sample(range(0,len(raw.iloc[:,0])), round(0.05 * len(raw.iloc[:,0]))) # 0.05
#raw = raw.iloc[sampleindex,:]
#raw.index = range(0,len(raw.iloc[:,0]))

#########################################################################


sentences1 = list(raw.iloc[:,5])
sentences2 = list(raw.iloc[:,6])
labellist = list(raw.iloc[:,0])
labelset = set(labellist)

labellisttest = list(rawtest.iloc[:,0])
sentences1test = list(rawtest.iloc[:,5])
sentences2test = list(rawtest.iloc[:,6])
labellistdev = list(rawdev.iloc[:,0])
sentences1dev = list(rawdev.iloc[:,5])
sentences2dev = list(rawdev.iloc[:,6])



# HANDLE LABELS


labeltrain = np.array(labellist)
labeltest = np.array(labellisttest)
labeldev = np.array(labellistdev)

labelkey = list(labelset)
labelvalue = list(range(0, len(labelkey)))
labeldict = {'contradiction': 0, 'entailment': 1, 'neutral' : 2}



ltrainn = []
ltestn = []
for each in labeltrain:
  ltrainn.append(labeldict[each])
for each in labeltest:
  ltestn.append(labeldict[each])
labeltrainhot = to_categorical(ltrainn)
labeltesthot = to_categorical(ltestn)

ldevn = []
for each in labeldev:
    ldevn.append(labeldict[each])
labeldevhot = to_categorical(ldevn)



# PARSE SENTENCES WITH NLTK pos_tag TO CREATE SET OF WORDS, ENSURING NO HOMONYM CONFUSION

wordset = set()


for each in sentences1:
    for every in nltk.word_tokenize(each.lower()):   #nltk.word_tokenize(each.lower())
        wordset.add(every)

for each in sentences2:
    for every in nltk.word_tokenize(each.lower()):
        wordset.add(every)



wordlist = list(wordset)
worddict = {}
for ind,each in enumerate(wordlist): 
    worddict[each] = ind    


# ADD SPECIAL TAGS TO WORD DICTIONARY
 
worddict = {k:(v+4) for k, v in worddict.items()}
worddict["<PAD>"] = 0
worddict["<START>"] = 1
worddict["<END>"] = 2
worddict["<UNK>"] = 3
worddict["<DELIM>"] = 4


  
# CREATE LISTS OF NLTK POS TUPLE IDs TO REPRESENT SENTENCES   (TRY NO POS TAG TO REDUCE VOCAB SIZE)

listedsentence1 = []
for each in sentences1:
    temp = [1]
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))   nltk.word_tokenize(each.lower())
        temp.append(worddict[every])
    listedsentence1.append(temp)
        
listedsentence2 = []    
for each in sentences2:
    temp = [1]
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        temp.append(worddict[every])
    listedsentence2.append(temp)


listedsentences1test = []
for each in sentences1test:
    temp = [1]
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        if every not in worddict:
            temp.append(3)                         # 3 is the index of "<UNK>"
        else:
            temp.append(worddict[every])
    listedsentences1test.append(temp)
        
listedsentences2test = []    
for each in sentences2test:
    temp = [1]
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        if every not in worddict:
            temp.append(3)
        else:
            temp.append(worddict[every])
    listedsentences2test.append(temp)
    

listedsentences1dev = []
for each in sentences1dev:
    temp = [1]
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        if every not in worddict:
            temp.append(3)
        else:
            temp.append(worddict[every])
    listedsentences1dev.append(temp)
        
listedsentences2dev = []    
for each in sentences2dev:
    temp = [1]
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        if every not in worddict:
            temp.append(3)
        else:
            temp.append(worddict[every])
    listedsentences2dev.append(temp)
    


# CREATE INVERSE WORD DICTIONARY

reverseworddict = dict([(value,key) for (key, value) in worddict.items()])




# PADDING


# treat maxlens differently to reduce padding noise? THAT DIDN'T HELP MUCH, I HAVE TO ARTIFICIALLY CHOP THEM EACH DOWN TO ~20 I THINK

maxlen1 = 24
maxlen2 = 24

listedsentence1 = pad_sequences(listedsentence1, maxlen=maxlen1, value=worddict["<PAD>"], padding='pre')  # maxlen
listedsentence2 = pad_sequences(listedsentence2, maxlen=maxlen2, value=worddict["<PAD>"], padding='post')   # maxlen


listedsentences1test = pad_sequences(listedsentences1test, maxlen=maxlen1, value=worddict["<PAD>"], padding='pre')  # maxlen
listedsentences2test = pad_sequences(listedsentences2test, maxlen=maxlen2, value=worddict["<PAD>"], padding='post')   # maxlen


listedsentences1dev = pad_sequences(listedsentences1dev, maxlen=maxlen1, value=worddict["<PAD>"], padding='pre')  # maxlen
listedsentences2dev = pad_sequences(listedsentences2dev, maxlen=maxlen2, value=worddict["<PAD>"], padding='post')   # maxlen


#################

# TURN DATA TO NUMPY AND SPLIT TEST/TRAIN


new1 = np.array(listedsentence1)
new2 = np.array(listedsentence2)

new1test = np.array(listedsentences1test)
new2test = np.array(listedsentences2test)

new1dev = np.array(listedsentences1dev)
new2dev = np.array(listedsentences2dev)


train1 = new1
train2 = new2
test1 = new1test
test2 = new2test
dev1 = new1dev
dev2 = new2dev


train1.shape

####################################################

# ROUND DOWN TRAIN AND TEST SETS TO BE DIVISIBLE BY BATCH SIZE (RNNs seem to require exactness, at least in stateful mode)

BATCH_SIZE = 64



def Create_Model(length, vocab_size):
    
    
	# sentence 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 512)(inputs1)
    
    conv1 = Conv1D(filters=256, kernel_size=3, activation='relu')(embedding1)
    drop1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)

    flat1 = Flatten()(pool1)
    
    # sentence 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 512)(inputs2)
    
    conv2 = Conv1D(filters=256, kernel_size=3, activation='relu')(embedding2)
    drop2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)

    flat2 = Flatten()(pool2)
    
    merged = concatenate([flat1, flat2]) # axis = 1

    
    dense = Dense(4096, activation='relu')(merged)
    drop0 = Dropout(0.2)(dense)
    dense0 = Dense(1024, activation='relu')(drop0)
    dropl = Dropout(0.2)(dense0)
    dense1 = Dense(256, activation='relu')(dropl)
    dropm = Dropout(0.2)(dense1)
    dense2 = Dense(64, activation='relu')(dropm)
    dropn = Dropout(0.2)(dense2)
    dense3 = Dense(16,activation='relu')(dropn)
    dropo = Dropout(0.2)(dense3)
    outputs = Dense(3, activation='softmax')(dropo)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(Adam(lr = 0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model


#################################################### 

# TRAIN MODEL

length = len(listedsentence1[0])
vocab_size = len(worddict)

model = Create_Model(length, len(reverseworddict))   #vocab_size

model.fit([train1, train2], labeltrainhot, epochs=6, batch_size=BATCH_SIZE, validation_data=([test1,test2], labeltesthot), callbacks=[tensorboard_callback])


####################################################

# CREATE MODEL PREDICTIONS FOR CONFUSION MATRIX

nnpredictions = model.predict([test1,test2])

testlabellist = list(labeltest)

labeldictreverse = { 0 : 'contradiction', 1 : 'entailment', 2 : 'neutral'}



preds = []
for each in nnpredictions:
    index = np.argmax(each)
    preds.append(labeldictreverse[index])
correct = []    
for each in range(0,len(preds)):
    if preds[each] == testlabellist[each]:
        correct.append(1)
    else:
        correct.append(0)    
accuracy = sum(correct)/len(correct)
print(accuracy)        


len(preds)
cf = pd.crosstab(np.array(preds), np.array(testlabellist))



sb.heatmap(cf, annot = True, cmap = "Blues", fmt='g')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Testing Confusion Matrix')





# Dev CM

nnpredictions = model.predict([dev1, dev2])

testlabellist = list(labeldev)

labeldictreverse = { 0 : 'contradiction', 1 : 'entailment', 2 : 'neutral'}



preds = []
for each in nnpredictions:
    index = np.argmax(each)
    preds.append(labeldictreverse[index])
correct = []    
for each in range(0,len(preds)):
    if preds[each] == testlabellist[each]:
        correct.append(1)
    else:
        correct.append(0)    
accuracy = sum(correct)/len(correct)
print(accuracy)        


len(preds)
cf = pd.crosstab(np.array(preds), np.array(testlabellist))



sb.heatmap(cf, annot = True, cmap = "Blues", fmt='g')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Dev Confusion Matrix')


