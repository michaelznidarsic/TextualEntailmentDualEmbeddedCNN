# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:39:33 2020

@author: mznid
"""

import pandas as pd
import numpy as np
import os
import random

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import keras   
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, AveragePooling2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding, LSTM,Bidirectional,TimeDistributed,AveragePooling1D,GlobalMaxPooling1D,Reshape,Input,Concatenate,concatenate
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sb

import nltk


# UNLIST GPU OPTION FOR TENSORFLOW 2.1. GPU RUNS OUT OF MEMORY WITH THIS NN MODEL AND EVEN RUNS SLOWER THAN CPU 

tf.config.set_visible_devices([], 'GPU')



# LOAD AND CLEAN

raw = pd.read_csv("D:\\stanford-natural-language-inference-corpus\\snli_1.0_train.csv")

#for ind,each in enumerate(sentences2):
#    if type(each) == float:
#        print(ind)
nullset = {91479,91480,91481,311124,311125,311126}        
nonnullset = set(range(0,len(raw.iloc[:,0]))) - nullset
raw = raw.iloc[sorted(list(nonnullset)),:]


#########################################################################

# DOWNSAMPLE

#sampleindex = random.sample(range(0,len(raw.iloc[:,0])), round(0.1 * len(raw.iloc[:,0]))) # 0.05
#raw = raw.iloc[sampleindex,:]
#raw.index = range(0,len(raw.iloc[:,0]))


#########################################################################


sentences1 = list(raw.iloc[:,5])
sentences2 = list(raw.iloc[:,6])
labellist = list(raw.iloc[:,9])
labelset = set(labellist)




# TEST/TRAIN SAMPLING

index = range(0,len(labellist))
trainindex = random.sample(index, round(0.7 * len(labellist)))
testindex = set(index) - set(trainindex)
testindex = list(testindex)
random.shuffle(testindex)



# HANDLE LABELS

labeltrain = np.array(labellist)[trainindex]
labeltest = np.array(labellist)[testindex]

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




# PARSE SENTENCES WITH NLTK pos_tag TO CREATE SET OF WORDS, ENSURING NO HOMONYM CONFUSION

wordset = set()

for each in sentences1:
    for every in nltk.pos_tag(nltk.word_tokenize(each.lower())):
        x,y = every
        wordset.add(every)

for each in sentences2:
    for every in nltk.pos_tag(nltk.word_tokenize(each.lower())):
        x,y = every
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
worddict["<UNUSED>"] = 4


  
# CREATE LISTS OF NLTK POS TUPLE IDs TO REPRESENT SENTENCES

listedsentence1 = []
for each in sentences1:
    temp = [1]
    for every in nltk.pos_tag(nltk.word_tokenize(each.lower())):
        x,y = every
        temp.append(worddict[every])
    listedsentence1.append(temp)
        
listedsentence2 = []    
for each in sentences2:
    temp = [1]
    for every in nltk.pos_tag(nltk.word_tokenize(each.lower())):
        x,y = every
        temp.append(worddict[every])
    listedsentence2.append(temp)
    


# CREATE INVERSE WORD DICTIONARY

reverseworddict = dict([(value,key) for (key, value) in worddict.items()])


# FIND MAX LENGTH

maxlen = 0
for each in listedsentence1:
    if len(each) > maxlen:
        maxlen = len(each)
for each in listedsentence2:
    if len(each) > maxlen:
        maxlen = len(each)
print(maxlen)



# PADDING

for ind,each in enumerate(listedsentence1):
    if len(each) < maxlen:
        listedsentence1[ind] += [0] * (maxlen - len(each))
    listedsentence1[ind] += [2]
    
for ind,each in enumerate(listedsentence2):
    if len(each) < maxlen:
        listedsentence2[ind] += [0] * (maxlen - len(each))
    listedsentence2[ind] += [2]




# TURN DATA TO NUMPY AND SPLIT TEST/TRAIN

new1 = np.array(listedsentence1)
new2 = np.array(listedsentence2)


train1 = new1[trainindex,]
train2 = new2[trainindex,]
test1 = new1[testindex,]
test2 = new2[testindex,]




####################################################

# CREATE MODEL

def Create_Model(length, vocab_size):
    
	# sentence 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 512)(inputs1)
    
    conv1 = Conv1D(filters=256, kernel_size=16, activation='relu')(embedding1)
    drop1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)

    conv10 = Conv1D(filters=256, kernel_size=4, activation='relu')(pool1)
    drop10 = Dropout(0.1)(conv10)
    pool10 = MaxPooling1D(pool_size=2)(drop10)

    flat1 = Flatten()(pool10)
    
    # sentence 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 512)(inputs2)
    
    conv2 = Conv1D(filters=256, kernel_size=16, activation='relu')(embedding2)
    drop2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)

    conv20 = Conv1D(filters=256, kernel_size=4, activation='relu')(pool2)
    drop20 = Dropout(0.1)(conv20)
    pool20 = MaxPooling1D(pool_size=2)(drop20)

    flat2 = Flatten()(pool20)
    
    merged = concatenate([flat1, flat2]) # axis = 1

    
    dense = Dense(4096, activation='relu')(merged)
    drop0 = Dropout(0.25)(dense)
    dense0 = Dense(1024, activation='relu')(drop0)
    dropl = Dropout(0.25)(dense0)
    dense1 = Dense(256, activation='relu')(dropl)
    dropm = Dropout(0.25)(dense1)
    dense2 = Dense(64, activation='relu')(dropm)
    dropn = Dropout(0.25)(dense2)
    dense3 = Dense(16,activation='relu')(dropn)
    dropo = Dropout(0.25)(dense3)
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

model = Create_Model(length, vocab_size)

model.fit([train1,train2], labeltrainhot, epochs=6, batch_size=64, validation_data=([test1,test2], labeltesthot))


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



cf = pd.crosstab(np.array(preds), np.array(testlabellist))



sb.heatmap(cf, annot = True, cmap = "Blues", fmt='g')
plt.xlabel('Actual')
plt.ylabel('Prediction')




