# -*- coding: UTF8 -*-
import numpy
import math
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
import datetime
import util
import glob
import random
import sys
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(self.validation_data[0]))).round()##.model
        val_targ = self.validation_data[1]###.model
        _val_f1 = f1_score(val_targ, val_predict,average='micro')
        _val_recall = recall_score(val_targ, val_predict,average=None)###
        _val_precision = precision_score(val_targ, val_predict,average=None)###
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        #print("— val_f1: %f — val_precision: %f — val_recall: %f" %(_val_f1, _val_precision, _val_recall))
        print("— val_f1: %f "%_val_f1)
        return

metrics = Metrics()

material = 'data/24s-1/*'
size = 10
trainportion = 0.9
validateportion = 0.05
cut1 = int(size*trainportion)
cut2 = int(size*(trainportion+validateportion))
dictfile = 'data/vector/sjwsg50.txt'
dense = True# 1 = dense, 0 = one-hot sparse
charstop = True # True means label attributes to previous char
modelname = material.replace('/','').replace('*','')+str(size)+"sg50"
validate_interval = 10000
hidden_size = 50
learning_rate = 0.001
random.seed(101)

print ("Material:", material)
print ("Size:", size, "entries,", trainportion, "as training", validateportion, "as validation")
print ("Dense:", dense)
print ("charstop:", charstop)

starttime = datetime.datetime.now()
print ("Starting Time:",starttime)

print ("Preparing text...")
li = [line for line in util.file_to_lines(glob.glob(material))]
random.shuffle(li)
li = li[:size]

print ("Preparing dictionaries...")
if dense: vdict = util.lstmvec(dictfile)
else: charset = util.make_charset(li,7)

print ("Preparing datasets...")
dataset_train = li[:cut1]
dataset_validate = li[cut1:cut2]
dataset_test = li[cut2:]

dataset = []
print(type(dataset_train))

while dataset_train:
    x, y = util.line_toseq(dataset_train.pop(), charstop)
    if dense: dataset.append(util.seq_to_densevec(x, y, vdict))
    else: dataset.append(util.seq_to_sparsevec(x,y,charset))
    if not len(dataset_train)%1000: print ("len(dataset_train)", len(dataset_train))
dataset_train = dataset

dataset = []
while dataset_validate:
    x, y = util.line_toseq(dataset_validate.pop(), charstop)
    if dense: dataset.append(util.seq_to_densevec(x, y, vdict))
    else: dataset.append(util.seq_to_sparsevec(x,y,charset))
    if not len(dataset_validate)%1000: print ("len(dataset_validate)", len(dataset_validate))
dataset_validate = dataset

dataset = []
while dataset_test:
    x, y = util.line_toseq(dataset_test.pop(0), charstop)
    if dense: dataset.append(util.seq_to_densevec(x, y, vdict))
    else: dataset.append(util.seq_to_sparsevec(x,y,charset))
    if not len(dataset_test)%1000: print ("len(dataset_test)", len(dataset_test))
dataset_test = dataset
look_back = 1

trainX = numpy.array(dataset_train[0][0])
trainY = numpy.array(dataset_train[0][1])
testX = numpy.array(dataset_test[0][0])
testY = numpy.array(dataset_test[0][1])
'''
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
'''
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

max_features = 600
maxlen = 50


# create and fit the LSTM network
model = Sequential()
#model.add(LSTM(4,input_shape=(1,50)))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
model.add(Bidirectional(LSTM(units=20),input_shape=(1,50)))
#model.add(Dropout(0.5))
#model.add(Dense(1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


model.fit(trainX, trainY, validation_data=(trainX, trainY),epochs=30, batch_size=64, verbose=2,callbacks=[metrics])

print("Start evaluation...")  
scores = model.evaluate(trainX, trainY, verbose=1)  
print("Score=",str(scores))  

# make predictions
trainPredict = model.predict_proba(trainX)
testPredict = model.predict_classes(testX)
print('res:',testPredict)
trainPro = model.predict_proba(testX)
print('pro:',trainPro)
predict_classes = testPredict
#predict_classes = testPredict.reshape(-1)
real_data = dataset_test[0][1] 
f_score = f1_score(real_data, predict_classes)
r = recall_score(real_data, predict_classes)        
p = precision_score(real_data, predict_classes) 
print ("Presicion:", p)
print ("Recall:", r)
print ("F1-score:", f_score)

