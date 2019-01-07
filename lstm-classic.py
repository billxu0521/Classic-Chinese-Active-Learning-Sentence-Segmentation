#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 01:45:48 2018

@author: billxu
"""
import sys
import glob
import random
import util
import datetime
from urllib.parse import unquote
import numpy
import csv
import math
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,InputLayer,Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

#讀檔
def file_to_lines(filenames):
    file = open(fn, 'r')
    for line in file:
        #line = line.decode('utf8').replace('\n',"")
        line = line.replace('\n',"")
        if len(line)>0:
            yield line
    file.close()
    
#計算F-score
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

#宣告起始資料
material = 'data/shiji-3/*'
dictfile = 'data/vector/sjwsg50.txt'
dense = True# 1 = dense, 0 = one-hot sparse
charstop = True # True means label attributes to previous char
modelname = material.replace('/','').replace('*','')+"sg50"
rowdata = []
filenames = glob.glob(material)

starttime = datetime.datetime.now()
print ("Starting Time:",starttime)

for fn in filenames:
    li = [line for line in file_to_lines(glob.glob(fn))]#已經切成陣列
    rowdata.append(li)

print ("Preparing dictionaries...")
if dense: vdict = util.lstmvec(dictfile)
else: charset = util.make_charset(li,7)

#print(rowdata)
#建立對應的陣列，作為判別是否成為訓練資料 0為不作為訓練資料 1為做訓練資料
traindataidx = numpy.zeros(len(rowdata),int) #陣列長度

#建立LOG
filedatetime = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H%M%S')
f = open(filedatetime + "_log.txt", 'w')

#csv欄位
log_csv_text = [['Round','Block','Presicion','Recall','F1-score']]
for i in range(len(rowdata)):    
    #第i回
    roundtext = i+1
    #訓練模型名稱
    modelname = material.replace('/','').replace('*','')+"_BiLSTM_classic_round_"+str(i)+".m"
    print('Round:',roundtext)
    log_text = "=====Round:" + str(i+1) + "======" + "\n"
    #依序成為訓練資料
    traindataidx[i] = 1
    #整理訓練資料與測試資料
    trainidx = [] #作為訓練資料的索引
    testidx = [] #作為測試資料的索引
    for a in range(len(traindataidx)):
        if traindataidx[a] == 1:
            trainidx.append(a)
        elif traindataidx[a] == 0: #最後一次是空的
            testidx.append(a)
    if (i+1) == len(rowdata):
        print('Last Round')
        break
    print('train:',trainidx)
    print('test:',testidx)
    
    traindataary = []
    testdata = []
    
    for i in trainidx:
        traindataary += rowdata[i]
    testdataary = []
    
    for i in testidx:
        _data = rowdata[i]
        testdata.append(_data)
    
    #資料處理[訓練資料]    
    dataset_train = []
    dataset = []

    while traindataary:
        x, y = util.line_toseq(traindataary.pop(), charstop)
        if dense: dataset.append(util.seq_to_densevec(x, y, vdict))
        else: dataset.append(util.seq_to_sparsevec(x,y,charset))
        if not len(traindataary)%1000: print ("len(dataset_train)", len(traindataary))
    dataset_train = dataset

    #資料集重新轉面
    trainX = numpy.array(dataset_train[0][0])
    input_shape = trainX.shape[1]
    trainX = numpy.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
    print(trainX.shape[0])
    trainY = to_categorical(dataset_train[0][1])
    #trainY = dataset_train[0][1]
    
    #進行建模設定
    model = Sequential()
    model.add(InputLayer(input_shape=(1, 50)))
    model.add(Bidirectional(LSTM(1)))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.compile(loss='binary_crossentropy', #categorical_crossentropy , binary_crossentropy
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])
    model.summary()
    
    #訓練模型檔案
    model.fit(trainX, trainY, validation_data=(trainX, trainY),epochs=20, batch_size=32, verbose=2)
    
    #開始測試
    print (datetime.datetime.now())
    print ("Start closed testing...")
    results = []
    
    f.write(str(log_text))
    dataset_test = []
    for j in range(len(testdata)):
        #print(testdata[j])
        _dataset = []
        
        #while testdata[j]:
        for line in testdata[j]:
            x, y = util.line_toseq(line, charstop)
            if dense: _dataset.append(util.seq_to_densevec(x, y, vdict))
            else: _dataset.append(util.seq_to_sparsevec(x,y,charset))
            if not len(testdata[j])%1000: print ("len(dataset_test)", len(testdata[j]))
        
        dataset_test = dataset
        testX = numpy.array(dataset_test[0][0])
        #testY = numpy.array(dataset_test[0][1])
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        testY = to_categorical(dataset_train[0][1])
        #testY = dataset_train[0][1]

        #第j區塊
        blocktext = j+1
        scores = model.evaluate(testX, testY, verbose=2)  
        print('scores:',scores)
        testPredict = model.predict_classes(testX)
        #print('res:',testPredict)
        trainPro = model.predict_proba(testX)
        #print('pro:',trainPro)
        #predict_classes = testPredict.reshape(-1)
        predict_classes = testPredict
        real_data = dataset_test[0][1]   #原始資料
        f_score = f1_score(real_data, predict_classes)
        r = recall_score(real_data, predict_classes)        
        p = precision_score(real_data, predict_classes) 
        log_text += "Presicion:" + str(p) +'\n'
        log_text += "Recall:" + str(r) +'\n'
        log_text += "F1-Score:" + str(f_score) + '\n'
        log_text += '\n' + "=============" + '\n'
        log_csv_text.append([str(roundtext),str(blocktext),str(p),str(r),str(f_score)])
        print ("Presicion:", p)
        print ("Recall:", r)
        print ("F1-score:", f_score)
        f.write(str(log_text))
        log_text = "----Doc Result:" + str(blocktext) +"-----" + "\n"
        #log_text += "Total tokens in Test Set:" + str(tp+fp+fn+tn) +'\n'
        #log_text += "Total S in REF:" + str(tp+fn) +'\n'
        #log_text += "Total S in OUT:" + str(tp+fp) +'\n'
        #print ("Total tokens in Test Set:", tp+fp+fn+tn)
        #print ("Total S in REF:", tp+fn)
        #print ("Total S in OUT:", tp+fp)
        

#寫入csv
with open(filedatetime + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for list in log_csv_text:
        writer.writerow(list)
f.close()
