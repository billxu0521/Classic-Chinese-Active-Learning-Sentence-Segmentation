#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 01:45:48 2018

@author: billxu
"""

import sys
import glob
import random
import pycrfsuite
import crf
#import runcrf
import util
import datetime
import numpy



def dataary(li):
    data = []
    for line in li:
        x, y = util.line_toseq(line, charstop)
        #print(x)
        #print(y[:5])
    
        #這邊在做文本做gram
        d = crf.x_seq_to_features_discrete(x, charstop,1), y
        data.append(d)
    return data

#讀檔
def file_to_lines(filenames):
    file = open(fn, 'r')
    for line in file:
        #line = line.decode('utf8').replace('\n',"")
        line = line.replace('\n',"")
        if len(line)>0:
            yield line
    file.close()

#宣告起始資料
material = 'data/24s/*'
size = 8
trainportion = 0.9
dictfile = 'data/vector/24scbow300.txt'
crfmethod = "l2sgd"  # {‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}
charstop = True # True means label attributes to previous char
features = 1 # 1=discrete; 2=vectors; 3=both
random.seed(101)
#訓練模型名稱
modelname = material.replace('/','').replace('*','')+str(size)+str(charstop)+".m"
rowdata = []
filenames = glob.glob(material)
 
for fn in filenames:
    li = [line for line in file_to_lines(glob.glob(fn))]#已經切成陣列
    rowdata.append(li)

print(rowdata)
#rowdata = ['1111','2222','33333']
#建立對應的陣列，作為判別是否成為訓練資料 0為不作為訓練資料 1為做訓練資料
traindataidx = numpy.zeros(len(rowdata),int) #陣列長度
print(traindataidx)

for i in range(len(rowdata)):
    print('Round:',i+1)
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
        print('Last')
        break
    print('train:',trainidx)
    print('test:',testidx)
    
    traindataary = []
    for i in trainidx:
        traindataary = numpy.hstack((traindataary,rowdata[i]))
    testdataary = []
    for i in trainidx:
        testdataary = numpy.hstack((testdataary,rowdata[i]))
    
    #資料處理
    traindata = dataary(traindataary)
    testdata = dataary(testdataary)
    #進行建模
    trainer = pycrfsuite.Trainer()
    for t in traindata:
        x, y = t
        trainer.append(x, y)
    trainer.select(crfmethod)#做訓練
    trainer.set('max_iterations',10) #測試迴圈
    #trainer.set('delta',0)
    #print ("!!!!before train", datetime.datetime.now())
    trainer.train(modelname)
    #print ("!!!!after train", datetime.datetime.now())
    
    tagger = pycrfsuite.Tagger()
    #建立訓練模型檔案
    tagger.open(modelname)
    tagger.dump(modelname+".txt")
    
    #測試(輸出資料)
    
    #準備第二次


