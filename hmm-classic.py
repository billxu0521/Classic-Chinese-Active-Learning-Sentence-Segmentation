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
import util
import datetime
from urllib.parse import unquote
import numpy
import csv
import nltk
from nltk.tag import hmm

#資料處理

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
material = 'data/24s-1/*'
size = 8
trainportion = 0.9
dictfile = 'data/vector/24scbow300.txt'
crfmethod = "l2sgd"  # {‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}
charstop = True # True means label attributes to previous char
features = 1 # 1=discrete; 2=vectors; 3=both
random.seed(101)

rowdata = []
filenames = glob.glob(material)
 
for fn in filenames:
    li = [line for line in file_to_lines(glob.glob(fn))]#已經切成陣列
    rowdata.append(li)

#print(rowdata)
#rowdata = ['1111','2222','33333']
#建立對應的陣列，作為判別是否成為訓練資料 0為不作為訓練資料 1為做訓練資料
traindataidx = numpy.zeros(len(rowdata),int) #陣列長度
print(traindataidx)

#建立LOG
filedatetime = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H%M%S')
f = open(filedatetime + "_log.txt", 'w')
#csv欄位
log_csv_text = [['Round','Block','Presicion','Recall','F1-score']]
for i in range(len(rowdata)):
    #第i回
    roundtext = i+1
    #訓練模型名稱
    modelname = material.replace('/','').replace('*','')+str(size)+str(charstop)+"_round_"+str(i)+".m"
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
        traindataary = numpy.hstack((traindataary,rowdata[i]))
    testdataary = []
    for i in testidx:
        testdataary = rowdata[i]
        testdata.append(testdataary)
    #print(testdata)
    traindata = []
    #資料處理    
    for line in traindataary:
        x, y = util.line_toseq(line, charstop)
        traindata.append(zip(x,y)) 
    
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(traindata)
    #建立訓練模型檔案
    
    
    #開始測試
    print (datetime.datetime.now())
    print ("Start closed testing...")
    results = []
    f.write(str(log_text))
    
    for j in range(len(testdata)):
        #第j區塊
        blocktext = j+1
        results = []
        for line in testdata[j]:
            x, yref = util.line_toseq(line, charstop)
            out = tagger.tag(x)
            all_x = []
            print(tagger.probability(out))
            _, yout = zip(*out)
            results.append(util.eval(yref, yout, "S"))
        
       
        tp, fp, fn, tn = zip(*results)
        tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)
        
        p, r = tp/(tp+fp), tp/(tp+fn)
        f_score = 2*p*r/(p+r)
        log_text = "----Doc Result:" + str(blocktext) +"-----" + "\n"
        log_text += "Total tokens in Test Set:" + str(tp+fp+fn+tn) +'\n'
        log_text += "Total S in REF:" + str(tp+fn) +'\n'
        log_text += "Total S in OUT:" + str(tp+fp) +'\n'
        log_text += "Presicion:" + str(p) +'\n'
        log_text += "Recall:" + str(r) +'\n'
        log_text += "F1-Score:" + str(f_score) + '\n'
        log_text += '\n' + "=============" + '\n'
        log_csv_text.append([str(roundtext),str(blocktext),str(p),str(r),str(f_score)])
        print ("Total tokens in Test Set:", tp+fp+fn+tn)
        print ("Total S in REF:", tp+fn)
        print ("Total S in OUT:", tp+fp)
        print ("Presicion:", p)
        print ("Recall:", r)
        print ("F1-score:", f_score)
        f.write(str(log_text))
      
#寫入csv
with open(filedatetime + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for list in log_csv_text:
        writer.writerow(list)
f.close()




