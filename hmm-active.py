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
def dataary(li,gram):
    data = []
    for line in li:
        x, y = util.line_toseq(line, charstop)
        #print(x)
        #print(y[:5])
    
        #這邊在做文本做gram
        d = crf.x_seq_to_features_discrete(x, charstop,gram), y
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
log_csv_text = [['Round','Block','Presicion','Recall','F1-score','Block-uncertain']]

text_score = [] #紀錄每個區塊的不確定
for i in range(len(rowdata)):
    #第i回
    roundtext = i+1
    text_score.sort(key=lambda x:x[1])
    #訓練模型名稱
    modelname = material.replace('/','').replace('*','')+str(size)+str(charstop)+"_round_"+str(i)+".m"
    print('Round:',roundtext)
    log_text = "=====Round:" + str(i+1) + "======" + "\n"
    
    #依序成為訓練資料
    if text_score != []:
        traindataidx[int(text_score[0][0])] = 1
    elif text_score == []:
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
    log_text += "use trindata:" +str(trainidx) + "\n"
    log_text += "u_score" +str(text_score) + "\n"
    print('train:',trainidx)
    print('test:',testidx)
    text_score = [] #重置
    traindataary = []
    testdata = []
    for i in trainidx:
        traindataary = numpy.hstack((traindataary,rowdata[i]))
    testdataary = []
    for i in testidx:
        testdataary = rowdata[i]
        testdata.append(testdataary)
    traindata = []
    #資料處理    
    for line in traindataary:
        x, y = util.line_toseq(line, charstop)
        traindata.append(zip(x,y)) 
    
    #進行建模
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(traindata)

    #建立訓練模型檔案
    
    #開始測試
    print (datetime.datetime.now())
    print ("Start closed testing...")
    results = []
    lines = []
    Spp = []
    Npp = []
    f.write(str(log_text))
    for j in range(len(testdata)):
        #第j區塊
        blocktext = j+1
        for line in testdata[j]:
            x, yref = util.line_toseq(line, charstop)
            out = tagger.tag(x)
            _, yout = zip(*out)
            results.append(util.eval(yref, yout, "S"))
            sp = 0
            np = 0
            for i in range(len(yout)):
                sp = tagger.probability([x[i],'S'])
                Spp.append(sp) #S標記的機率
                #print(sp)
                np = tagger.probability([x[i],'N']) 
                Npp.append(np)#N標記的機率
                #print(np)
            results.append(util.eval(yref, yout, "S"))
            lines.append(util.seq_to_line([x['gs0'] for x in x],yout,charstop,Spp,Npp))
            #print(util.seq_to_line([x['gs0'] for x in xseq], (str(sp) +'/'+ str(np)),charstop))
        
        U_score = 0
        p_Scount = 0
        p_Ncount = 0
        for i in range(len(Spp)):
            _s = 0
            if Spp[i] > Npp[i]:
                _s = Spp[i]
            else :_s = Npp[i]
            _s = (_s - 0.5) * 10
            U_score = U_score + _s
            p_Scount = p_Scount + Spp[i]
            p_Ncount = p_Ncount + Npp[i]
           
        All_u_score = (U_score / len(Spp)) #區塊不確定性
        text_score.append([str(testidx[j]),All_u_score])
        
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
        log_text += "character count:" + str(len(Spp)) + '\n'
        log_text += "Uncertain-Score:" + str((U_score / len(Spp))) + '\n'
        log_text += '\n' + "=============" + '\n'
        log_csv_text.append([str(roundtext),str(blocktext),str(p),str(r),str(f_score),str(All_u_score)])
        print ("Total tokens in Test Set:", tp+fp+fn+tn)
        print ("Total S in REF:", tp+fn)
        print ("Total S in OUT:", tp+fp)
        print ("Presicion:", p)
        print ("Recall:", r)
        print ("F1-score:", f_score)
        print ("character count:" + str(len(Spp)))
        print ("block uncertain rate:" + str((U_score / len(Spp)))) 
        f.write(str(log_text))
      
#寫入csv
with open(filedatetime + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for list in log_csv_text:
        writer.writerow(list)
f.close()




