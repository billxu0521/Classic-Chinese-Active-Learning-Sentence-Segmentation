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

#資料處理
def dataary(li,gram,features,vdict):
    data = []
    for line in li:
        x, y = util.line_toseq(line, charstop)
        #print(x)
        #print(y[:5])
        #這邊在做文本做gram
        if features == 1:
            d = crf.x_seq_to_features_discrete(x, charstop,gram), y
        elif features == 2:
            d = crf.x_seq_to_features_vector(x, vdict, charstop), y
        elif features == 3:
            d = crf.x_seq_to_features_both(x, vdict, charstop,gram), y
        #d = crf.x_seq_to_features_discrete(x, charstop,gram), y
        data.append(d)
    return data

#讀檔
def file_to_lines(filenames):
    file = open(fn, 'r')
    allline = ""
    for line in file:
        #line = line.decode('utf8').replace('\n',"")
        line = line.replace('\n',"")
        if line != "":
            allline += line
        #if len(line)>0:
        #    yield line
    yield allline
    file.close()    

#宣告起始資料
dataname = 'sumen'
material = 'data/' + dataname + '/*'
dictfile = dataname + '_word2vec.model.txt'
crfmethod = "l2sgd"  # {‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}
charstop = True # True means label attributes to previous char
rowdata = []
features = 3 #資料清洗模式
gram = 1 #特徵樣板
filenames = glob.glob(material)
ft = open(str(dataname) + "_text.txt", 'w')

starttime = datetime.datetime.now()
print ("Starting Time:",starttime)

for fn in filenames:
    li = [line for line in file_to_lines(glob.glob(fn))]#已經切成陣列
    rowdata.append(li)
random.shuffle(rowdata)#做亂數取樣

#rowdata = ['1111','2222','33333']
#建立對應的陣列，作為判別是否成為訓練資料 0為不作為訓練資料 1為做訓練資料
traindataidx = numpy.zeros(len(rowdata),int) #陣列長度
print(traindataidx)

#讀取字典
if features > 1:
    vdict = util.readvec(dictfile)#先處理文本
    print ("Dict:", dictfile)

#建立LOG
filedatetime = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H%M%S')
f = open(filedatetime + "_CRF_classic_round_log.txt", 'w')
#csv欄位
log_csv_text = [['Type','Round','Test Part','Presicion','Recall','F1-score','U-score']]
log_text = ''
#資料處理
alldata = []
for i in rowdata:
    alldata.append(dataary(i,gram,features,vdict)) 
ft.write(str(alldata))
ft.close()
#整理文本區塊的資訊
text_obj = {}
for i in range(len(alldata)): #字數
    count = 0
    roundtext = i #序號 從0開始
    rowdatarya = dataary(rowdata[i],1,features,vdict) #整理內文
    count += len(rowdatarya[0][0])
    text_obj[roundtext]=([count,0])
    log_text += 'Part:' + str(i) + '<' + str(count) + '>' +'\n' 

text_score = [] #紀錄每個區塊的不確定
for i in range(len(alldata)):
    #第i回
    roundtext = i+1
    #訓練模型名稱
    modelname = material.replace('/','').replace('*','')+"_CRF_classic_round_"+str(i)+".m"
    print('Round:',roundtext)
    log_text = 'Starting Time:' + str(datetime.datetime.now()) + '\n'
    log_text += "=====Round:" + str(i+1) + "======" + "\n"
    
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
    
    print('train:',trainidx)
    print('test:',testidx)
    text_score = [] #重置
    testdata = []
    traindata = []
    for i in trainidx:
        _data = alldata[i][0][0],alldata[i][0][1]
        traindata.append(_data)

    for i in testidx:
        _data = alldata[i][0][0],alldata[i][0][1]
        testdata.append(_data)
    
    
    #進行建模
    trainer = pycrfsuite.Trainer()
    for t in traindata:
        x, y = t
        trainer.append(x, y)
    
    trainer.select(crfmethod)#做訓練
    trainer.set('max_iterations',10) #測試迴圈

    trainer.train(modelname)
    tagger = pycrfsuite.Tagger()
    
    #建立訓練模型檔案
    tagger.open(modelname)
    tagger.dump(modelname+".txt")
    if roundtext == len(rowdata):
        print('Last Round')
        break
    #開始測試
    print (datetime.datetime.now())
    print ("Start closed testing...")
    results = []
    lines = []
    Spp = []
    Npp = []
    all_len = 0 
    f.write(str(log_text))
    log_text = ''

    while testdata:
        x, yref = testdata.pop(0)
        yout = tagger.tag(x)
        all_len += len(yout)
        pr = tagger.probability(yref)
        sp = 0
        np = 0
        for i in range(len(yout)):
            sp = tagger.marginal('S',i)
            Spp.append(sp) #S標記的機率
            #print(sp)
            np = tagger.marginal('N',i) 
            Npp.append(np)#N標記的機率
            #print(np)
        results.append(util.eval(yref, yout, "S"))
        
        score_array = []
        All_u_score = 0
        p_Scount = 0
        p_Ncount = 0
        for i in range(len(Spp)):
            _s = 0
            if Spp[i] > Npp[i]:
                _s = Spp[i]
            else :_s = Npp[i]
            _s = (_s - 0.5) * 10
            #U_score = U_score + _s
            p_Scount = p_Scount + Spp[i]
            p_Ncount = p_Ncount + Npp[i]
            score_array.append(_s)
    for i in range(len(testidx)):
        U_score = 0 #文本區塊的不確定值
        text_count = 0 #字數
        end = 0
        if i == 0:
            start = 0
        else:
            start = end
        end = text_obj[testidx[i]][0]
        #print(text_obj[testidx[i]])
        #print(len(score_array),end)
        for a in range(start,end):
            text_count = text_obj[testidx[i]][0]
            U_score += score_array[a]
        U_score = U_score / text_count
        text_obj[testidx[i]][1] = U_score
        All_u_score += U_score
        text_score.append([str(testidx[i]),U_score])
        
    tp, fp, fn, tn = zip(*results)
    tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)
    #print(tp, fp, fn, tn)
    if tp <= 0 or fp <= 0 :
        p = 0
        r = 0
        f_score = 0
    else :
        p, r = tp/(tp+fp), tp/(tp+fn)
        f_score = 2*p*r/(p+r)
    
    log_text += "----Test Result----\n"
    log_text += "Total tokens in Test Set:" + str(tp+fp+fn+tn) +'\n'
    log_text += "Total S in REF:" + str(tp+fn) +'\n'
    log_text += "Total S in OUT:" + str(tp+fp) +'\n'
    log_text += "Pr:" + str(pr) +'\n'
    log_text += "Presicion:" + str(p) +'\n'
    log_text += "Recall:" + str(r) +'\n'
    log_text += "F1-Score:" + str(f_score) + '\n'
    log_text += '\n' + "=============" + '\n'
    log_text += 'End Time:' + str(datetime.datetime.now()) + '\n'
    log_text += '\n'
    log_csv_text.append(['test-score',str(roundtext),'',str(p),str(r),str(f_score),''])
    for a in  range(len(text_score)):
        log_csv_text.append(['Un-score',str(roundtext),str(a),'','','',str(text_score[a])])
    print ("Total tokens in Test Set:", tp+fp+fn+tn)
    print ("Total S in REF:", tp+fn)
    print ("Total S in OUT:", tp+fp)
    print ("Presicion:", p)
    print ("Recall:", r)
    print ("F1-score:", f_score)
    f.write(str(log_text))
    #重置
    log_text = ''
    trainer.clear() 
        
#寫入csv
with open(filedatetime + '_classic.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for list in log_csv_text:
        writer.writerow(list)
f.close()