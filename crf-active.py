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
from scipy.stats import linregress

#資料處理
def dataary(li,gram,features,vdict):
    data = []
    lineary = ''
    for _line in li:   
        lineary += _line
    
    lineary = lineary.replace("：", "")
    lineary = lineary.replace("、", "")
    lineary = lineary.replace("！", "")
    lineary = lineary.replace(".", "")
    lineary = lineary.replace("？", "")
    lineary = lineary.replace("》", "")
    lineary = lineary.replace("《", "")
    x, y = util.line_toseq(lineary, charstop)
    del y[0]
    y = y + ['N']
    #這邊在做文本做gram
    if features == 1:
        d = crf.x_seq_to_features_discrete(x, charstop,gram), y
    elif features == 2:
        d = crf.x_seq_to_features_vector(x, vdict, charstop), y
    elif features == 3:
        d = crf.x_seq_to_features_both(x, vdict, charstop,gram), y
    #d = crf.x_seq_to_features_discrete(x, charstop,gram), y
    data.append(d)
    '''
    for line in li:
        
        x, y = util.line_toseq(line, charstop)
        
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
    '''
    return data

#讀檔
def file_to_lines(filenames):
    print('open:',filenames)
    file = open(fn, 'r')
    #allline = []
    for line in file:
        #line = line.decode('utf8').replace('\n',"")
        line = line.replace('\n',"")
        #if line != "":
            #allline.append(line)
        if len(line)>0:
            yield line
    #yield allline
    file.close()    
    
#宣告起始資料
dataname = 'ws3'
material = 'data/' + dataname + '/*'
dictfile = dataname + '_word2vec.model.txt'
crfmethod = "lbfgs"  # {‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}
charstop = True # True means label attributes to previous char
rowdata = []
features = 1 #資料清洗模式
gram = 2 #特徵樣板
filenames = glob.glob(material)
filenames = sorted(filenames)
part_log = filenames #紀錄檔名
u_score_log = [] #紀錄各區塊分數
starttime = datetime.datetime.now()
print ("Starting Time:",starttime)

for fn in filenames:
    li = [line for line in file_to_lines(glob.glob(fn))]#已經切成陣列
    rowdata.append(li)

#建立對應的陣列，作為判別是否成為訓練資料 0為不作為訓練資料 1為做訓練資料
traindataidx = numpy.zeros(len(rowdata),int) #陣列長度
vdict = []

#讀取字典
if features > 1:
    vdict = util.readvec(dictfile)#先處理文本
    print ("Dict:", dictfile)

#建立LOG
filedatetime = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H%M%S')
f = open(filedatetime + "_" + str(dataname) + "_CRF_active_round_log.txt", 'w')

#這邊處理CSV需要的資訊
all_pre = numpy.array([])
all_recall = numpy.array([])
all_speci = numpy.array([])
all_fscore = numpy.array([])
all_textcount = numpy.array([])
all_segcount = numpy.array([])
all_text_score = []#紀錄每個區塊的不確定
log_csv_text = [['Type','Round','Test Part','Presicion','Recall','F1-score','U-score']]
data_csv_text = [['','Presicion','Recall','Specificity','F1-score']] #分析表標題
log_text = ''

#資料處理
alldata = []
test = []
all_text_count = 0
#整理文本區塊的資訊
text_obj = {}
for a in range(len(rowdata)):
    count = 0
    segcount = 0
    _data = []
    for x in rowdata[a]:
        for i in x:
            if i in util.puncts:
                segcount += 1
                continue
            else:
                count += 1
    roundtext = a #序號 從0開始
    _d = dataary(rowdata[a],gram,features,vdict)
    print('text_count:',count)
    all_textcount = numpy.append(all_textcount,count)
    all_segcount = numpy.append(all_segcount,segcount)
    all_text_count += count
    _data.extend(_d)    
    print('data_seq:',len(_data))
    text_obj[roundtext]=([count,0])
    alldata.append(_data) 
    log_text += 'Part:' + str(a) + '/count:' + str(count) +'\n' 
    data_csv_text[0].append(part_log[a])
    all_text_score.append([None])
    
print('alldata_seq:',len(alldata))
print('alltext_count:',all_text_count)
log_text += 'All_text_count:' + str(all_text_count)  +'\n'
f.write(str(log_text))

text_score = [] #紀錄每個區塊的不確定
for i in range(len(rowdata)):
    #第i回
    roundtext = i+1
    text_score.sort(key=lambda x:x[1],reverse=True)
    if text_score != []:
        print(text_score)
    #訓練模型名稱
    modelname = material.replace('/','').replace('*','')+"_CRF_active_round_"+str(i)+".m"
    print('Round:',roundtext)
    log_text = 'Starting Time:' + str(datetime.datetime.now()) + '\n'
    log_text += "=====Round:" + str(i+1) + "======" + "\n"
    
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
    log_text += "use trindata:" +str(trainidx) + "\n"
    log_text += "u_score" +str(text_score) + "\n"
    print('train:',trainidx)
    print('test:',testidx)
    text_score = [] #重置
    traindataary = []
    testdata = []
    traindata = []
    testtextidx = [] #每段文本長度
    count = 0 #計算文本長度
    testdataary = []
    countary = []
    train_x = []
    train_y = []
    for i in trainidx:
        for a in alldata[i]:
            
            train_x.extend(a[0])
            train_y.extend(a[1])
            #_d = a[0],a[1]
            #traindata.append(_d)
    _d = train_x,train_y
    traindata = _d
    print('traindata_seq:',len(traindata))           
    #ft = open(dataname + str(roundtext) + 'test_log.txt', 'w')
    #ft.write(str(traindata))
    #ft.close()
    
    for i in testidx:
        testdata_x = []
        testdata_y = []
        for a in alldata[i]:
            #testdata_x.extend(a[0])
            #testdata_y.extend(a[1])
            #_d = a[0],a[1]
            #testdata.append(_d)
            test_data = a[0],a[1]
            testdata.append(test_data)
    
    print('testdata_seq:',len(testdata))
    
    #進行建模
    trainer = pycrfsuite.Trainer()
    #for t in traindata:
        #x, y = t
        #trainer.append(x, y)
    trainer.append(traindata[0], traindata[1])
    trainer.select(crfmethod)  #做訓練
    trainer.set('max_iterations',30) #測試迴圈
    #trainer.set('delta',0)
    #print ("!!!!before train", datetime.datetime.now())
    trainer.train(modelname)
    #print ("!!!!after train", datetime.datetime.now())
    
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
    all_len= 0
    f.write(str(log_text))
    log_text = ''
    while testdata:
        xseq, yref = testdata.pop()
        #print(xseq)
        yout = tagger.tag(xseq)
        all_len += len(yout)
        #print(len(xseq),len(yout),all_len)
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
        lines.append(util.seq_to_line([x['gs0'] for x in xseq],yout,charstop,Spp,Npp))
        #print(util.seq_to_line([x['gs0'] for x in xseq], (str(sp) +'/'+ str(np)),charstop))
        
        score_array = []
        All_u_score = 0
        for i in range(len(Spp)):
            _s = 0
            if Spp[i] > Npp[i]: _s = Spp[i]
            else :_s = Npp[i]
            #_s = (_s - 0.5) * 10
            _s = (1 - _s)
            #U_score = U_score + _s
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
        all_text_score[testidx[i]].append(U_score)
    #All_u_score = (U_score / len(Spp)) #區塊不確定性   
    tp, fp, fn, tn = zip(*results)
    tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)
    print(tp, fp, fn, tn)

    if tp <= 0 or fp <= 0 :
        p = 0
        r = 0
        s = 0
        f_score = 0
    else :
        p, r, s= tp/(tp+fp), tp/(tp+fn) ,tn/(fp+tn)
        f_score = 2*p*r/(p+r)
    
    log_text += "----Doc Result-----" + "\n"
    log_text += "Total tokens in Test Set:" + str(tp+fp+fn+tn) +'\n'
    log_text += "Total S in REF:" + str(tp+fn) +'\n'
    log_text += "Total S in OUT:" + str(tp+fp) +'\n'
    log_text += "Presicion:" + str(p) +'\n'
    log_text += "Recall:" + str(r) +'\n'
    log_text += "F1-Score:" + str(f_score) + '\n'
    log_text += "character count:" + str(len(Spp)) + '\n'
    #log_text += "Uncertain-Score:" + str((U_score / len(Spp))) + '\n'
    log_text += '\n' + "=============" + '\n'
    log_text += 'End Time:' + str(datetime.datetime.now()) + '\n'
    log_text += '\n'
    
    #建立分析表資料
    all_pre = numpy.append(all_pre,p)
    all_recall = numpy.append(all_recall,r)
    all_speci = numpy.append(all_speci,s)
    all_fscore = numpy.append(all_fscore,f_score)
    _data_log = []
    _data_log.append(str(p)) 
    _data_log.append(str(r))
    _data_log.append(str(s)) 
    _data_log.append(str(f_score))
    _data_log.extend([None] * len(rowdata))
    
    #紀錄每個區塊的不確定值
    log_csv_text.append(['test-score',str(roundtext),'',str(p),str(r),str(f_score),''])
    for a in  range(len(text_score)):
        log_csv_text.append(['Un-score',str(roundtext),str(text_score[a][0]),'','','',str(text_score[a][1])])
        _data_log[int(text_score[a][0])+4] = str(text_score[a][1])
    
    _data_log.insert(0,str(roundtext))
    data_csv_text.append(_data_log)
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

#整理CSV需要的資料
allround = (numpy.arange(len(rowdata) - 1)) #計算斜率用
avr_pre = numpy.mean(all_pre)
avr_recall = numpy.mean(all_recall)
avr_speci = numpy.mean(all_speci)
avr_fscore = numpy.mean(all_fscore)
max_pre = numpy.max(all_pre)
max_recall = numpy.max(all_recall)
max_speci = numpy.max(all_speci)
max_fscore = numpy.max(all_fscore)
slope_pre = linregress(allround, all_pre.tolist())   
slope_recall = linregress(allround, all_recall.tolist())   
slope_speci = linregress(allround, all_speci.tolist())   
slope_fscore = linregress(allround, all_fscore.tolist())   
avr_data_log = ['Avr',avr_pre,avr_recall,avr_speci,avr_fscore]
max_data_log = ['Max',max_pre,max_recall,max_speci,max_fscore]
slope_data_log = ['Slope',slope_pre.slope,slope_recall.slope,slope_speci,slope_fscore.slope]
all_count_log = ['AllTextCount','','',''] 
all_segcount_log = ['AllSegCount','','',''] 
all_textseg_log = ['','','','']  #斷句率
avr_data = []
max_data = []
slope_data = []
for a in range(len(all_text_score)):
    _all_u_score = 0
    _count = 0
    _slope_ary = []
    for i in all_text_score[a]:
        if i == None:continue
        _count += 1
        _all_u_score += i
        _slope_ary.append(i)       
    if _count <= 0:
        avr_data_log.append('')
        slope_data.append('')
        max_data.append('')
    else:
        _avr = _all_u_score/_count
        avr_data.append(_avr)
        _x = numpy.arange(len(_slope_ary)) #計算斜率用
        slope_res = linregress(_x, _slope_ary)   
        slope_data.append(slope_res.slope)
        _uscore_ary = numpy.array(_slope_ary)
        _max = numpy.max(_uscore_ary)
        max_data.append(_max) #個回合不確定最大值
all_textseg = []  #斷句率
for x in range(len(all_textcount)):
    _a = all_textcount[x]/all_segcount[x]
    all_textseg.append(_a)
max_data_log.extend(max_data)
slope_data_log.extend(slope_data)
avr_data_log.extend(avr_data)
all_count_log.extend(all_textcount)
all_segcount_log.extend(all_segcount)
all_textseg_log.extend(all_textseg)
data_csv_text.append(avr_data_log)
data_csv_text.append(max_data_log)
data_csv_text.append(slope_data_log)
data_csv_text.append(all_count_log)
data_csv_text.append(all_segcount_log)
data_csv_text.append(all_textseg_log)

#寫入csv
with open(filedatetime + '_active.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for list in log_csv_text:
        writer.writerow(list)
with open(filedatetime + 'data_active.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for list in data_csv_text:
        writer.writerow(list)
f.close()
