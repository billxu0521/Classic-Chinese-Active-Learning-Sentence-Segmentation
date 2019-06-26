#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 01:45:48 2018

@author: billxu
"""
import sys
import glob
import util
import datetime
from urllib.parse import unquote
import numpy
import csv
from keras.preprocessing import sequence  
from keras.preprocessing.text import Tokenizer  
import util
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.embeddings import Embedding  
from keras.layers import Input,Activation,Embedding,LSTM,TimeDistributed,Dense,Bidirectional,InputLayer,Dropout,Flatten
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from scipy.stats import linregress


#資料處理
def dataary(rowdata):
    data_x = []
    data_y = []
    data = []
    while rowdata:
        x, y = util.line_toseq(rowdata.pop(), charstop)
        data_x.append(x)
        data_y.append(y)

    data.append([data_x,data_y])
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
dataname = 'ws2'
material = 'data/' + dataname + '/*'
rowdata = []
#features = 1 #資料清洗模式
#gram = 2 #特徵樣板
charstop = True # True means label attributes to previous char
filenames = glob.glob(material)
filenames = sorted(filenames)
part_log = filenames #紀錄檔名
u_score_log = [] #紀錄各區塊分數
starttime = datetime.datetime.now()
MAX_LEN_OF_TOKEN = 64  

print ("Starting Time:",starttime)

for fn in filenames:
    li = [line for line in file_to_lines(glob.glob(fn))]#已經切成陣列
    rowdata.append(li)

#random.shuffle(rowdata)#做亂數取樣
#rowdata = ['1111','2222','33333']
#建立對應的陣列，作為判別是否成為訓練資料 0為不作為訓練資料 1為做訓練資料
traindataidx = numpy.zeros(len(rowdata),int) #陣列長度
print(traindataidx)
vdict = []

#建立LOG
filedatetime = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H%M%S')
f = open(filedatetime + "_" + str(dataname) + "_LSTM_classic_round_log.txt", 'w')

#這邊處理CSV需要的資訊
all_pre = numpy.array([])
all_recall = numpy.array([])
all_fscore = numpy.array([])
all_textcount = numpy.array([])
all_segcount = numpy.array([])
all_text_score = []#紀錄每個區塊的不確定
log_csv_text = [['Type','Round','Test Part','Presicion','Recall','F1-score','U-score']]
data_csv_text = [['','Presicion','Recall','F1-score']] #分析表標題
log_text = ''

#資料處理
alldata = []
test = []
all_text_count = 0
#整理文本區塊的資訊
text_obj = {}
for a in range(len(rowdata)):
    count = 0
    _data = []
    segcount = 0
    for x in rowdata[a]:
        for i in x:
            if i in util.puncts:
                segcount += 1
                continue
            else:
                count += 1
    roundtext = a #序號 從0開始
    _d = dataary(rowdata[a])
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
for i in range(len(alldata)):
    #第i回
    roundtext = i+1
    text_score.sort(key=lambda x:x[1],reverse=True)
    if text_score != []:
        print(text_score)
    #訓練模型名稱
    modelname = material.replace('/','').replace('*','')+"_BiLSTM_active_round_"+str(i)+".h5"
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
   
    print('train:',trainidx)
    print('test:',testidx)
    text_score = [] #重置
    testdata = []
    traindata = []
    c = 0
    for i in trainidx:
        for a in alldata[i]:
            _d = a[0],a[1]
            traindata.append(_d)
            
    for i in testidx:
        for a in alldata[i]:
            _d = a[0],a[1]
            testdata.append(_d)
     
    print('traindata_seq:',len(traindata))   
         
    trainrow_x = []
    trainrow_y = []
 
    for t in traindata:
        x, y = t
        trainrow_x.extend(x)
        trainrow_y.extend(y)
    
    train_x = []
    train_y = []
    
    for i in trainrow_x:
        for a in i:
            train_x.append(a)
    for i in trainrow_y:
        for a in i:
            #train_label.append(lambda a: a != 'S')
            if a == 'S':
                a = 1
            else:
                a = 0
            train_y.append(a)
            
    print('trainX:',len(train_x))
    print('trainY:',len(train_y))

    token = Tokenizer(num_words = 100)  
    token.fit_on_texts(train_x)  
    c = 0  
    for t,i in token.word_index.items():  
        #print("\t'{}'\t{}".format(t, i))  
        c += 1  
        if c == 10:  
            break  
        
    x_train_seq = token.texts_to_sequences(train_x)  
    x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN_OF_TOKEN)  
    y_train = np.array(train_y)
    y_train = to_categorical(y_train)   
    
    model = Sequential()
    model.add(Embedding(20000, 128, input_length=MAX_LEN_OF_TOKEN))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    #進行建模
    train_history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
    
    #建立訓練模型檔案
    model.save(modelname)
    #model.load_weights('my_model_weights.h5')

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
    f.write(str(log_text))
    log_text = ''
    
    while testdata:
        x = []
        yref = []
        _x, _yref = testdata.pop()
        for i in _x:
            for a in i:
                x.append(a)
        for i in _yref:
            for a in i:
                #train_label.append(lambda a: a != 'S')
                if a == 'S':
                    a = 1
                else:
                    a = 0
                yref.append(a)
        #LSTM需要轉換
        x_test_seq = token.texts_to_sequences(x)  
        x_test = sequence.pad_sequences(x_test_seq, maxlen=MAX_LEN_OF_TOKEN)
        
        yout = model.predict_classes(x_test)
        #pr = tagger.probability(yref)
        p_1 = 0
        p_2 = 0
        prob = model.predict_proba(x_test)
        for i in range(len(yout)):
            p_1 = prob[i, 0]
            Spp.append(p_1) #標記的機率
            np_2p = prob[i, 1]
            Npp.append(p_2)#標記的機率
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
            #_s = (_s - 0.5) * 10
            _s = (1 - _s)
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
        all_text_score[testidx[i]].append(U_score)
        
   
    f_score = f1_score(yref, yout)
    r = recall_score(yref, yout)        
    p = precision_score(yref, yout) 
    
    log_text += "----Test Result----\n"
    
    #log_text += "Pr:" + str(pr) +'\n'
    log_text += "Presicion:" + str(p) +'\n'
    log_text += "Recall:" + str(r) +'\n'
    log_text += "F1-Score:" + str(f_score) + '\n'
    log_text += '\n' + "=============" + '\n'
    log_text += 'End Time:' + str(datetime.datetime.now()) + '\n'
    log_text += '\n'
    
    #建立分析表資料
    all_pre = numpy.append(all_pre,p)
    all_recall = numpy.append(all_recall,r)
    all_fscore = numpy.append(all_fscore,f_score)
    _data_log = []
    _data_log.append(str(p)) 
    _data_log.append(str(r)) 
    _data_log.append(str(f_score))
    _data_log.extend([None] * len(rowdata))
    
    #紀錄每個區塊的不確定值
    log_csv_text.append(['test-score',str(roundtext),'',str(p),str(r),str(f_score),''])
    for a in  range(len(text_score)):
        log_csv_text.append(['Un-score',str(roundtext),str(text_score[a][0]),'','','',str(text_score[a][1])])
        _data_log[int(text_score[a][0])+3] = str(text_score[a][1])
    
    _data_log.insert(0,str(roundtext))
    data_csv_text.append(_data_log)
    print ("Presicion:", p)
    print ("Recall:", r)
    print ("F1-score:", f_score)
    f.write(str(log_text))
    #重置
    if roundtext == len(rowdata):
        print('Last Round')
        #break
    log_text = ''
    #traindataidx[(roundtext - 1)] = 1
    
#整理CSV需要的資料
#allround = (numpy.arange(len(rowdata) )) #跑不同模型計算斜率用
allround = (numpy.arange(len(rowdata) - 1)) #計算斜率用

avr_pre = numpy.mean(all_pre)
avr_recall = numpy.mean(all_recall)
avr_fscore = numpy.mean(all_fscore)
max_pre = numpy.max(all_pre)
max_recall = numpy.max(all_recall)
max_fscore = numpy.max(all_fscore)
print(allround)
print(all_pre)

slope_pre = linregress(allround, all_pre.tolist())  
 
slope_recall = linregress(allround, all_recall.tolist())   
slope_fscore = linregress(allround, all_fscore.tolist())   

avr_data_log = ['Avr',avr_pre,avr_recall,avr_fscore]
max_data_log = ['Max',max_pre,max_recall,max_fscore]
slope_data_log = ['Slope',slope_pre.slope,slope_recall.slope,slope_fscore.slope]
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
with open(filedatetime + '_classic.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for list in log_csv_text:
        writer.writerow(list)
with open(filedatetime + 'data_classic.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for list in data_csv_text:
        writer.writerow(list)
f.close()