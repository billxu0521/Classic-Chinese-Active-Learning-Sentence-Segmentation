# -*- coding: utf8 -*-
import sys
import glob
import random
import pycrfsuite
import crf
import util
import datetime

material = 'data/sumen/*'
#material = "data/sjw/A05*"
size = 1300
trainportion = 0.9
dictfile = 'sumen_word2vec.model.txt'
crfmethod = "l2sgd"  # {‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}
charstop = True # True means label attributes to previous char
features = 1 # 1=discrete; 2=vectors; 3=both
random.seed(101)

#宣告指令式
"python runcrf.py 'data/sjw/*' 80 data/vector/vectors300.txt 1 1"
args = sys.argv
if len(args)>1:
    material = args[1]
    size = int(args[2])
    dictfile = args[3]
    features = int(args[4])
    charstop = int(args[5])
cut = int(size*trainportion)

#訓練模型名稱
modelname = material.replace('/','').replace('*','')+str(size)+str(charstop)+".m"

print ("Material:", material)
print ("Size:", size, "entries,", trainportion, "as training")

print (datetime.datetime.now())

# Prepare li: list of random lines
if features > 1:
    vdict = util.readvec(dictfile)#先處理文本
    print ("Dict:", dictfile)
li = [line for line in util.file_to_lines(glob.glob(material))]#已經切成陣列
random.shuffle(li)#做亂數取樣
print(len(li))
#li = li[:size]

# Prepare data: list of x(char), y(label) sequences
data = []

for line in li:
    x, y = util.line_toseq(line, charstop)
    #print(x)
    #print(y[:5])

    #這邊在做文本做gram
    if features == 1:
        d = crf.x_seq_to_features_discrete(x, charstop,1), y
    elif features == 2:
        d = crf.x_seq_to_features_vector(x, vdict, charstop), y
    elif features == 3:
        d = crf.x_seq_to_features_both(x, vdict, charstop,1), y
    data.append(d)

#date = [(['劉','敬','者','齊','人','也','漢','五','年'], ['S', 'S', 'N','S', 'N', 'N','N', 'S', 'N'])]
traindata = data[:cut]
#traindata = date
testdata = data[cut:]

trainer = pycrfsuite.Trainer()
#print trainer.params()
#print(traindata[0])
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

modelname = 'datasumen_CRF_classic_round_9.m'
#建立訓練模型檔案
tagger.open(modelname)
tagger.dump(modelname+".txt")

#print(tagger.marginal('S',1))

print (datetime.datetime.now())
print ("Start testing...")
results = []
while testdata:
    x, yref = testdata.pop()
    
    yout = tagger.tag(x)
    pr = tagger.probability(yref)
    results.append(util.eval(yref, yout, "S"))
tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print ("Total tokens in Test Set:", tp+fp+fn+tn)
print ("Total S in REF:", tp+fn)
print ("Total S in OUT:", tp+fp)
print ("Presicion:", p)
print ("Recall:", r)
print ("*******************F1-score:", 2*p*r/(p+r))


print (datetime.datetime.now())
print ("Start closed testing...")
results = []
#test_y = ['S', 'N', 'N','S', 'S', 'N','S', 'S', 'S']
while traindata:
    x, yref = traindata.pop()
    yout = tagger.tag(x)
    pr = tagger.marginal('S',0)
    #pp = tagger.probability(test_y)
    results.append(util.eval(yref, yout, "S"))

tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print ("Total tokens in Train Set:", tp+fp+fn+tn)
print ("Total S in REF:", tp+fn)
print ("Total S in OUT:", tp+fp)
print ("Presicion:", p)
print ("Recall:", r)
print ("*******************F1-score:", 2*p*r/(p+r))
print ("*******************:", pr)
#print ("*******************:", pp)
print ("*******************:", yout)
print (datetime.datetime.now())
