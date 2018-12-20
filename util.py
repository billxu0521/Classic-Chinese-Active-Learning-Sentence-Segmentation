# -*- coding: utf8 -*-
import math
#puncts_str = "\t .!?#$%&\'()*+,-/:;<=>@[\]^_`{|}~！？｡。'\"＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏'·".decode('utf8')
puncts = "\t .!?#$%&\'()*+,-/:;<=>@[\]^_`{|}~！？｡。'\"＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏'·"

def file_to_lines(filenames):
    print(filenames)
    for fn in filenames:
        file = open(fn, 'r')
        for line in file:
            #line = line.decode('utf8').replace('\n',"")
            line = line.replace('\n',"")
            if len(line)>0:
                yield line
        file.close()
        
def text_to_lines(texts):

    #line = line.decode('utf8').replace('\n',"")
    texts = texts.replace('\r',"") 
    texts = texts.split( '\n' )        
    #if len(line)>0:
        #yield line
    return texts
        
#增加S,E
def line_toseq(line, charstop):
    if charstop: return char_stop_toseq(line)
    else: return stop_char_toseq(line)

def make_charset(list_of_sentences, includefreq):
    li = [line_toraw(line) for line in list_of_sentences]
    st = "".join(li)
    d = {}
    for u in st:
        #if d.has_key(u):
        if u in d:
            d[u]=d[u]+1
        else:
            d[u]=1
    #for u in d.keys():
    for u in list(d.keys()):
        if d[u]<includefreq:
            del d[u]
    return d.keys()

def line_toraw(line):
    import re
    #return re.sub('[%s]' % re.escape(puncts+"\n".decode('utf8')), '', line)
    return re.sub('[%s]' % re.escape(puncts+"\n"), '', line)

def seq_to_line(x, y, charstop ,s_str ,n_str):
    assert len(x)==len(y)
    s = ""
    score = 0
    for a, b ,c ,d in zip(x, y, s_str,n_str):
        if b =='S': b=','
        else: b=''
        if c >= d : score = c
        else: score = d
        score = (score - 0.5) * 10
        score =  float('%2.2f' % score)
        if charstop: s = s+a+'('+ str(score) +')'+b
        else: s = s+b+a+'('+str(c)+'/'+str(d)+')'
    return s

def decode_totext(rawtext, outputs, charstop):
    assert len(rawtext) == len(outputs)
    text = []
    for i in range(len(rawtext)):
        x = rawtext[i]
        y = outputs[i]
        if charstop: y = y[1:]
        else: y = y[:-1]
        line = ""
        #print x, y, len(x), len(y)
        assert len(x) == len(y)
        for j in range(len(x)):
            if y[j][0]:
                if charstop: line = line + x[j] + "."
                else: line = line + "."+x[j]
            else: line = line + x[j]
        #line = line + x[j]
        text.append(line)
    return text

def seq_to_sparsevec(x,y,charset):
    #input: x, y: [char,...], [label,...]
    from scipy.sparse import csr_matrix

    xseq = []
    for c in x:
        vec = [0] * len(charset)
        if c in charset: vec[charset.index(c)] = 1
        xseq.append(vec)

    yseq = []
    for l in y:
        if l == 'N': vec = [0]
        else: vec = [1]
        yseq.append(vec)

        #print "hi"
    #print xseq.nbytes, yseq.nbytes
    #print csr_matrix(xseq).nbytes, csr_matrix(yseq).nbytes
    return csr_matrix(xseq), csr_matrix(yseq)

def seq_to_densevec(x, y, mydict):
    #xseq = [mydict.get(c, default=mydict["zero"]) for c in x]
    xseq = []
    for c in x:
       #if mydict.has_key(c):
        if c in mydict:
            xseq.append(mydict[c])
        else:
            xseq.append(mydict["zero"])

    yseq = []
    for l in y:
        if l == 'N': vec = [0]
        else: vec = [1]
        yseq.append(vec)

    return xseq, yseq

#在句首加上S 在句尾最後一字定義為S 輸出兩個陣列 一個為標註 一個為拆解後原文
def char_stop_toseq(line):
    # attributes segmentation to the preceding character
    line = "S,"+line
    c = []
    l = []
    for u in line:
        if u in puncts:
            l[-1]="S"
        else:
            c.append(u)
            l.append("N")
    return c,l

def stop_char_toseq(line):
    # attributes segmentation to the following character
    line = line+",E"
    c = []
    l = []
    currentlabel = "N"
    for u in line:
        if u in puncts:
            currentlabel = "S"
        else:
            c.append(u)
            l.append(currentlabel)
            currentlabel = "N"
    assert len(c) == len(l)
    return c,l

def readvec(vecfilename):
    mydict = {}
    file = open(vecfilename, 'r')
    for line in file:
        line = line.split()
        #gram = line[0].decode('utf8')
        gram = line[0]
        params = line[1:]
        params = [float(p) for p in params]
        base = math.sqrt(sum([p*p for p in params]))
        #params = [p/base for p in params]
        
        data  = {}
        for i in range(len(params)):
            data[str(i)] = params[i]/base
        
        mydict[gram] = data
    #mydict[gram] = params
    file.close()
    return mydict

def readvec2(vecfilename):
    mydict = {}
    file = open(vecfilename, 'r')
    for line in file:
        line = line.split()
        #gram = line[0].decode('utf8')
        gram = line[0]
        params = line[1:]
        params = [float(p) for p in params]
        data  = {}
        for i in range(len(params)):
            data[str(i)] = params[i]
        
        mydict[gram] = data
    file.close()
    return mydict

def lstmvec(vecfilename):
    mydict = {}
    file = open(vecfilename, 'r')
    for line in file:
        line = line.split()
        #gram = line[0].decode('utf8')
        gram = line[0]
        params = line[1:]
        params = [float(p) for p in params]
        base = math.sqrt(sum([p*p for p in params]))
        params = [p/base for p in params]
        
        mydict[gram] = params
    zr = [p*0 for p in params]
    mydict["zero"] = zr
    file.close()
    return mydict

def eval(ref, out, signi):
    assert len(ref) == len(out)
    tp = 0.0 #正確斷句 true positives
    fp = 0.0 #錯誤斷句 false positives
    fn = 0.0 #錯誤不斷句 false negatives
    tn = 0.0 #正確不斷句 true negatives
    for i in range(len(ref)):
        if ref[i] == signi: #如果原始資料有S標記
            if out[i] == signi: tp = tp+1 #如果輸出結果跟原始資料一樣 正確的S
            else: fn = fn+1 #原始資料是S 輸出資料沒有S 失敗的N
        elif out[i] == signi:
            fp = fp+1#輸出資料是S 原始資料沒有S  失敗的S
        else:
            tn = tn+1 #都沒事  正確的N
    return tp, fp, fn, tn


