# -*- coding: utf8 -*-
# crf.py #
# Convert sequences to features. #

# x是陣列 charstop是要不要用前置斷句
def x_seq_to_features_discrete(x, charstop,gram):
    
    if gram == 2:
        # features are for 2 chars before and after Segmentation
        # 1-gram and 2-gram
        if charstop: findex = [-1,0,1,2]
        else: findex = [-2,-1,0,1]
        xf = []
        ran = range(len(x)) #陣列長度
        for i in ran: 
            mydict = {}
            # 1-gram
            for j in findex: # -1 0 1 2
                if i+j in ran: 
                    mydict["gs"+str(j)]=x[i+j]  
                    
            # 2-gram
            for j in findex[:3]:
                if i+j in ran and i+j+1 in ran:
                    mydict["gd"+str(j)]=x[i+j]+x[i+j+1]
                    
            xf.append(mydict)
        return xf
    elif gram == 2:
        if charstop: findex = [-1,0,1,2]
        else: findex = [-2,-1,0,1]
        xf = []
        ran = range(len(x)) #陣列長度
        for i in ran: 
            mydict = {}
            # 1-gram
            for j in findex: # -1 0 1 2
                if i+j in ran: 
                    mydict["gs"+str(j)]=x[i+j]  
                    
            # 2-gram
            for j in findex[:3]:
                if i+j in ran and i+j+1 in ran:
                    mydict["gd"+str(j)]=x[i+j]+x[i+j+1]
                    
            xf.append(mydict)
        return xf
        
        
        return xf

def x_seq_to_features_vector(x, dict, charstop):
    if charstop: findex = [-1,0,1,2]
    else: findex = [-2,-1,0,1]
    
    xf = []
    for i in range(len(x)):
        mydict = {}
        for j in findex:
            if i+j>(-1) and i+j<len(x):
                try:
                    mydict["gv"+str(j)]=dict[x[i+j]]
                except KeyError:
                    pass
        xf.append(mydict)
    return xf

def x_seq_to_features_both(x, dict, charstop):
    f = x_seq_to_features_discrete(x, charstop)
    fv = x_seq_to_features_vector(x,dict, charstop)
    for i in range(len(f)):
        f[i].update(fv[i])
    return f
