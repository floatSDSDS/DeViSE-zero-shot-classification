
# Analysis main

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#%%
class aLog(object):
    def __init__(self, log):
        self.timestamp = log[0]['experiment_time']
        self.model = log[0]['model_name']
        self.dataset = log[0]['dataset']
        self.testmode = log[0]['test_mode']
        self.bestEpoch = log[0]['best_epoch']
        self.config = log[0]
        self.seen_class = log[1]
        self.unseen_class = log[2]
        self.logSepClass = log[3]
        
    def sepMeasure(self,mode='precision'):
        defOrder=dict(precision=0,recall=1,fmeasure=2,support=3)
        seplogFinal=self.logSepClass[self.bestEpoch]
        M=self.unseen_class.copy()
        for c in self.unseen_class.keys():
            M[c]=seplogFinal[defOrder[mode]][self.unseen_class[c]]
        return M
    
    def summaryMeasure(self,mode='precision',weighted=True,report='overall'):
        
        rst=0
        M=self.sepMeasure(mode)
        weight=self.sepMeasure('support')
        no_seenclass=len(self.seen_class)
        if self.testmode==1 and report !='overall':
            if report == 'seen':  
                Cset={k:v for k,v in self.unseen_class.items() if v<no_seenclass}
            elif report =='unseen':
                Cset={k:v for k,v in self.unseen_class.items() if v>=no_seenclass}
            mykeys=list(Cset.keys())
            M={k:M[k] for k in mykeys}
            weight={k:weight[k] for k in mykeys}
        
        no_class=len(M)
        no_sample=sum(list(weight.values()))
        
        if weighted:
            for c in M.keys():
                rst+=M[c]*weight[c]
            rst=rst/no_sample
        else:
            rst=sum(list(M.values()))/no_class
            
        return rst

#%%

def loadLogs (path):
    files= os.listdir(path) 
    Logs = []
    for file in files:
        if not os.path.isdir(file): 
            with open(path+"/"+file, 'rb') as f:
                Logs.append(aLog(pickle.load(f)))
    return Logs

def LD2L (lst,key):
    rst=[]
    for l in lst:
        rst.append(l[key])
    return rst

def avg_ACC(LogSet):
    accs=[]
    for log in LogSet:
        accs.append(log.bestACC())
    return sum(accs)/len(accs)

#%%
path=["C:/Users/floatsd/Documents/git/DeviSE/testDeVise/SMP1"]
Stat=[]
ClassSet=[]
weighted=True
for p in path:
    Logs=loadLogs(p)
    for i in range(len(Logs)):
        unseenC=set(Logs[i].unseen_class.keys())
        if unseenC not in ClassSet:
            ClassSet.append(unseenC)
        stat=dict(indx=ClassSet.index(unseenC),classSet='_'.join(unseenC),\
                  model=Logs[i].model,\
                  rec_o=Logs[i].summaryMeasure('recall',weighted,'overall'),\
                  rec_s=Logs[i].summaryMeasure('recall',weighted,'seen'),\
                  rec_u=Logs[i].summaryMeasure('recall',weighted,'unseen'),\
                  prc_o=Logs[i].summaryMeasure('precision',weighted,'overall'),\
                  prc_s=Logs[i].summaryMeasure('precision',weighted,'seen'),\
                  prc_u=Logs[i].summaryMeasure('precision',weighted,'unseen'),\
                  fm_o=Logs[i].summaryMeasure('fmeasure',weighted,'overall'),\
                  fm_s=Logs[i].summaryMeasure('fmeasure',weighted,'seen'),\
                  fm_u=Logs[i].summaryMeasure('fmeasure',weighted,'unseen'))
        Stat.append(stat)
col_order=['indx','model','classSet','rec_o','rec_s','rec_u',\
           'prc_o','prc_s','prc_u','fm_o','fm_s','fm_u']
df=pd.DataFrame(Stat,columns=col_order)

#%%
print('dataset:',Logs[0].dataset)
print('testmode:',Logs[0].testmode)

if Logs[0].testmode==1:
    for col in col_order[3:12]:
        print(df.groupby('model')[col].mean())
else: 
    for col in [col_order[3],col_order[6],col_order[9]]:
        print(df.groupby('model')[col].mean())
