import os
import time
import pickle
from random import *
import scipy.io as sio
import input_data
import model_torch as model
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import math

import torch.nn as nn

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

a = Random()
a.seed(1)

#%% setting
test_description='testDeVise'
rep_num = 1
id_split=range(0,10)
# SNIP, SMP18
choose_dataset="SNIP"
choose_model=['DeViSE']
# without seen: 0, with seen: 1, fixed with some classes: -1
dataSetting={}
dataSetting['test_mode']=1
retrainLSTM=False
######
dataSetting['random_class']=False
dataSetting['training_prob']=0.8
dataSetting['test_intrain_prob']=0.3

#%% config input path

dataSetting['data_prefix']='data/SNIP/'
dataSetting['dataset_name']='dataSNIP.txt'
dataSetting['wordvec_name']='wiki.en.vec'
dataSetting['sim_name_withS']='SNIP_similarity_M_zscore.mat'
dataSetting['sim_name_withOS']='SNIP10seen.mat'
if choose_dataset=="SMP18":
    dataSetting['data_prefix']='data/SMP18/'
    dataSetting['dataset_name']='dataSMP18.txt'
    dataSetting['wordvec_name']='sgns_merge_subsetSMP.txt'
    dataSetting['sim_name_withS']='SMP_similarity_M_zscore.mat'
    dataSetting['sim_name_withOS']='SMP44_wSeen_R1.mat'

if choose_dataset =='SNIP':
    dataSetting['unseen_class'] = [['playlist'], ['book']]
elif choose_dataset =='SMP18':
    dataSetting['unseen_class'] = [['天气'],['公交'],['app'],['飞机'],['电影'],['音乐']]
    
#dataSetting['unseen_class'] = []
#%% config setting
    
def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_time = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    s_cnum = np.unique(data['y_tr']).shape[0]
    u_cnum = np.unique(data['y_te']).shape[0]
    config = {}
    config['model_name'] = choose_model[0]
    config['dataset']=choose_dataset
    config['test_mode']=dataSetting['test_mode']
    config['training_prob']=dataSetting['training_prob']
    config['test_intrain_prob']=dataSetting['test_intrain_prob']
    config['wordvec']=dataSetting['wordvec_name']
    config['sim_name_withS']=dataSetting['sim_name_withS']
    config['sim_name_withOS']=dataSetting['sim_name_withOS']
    config['keep_prob'] = 0.5 # embedding dropout keep rate
    config['hidden_size'] = 32 # embedding vector size
    config['batch_size'] = 50 # vocab size of word vectors
    config['vocab_size'] = vocab_size # vocab size of word vectors (10,895)
    config['num_epochs'] = 10 # number of epochs
    config['num_epochs2']= 20
    config['max_time'] = max_time
    config['sample_num'] = sample_num #sample number of training data
    config['test_num'] = test_num #number of test data
    config['s_cnum'] = s_cnum # seen class num
    config['u_cnum'] = u_cnum #unseen class num
    config['word_emb_size'] = word_emb_size # embedding size of word vectors (300)
    config['margin'] = 1.0 # ranking loss margin
    config['d_a']=10
    config['r'] = 3 #self-attention weight hops
    config['learning_rate'] = 0.1
    config['lr_step_size']=10
    config['lr_gamma']=0.1
    config['sim_scale'] = 4 #sim scale
    config['nlayers'] = 2 # default for bilstm
    config['seen_class']=data['seen_class']
    config['unseen_class']=data['unseen_class']
    config['data_prefix']=dataSetting['data_prefix']
    config['ckpt_dir'] = './'+test_description+'/' #check point dir
    config['experiment_time']= time.strftime('%y%m%d%I%M%S')
    config['best_epoch']=0
    config['best_acc']=0
    config['report']=True
    config['cuda_id']=0
    config['untrain_classlen']=data['untrain_classlen']#XIAOTONG
    return config

#%% some functions
    
def generate_batch(n, batch_size):
    batch_index = a.sample(range(n), batch_size)
    return batch_index

def sort_batch1(batch_x, batch_y, batch_len, batch_ind,batch_index):
    batch_len_new = batch_len
    batch_len_new, perm_idx = batch_len_new.sort(0, descending=True)
    batch_x_new = batch_x[perm_idx]
    batch_y_new = batch_y[perm_idx]
    batch_ind_new = batch_ind[perm_idx]
    batch_index_new=torch.tensor(batch_index)[perm_idx]
    
    return batch_x_new, batch_y_new, batch_len_new, batch_ind_new,batch_index_new

def sort_batch2(batch_x, batch_y, batch_len, batch_ind,batch_label,batch_emb):
    batch_len_new = batch_len
    batch_len_new, perm_idx = batch_len_new.sort(0, descending=True)
    batch_x_new = batch_x[perm_idx]
    batch_y_new = batch_y[perm_idx]
    batch_ind_new = batch_ind[perm_idx]
    batch_label_new = batch_label[perm_idx.tolist()]
    batch_emb_new = batch_emb[perm_idx.tolist()]
    
    return batch_x_new, batch_y_new, batch_len_new, batch_ind_new, batch_label_new,batch_emb_new

def cos_loss(input, target):
    return 1 - F.cosine_similarity(input, target).mean()

def loss_devise(pred_emb,y_id,labels_emb):
    
    sc_no=labels_emb.shape[0] # 24
    pred_emb_=torch.unsqueeze(pred_emb.float(),1).repeat(1,sc_no,1) # [50, 24, 300]
    bsz=pred_emb_.shape[0] # 50
    labels_emb_=torch.from_numpy(labels_emb).cuda() # [24, 300]
    labels_emb_=torch.unsqueeze(labels_emb_,0).repeat(bsz,1,1) # [50, 24, 300]
    
    othersim=F.cosine_similarity(pred_emb_,labels_emb_,dim=2) # [50, 24]
    
    val=[]
    for i in range(bsz):
        idx=y_id[i]
        truesim=othersim[i,idx]
        othersim[i]=1-truesim+othersim[i]
        othersim[i,idx]=0
        val.append(sum([max(0,i) for i in othersim[i]]))
    return sum(val)/(len(val))
#%% zsl_evaluate

def zsl_evaluate(data,config,mymodel,newnet):
    
    newnet.eval()    
    mymodel.eval()
    
    x_te = torch.from_numpy(data['x_te']) # [len_te, TT]
    y_te_id = data['y_te'] # [len_te, 1]
    u_len = torch.from_numpy(data['u_len']) # [len_te]
    
    if torch.cuda.is_available():
        x_te = x_te.cuda(cuda_id)
        u_len = u_len.cuda(cuda_id)
    
    total_unseen_pred = np.array([], dtype=np.int64)
    total_y_test = np.array([], dtype=np.int64)
    
    with torch.no_grad():
        batch_te_original = x_te
        batch_len = u_len
        batch_test = y_te_id
        
        # sort by descending order for pack_padded_sequence
        batch_len, perm_idx = batch_len.sort(0, descending=True)
        batch_te = batch_te_original[perm_idx]
        
        if torch.cuda.is_available():
            perm_idx=perm_idx.cpu()
        
        batch_test = batch_test[perm_idx]
        batch_test = np.ravel(batch_test)
        
        
        mymodel(batch_te, batch_len, embedding)
        outp=mymodel.hh
        outp=outp*100
        outp=outp.double()
        
        outp2=newnet(outp)
        
        sc_no=data['uc_vec'].shape[0]
        outp2_=torch.unsqueeze(outp2.float(),1).repeat(1,sc_no,1) 
        bsz=outp2_.shape[0] 
        labels_emb_=torch.from_numpy(data['uc_vec']).cuda() 
        labels_emb_=torch.unsqueeze(labels_emb_,0).repeat(bsz,1,1) 
        sim=F.cosine_similarity(outp2_,labels_emb_,dim=2) 
        unseen_pred=torch.argmax(sim,dim=1)
        
#        for activation in outp2:
#            predtemp=activation.cpu().numpy()
#            predtemp=np.reshape(predtemp,(1,predtemp.shape[0]))
#            sim=cosine_similarity(predtemp,data['uc_vec'])
#            print(sim)
#            prd=np.argmax(sim,1)
#            total_unseen_pred = np.concatenate((total_unseen_pred, prd))
            
        total_y_test = np.concatenate((total_y_test, batch_test))
        acc=accuracy_score(total_y_test,unseen_pred)
        print('                     '+ config['dataset']+" ZStest Perfomance")
        print (classification_report(total_y_test, unseen_pred, digits=4))
        logclasses=precision_recall_fscore_support(total_y_test, unseen_pred)
    return acc,logclasses
#%% load data
    
data = input_data.read_datasets(dataSetting)

# load settings
config = setting(data)
cuda_id=config['cuda_id']

x_tr = torch.from_numpy(data['x_tr'])
y_tr = torch.from_numpy(data['y_tr'])
y_tr_id = torch.from_numpy(data['y_tr'])
y_te_id = torch.from_numpy(data['y_te'])
y_ind = torch.from_numpy(data['s_label'])
s_len = torch.from_numpy(data['s_len'])
embedding = torch.from_numpy(data['embedding'])
x_te = torch.from_numpy(data['x_te'])
u_len = torch.from_numpy(data['u_len'])
w2v=data['w2v']

if torch.cuda.is_available():
    x_tr =x_tr.cuda(cuda_id)
    y_tr =y_tr.cuda(cuda_id)
    y_tr_id = y_tr_id.cuda(cuda_id)
    y_te_id =y_te_id.cuda(cuda_id)
    y_ind =y_ind.cuda(cuda_id)
    s_len = s_len.cuda(cuda_id)
    embedding=embedding.cuda(cuda_id)
    x_te = x_te.cuda(cuda_id)
    u_len = u_len.cuda(cuda_id)
    print('------------------use gpu------------------')

#%%  
batch_num = int(config['sample_num'] / config['batch_size']+1)

# load model
mymodel=model.myLSTM(config,embedding).cuda()
loss_fn = F.cross_entropy
optimizer = optim.Adam(mymodel.parameters(), lr=config['learning_rate'])

if not os.path.exists(config['ckpt_dir']):
    os.mkdir(config['ckpt_dir'])

loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)
hh=np.zeros((config['sample_num'],config['r']*config['hidden_size']*2))
final=np.zeros((config['sample_num'],config['s_cnum']))
H=[None]*config['sample_num']

print('------------------training begin---------------------')
print('sample_number=',config['sample_num']," ,batch_size=",config['batch_size'],'batch_num=',batch_num)

#%% prepare data
wordvec_tr=[]
for w in data['label_tr']:
    wordvec_tr.append(w2v[w])
    
wordvec_tr=np.stack(wordvec_tr)
wordvec_tr=wordvec_tr.reshape((-1,config['word_emb_size']))

#data['uc_vec']=normalize(data['uc_vec'],norm='l2',axis=1)
#data['sc_vec']=normalize(data['sc_vec'],norm='l2',axis=1)
#with open('SNIP_emb.pkl','rb') as f:
#    utterance_tr=pickle.load(f)

#%% LSTM 1
#import torch._utils
#try:
#    torch._utils._rebuild_tensor_v2
#except AttributeError:
#    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#        tensor.requires_grad = requires_grad
#        tensor._backward_hooks = backward_hooks
#        return tensor
#    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
#    
#mymodel=model.myLSTM(config,embedding).cuda()
#mymodel.load_state_dict(torch.load('SNIP_model.pth'))
#
#for param in mymodel.parameters():
#    param.requires_grad = False
# Training cycle

nn.init.xavier_uniform_(mymodel.predict.weight)
for epoch in range(config['num_epochs']):
    avg_acc=0
    epoch_time = time.time()
    mymodel.train()
    for batch in range(batch_num):
        
        batch_index = generate_batch(config['sample_num'], config['batch_size'])
        batch_x = x_tr[batch_index]
        batch_y_id = y_tr_id[batch_index]
        batch_len = s_len[batch_index]
        batch_ind = y_ind[batch_index]

        # sort by descending order for pack_padded_sequence
        batch_x, batch_y_id, batch_len, batch_ind,batch_index_new = sort_batch1(batch_x, batch_y_id,\
                                                                                batch_len,batch_ind,batch_index)

        optimizer.zero_grad()
        mymodel.forward(batch_x, batch_len,embedding)
        outp=mymodel.final_output
        
        loss_val=loss_fn(outp,batch_y_id.long()).sum()
        loss_val.backward()
        optimizer.step()
        
        hh[batch_index_new,:]=mymodel.hh.cpu().detach().numpy()    
        final[batch_index_new,:]=mymodel.final_output.cpu().detach().numpy()    
        
# =============================================================================
#         i=0
#         for idx in batch_index:
# # =============================================================================
# #             H[idx]=mymodel.H.cpu().detach().numpy()[i]
# # =============================================================================
#             i+=1
# =============================================================================
            
        tr_batch_pred = np.argmax(outp.cpu().detach().numpy(), 1)
        acc = accuracy_score(batch_y_id.cpu(), tr_batch_pred)
        avg_acc+=acc
    avg_acc/=batch_num
    print('---epoch:',epoch,' ---acc:',avg_acc)
    

    
for param in mymodel.parameters():
    param.requires_grad = False
#utterance_tr=hh
utterance_tr=torch.from_numpy(hh).double()
#
#sio.savemat('embNew/'+config['dataset']+time.strftime('%y%m%d%I%M%S')+'.mat',
#            dict(data_emb=hh,label_ind=data['y_tr'],\
#            label_emb=data['sc_vec'],label_dict=data['sc_dict']))
#sio.savemat('embNew/'+'5d'+config['dataset']+time.strftime('%y%m%d%I%M%S')+'.mat',
#            dict(data_emb=final,label_ind=data['y_tr'],\
#            label_emb=data['sc_vec'],label_dict=data['sc_dict']))
#%
#%% phase 2
avg_acc=0
for rep_no in range(rep_num):
    logForClasses=[]
    log=[]
    
    config['experiment_time']= time.strftime('%y%m%d%I%M%S')
    filename=config['ckpt_dir']+'mode'+str(config['test_mode'])+'_'+\
    config['dataset']+'_'+config['model_name']+'_'+config['experiment_time']+'.pkl'
    
    p=0.2
    newnet = nn.Sequential(
                           nn.BatchNorm1d(config['r']*config['hidden_size']*2),
                           nn.Dropout(p),
                           nn.Linear(config['r']*config['hidden_size']*2,\
                                     out_features=config['r']*config['hidden_size']*2, bias=True),
                           nn.LeakyReLU(0.1),                          
                           nn.BatchNorm1d(config['r']*config['hidden_size']*2),
                           nn.Dropout(p),
                           nn.Linear(config['r']*config['hidden_size']*2,\
                                     out_features=config['r']*config['hidden_size']*2, bias=True),
                           nn.LeakyReLU(0.1),
                           nn.BatchNorm1d(config['r']*config['hidden_size']*2),
                           nn.Dropout(p),
                           nn.Linear(in_features=config['r']*config['hidden_size']*2, \
                                     out_features=300, bias=True))
    print(newnet)
    newnet=newnet.cuda().double()
    
    nn.init.xavier_uniform_(newnet[2].weight)
    
    print('second phase training begin')
    optimizer2 = optim.Adam(newnet.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer2, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
    best_acc=0
    for epoch in range(config['num_epochs2']):
        
        epoch_time = time.time()
        overall_train_time = 0.0
        scheduler.step()
        avg_loss=0
        avg_acc_tr=0
        newnet.train()
        mymodel.eval()
        for batch in range(batch_num):
            batch_index = generate_batch(config['sample_num'], config['batch_size'])
            batch_x = x_tr[batch_index]
            batch_y_id = y_tr_id[batch_index]
            batch_len = s_len[batch_index]
            batch_ind = y_ind[batch_index].cuda()
            batch_label=torch.from_numpy(wordvec_tr[batch_index,:]).cuda()
            batch_emb=utterance_tr[batch_index]
            batch_x, batch_y_id, batch_len, batch_ind, batch_label,batch_emb = sort_batch2(batch_x, batch_y_id, \
                                                                                batch_len, batch_ind, batch_label, \
                                                                                batch_emb)
            batch_emb=batch_emb.cuda()
            optimizer2.zero_grad()
            outp=newnet(batch_emb)  
            loss_val2=loss_devise(outp,batch_y_id,data['sc_vec'])
#            print('loss: ',loss_val2)

#            loss_val2=(1-F.cosine_similarity(outp.float().cuda(), batch_label)).mean()
#            print(F.cosine_similarity(outp.float().cuda(), batch_label))
            loss_val2.backward()
            optimizer2.step()
            avg_loss+=loss_val2
            
            total_unseen_pred=np.array([])
            for activation in outp:
                predtemp=activation.cpu().detach().numpy()
                predtemp=np.reshape(predtemp,(1,predtemp.shape[0]))
                sim=cosine_similarity(predtemp,data['sc_vec'])
    #            print(sim)
                prd=np.argmax(sim,1)
                total_unseen_pred = np.concatenate((total_unseen_pred, prd))
                
            acc_tr=accuracy_score(batch_y_id.cpu(),total_unseen_pred)
            avg_acc_tr+=acc_tr

        # test       
        train_time = time.time() - epoch_time
        acc,logC=zsl_evaluate(data,config,mymodel,newnet)
        
        # log and output
        if acc>best_acc:
            best_acc=acc
            config['best_epoch']=epoch
        logForClasses.append(logC)
        print('---exp_no:',rep_no+1 ,'/', rep_num ,'---epoch:',epoch+1, '/', config['num_epochs2'], \
              '---loss: ', round((avg_loss.item()/batch_num),4), \
              '---tr_acc:',round((avg_acc_tr/batch_num),4),'---te_acc: ',round(acc,4),\
              '---train_time:',round(train_time, 4))
        
    avg_acc+=acc
    print('average_acc:',avg_acc/rep_num)
    pickle.dump([config,data['sc_dict'],data['uc_dict'],\
                 logForClasses],open(filename, 'wb'))