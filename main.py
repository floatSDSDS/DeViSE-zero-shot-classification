import os
import time
import pickle
from random import *
import scipy.io as sio
import input_data
import model_torch as model
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import tool
import math

import torch.nn as nn

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import normalize

a = Random()
a.seed(1)

# =============================================================================
# model list
# 
# Caps: 2018 EMNLP Zero-shot User Intent Detection via Capsule Neural Networks
# CapsDim: Our model with attention expansion
# CapsWS: with our similarity and update W  
# CapsAll: with both two modification
# 
# =============================================================================
# Setting here!
test_description='testDeVise'
rep_num = 1
id_split=range(0,10)
# SNIP, SMP18
choose_dataset="SNIP"
# Caps, CapsDim,CapsWS, CapsAll
choose_model=[]
# without seen: 0, with seen: 1, fixed with some classes: -1
dataSetting={}
dataSetting['test_mode']=1
######
dataSetting['random_class']=False
dataSetting['training_prob']=0.8
dataSetting['test_intrain_prob']=0.3

# =============================================================================

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
    dataSetting['unseen_class'] = [['聊天'],['网站'],['email'],['地图'],['时间'],['健康']]

#==============================================================================
    
def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_time = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    s_cnum = np.unique(data['y_tr']).shape[0]
    u_cnum = np.unique(data['y_te']).shape[0]
    config = {}
    config['model_name'] = choose_model
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
    config['num_epochs'] = 20 # number of epochs
    config['max_time'] = max_time
    config['sample_num'] = sample_num #sample number of training data
    config['test_num'] = test_num #number of test data
    config['s_cnum'] = s_cnum # seen class num
    config['u_cnum'] = u_cnum #unseen class num
    config['word_emb_size'] = word_emb_size # embedding size of word vectors (300)
    config['d_a'] = 10 # self-attention weight hidden units number
    config['output_atoms'] = 10 #capsule output atoms
    config['r'] = 3 #self-attention weight hops
    config['num_routing'] = 3 #capsule routing num
    config['alpha'] = 0.001 # coefficient of self-attention loss
    config['margin'] = 1.0 # ranking loss margin
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


def generate_batch(n, batch_size):
    batch_index = a.sample(range(n), batch_size)
    return batch_index

def sort_batch(batch_x, batch_y, batch_len, batch_ind):
    batch_len_new = batch_len
    batch_len_new, perm_idx = batch_len_new.sort(0, descending=True)
    batch_x_new = batch_x[perm_idx]
    batch_y_new = batch_y[perm_idx]
    batch_ind_new = batch_ind[perm_idx]

    return batch_x_new, batch_y_new, \
           batch_len_new, batch_ind_new

def cos_loss(input, target):
    return 1 - F.cosine_similarity(input, target).mean()

if __name__ == "__main__":

    
    # load data
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
    
# Training cycle
    batch_num = int(config['sample_num'] / config['batch_size']+1)

    # load model
    lstm=model.myLSTM(config,embedding).cuda()
    loss_fn = F.cross_entropy
    optimizer = optim.Adam(lstm.parameters(), lr=config['learning_rate'])
    if not os.path.exists(config['ckpt_dir']):
        os.mkdir(config['ckpt_dir'])

    loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)
    hh=np.zeros((config['sample_num'],config['hidden_size']*2))
    H=[None]*config['sample_num']

    print('------------------training begin---------------------')
    print('sample_number=',config['sample_num']," ,batch_size=",config['batch_size'],'batch_num=',batch_num)

# LSTM 1
    for epoch in range(config['num_epochs']):
        avg_acc=0
        epoch_time = time.time()
        lstm.train()
        for batch in range(batch_num):
            
            batch_index = generate_batch(config['sample_num'], config['batch_size'])
            batch_x = x_tr[batch_index]
            batch_y_id = y_tr_id[batch_index]
            batch_len = s_len[batch_index]
            batch_ind = y_ind[batch_index]

            # sort by descending order for pack_padded_sequence
            batch_x, batch_y_id, batch_len, batch_ind = sort_batch(batch_x, batch_y_id, batch_len, batch_ind)

            optimizer.zero_grad()
            lstm.forward(batch_x, batch_len,embedding)
            outp=lstm.final_output
            
            loss_val=loss_fn(outp,batch_y_id.long()).sum()
            loss_val.backward()
            optimizer.step()
            
            hh[batch_index,:]=lstm.hh.cpu().detach().numpy()    
            
            i=0
            for idx in batch_index:
                H[idx]=lstm.H.cpu().detach().numpy()[i]
                i+=1
                
            tr_batch_pred = np.argmax(outp.cpu().detach().numpy(), 1)
            acc = accuracy_score(batch_y_id.cpu(), tr_batch_pred)
            avg_acc+=acc
        avg_acc/=batch_num
        print('acc:',avg_acc)
    print('----saving embeddings----')
    pickle.dump(hh,open(choose_dataset+'_Semb.pkl','wb'))
    pickle.dump(hh,open(choose_dataset+'_Semb.txt','wb'))
    pickle.dump(H,open(choose_dataset+'_Hemb.txt','wb'))
    torch.save(lstm.state_dict(), choose_dataset+'_model.pth')
    Wlstm=lstm.state_dict()
    
#%%
pickle.dump([H,data['label_tr'],data['label_te']],open(choose_dataset+'_Hemb.pkl','wb'))
