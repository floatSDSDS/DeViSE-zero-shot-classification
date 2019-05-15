#%% envs

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from random import *
a = Random()
a.seed(1)

#%% Funcs
def norm_matrix(matrix):
    """Nomralize matrix by column
            input: numpy array, dtype = float32
            output: normalized numpy array, dtype = float32
    """

    # check dtype of the input matrix
    np.testing.assert_equal(type(matrix).__name__, 'ndarray')
    np.testing.assert_equal(matrix.dtype, np.float32)
    # axis = 0  across rows (return size is  column length)
    row_sums = matrix.sum(axis = 1) # across columns (return size = row length)

    # Replace zero denominator
    row_sums[row_sums == 0] = 1
    #start:stop:step (:: === :)
    #[:,np.newaxis]: expand dimensions of resulting selection by one unit-length dimension
    # Added dimension is position of the newaxis object in the selection tuple
    norm_matrix = matrix / row_sums[:, np.newaxis]

    return norm_matrix 

def load_vec(file_path, w2v, in_max_len):
    """ load input data
        input:
            file_path: input data file
            w2v: word2vec model
            max_len: max length of sentence
        output:
            input_x: input sentence word ids
            input_y: input label ids
            s_len: input sentence length
            max_len: max length of sentence
    """
    
    input_x = [] # input sentence word ids
    input_y = [] # input label ids
    s_len = [] # input sentence length
    class_dict=[] 
    max_len = 0

    for line in open(file_path,'rb'):
        arr =str(line.strip(),'utf-8')
        arr = arr.split('\t')
        label = [w for w in arr[0].split(' ')]
        question = [w for w in arr[1].split(' ')]
        if len(label)>1:
            label=[' '.join(label)]
        if not label in class_dict:
            class_dict.append(label)
            
        # trans words into indexes
        x_arr = []
        for w in question:
            if w in w2v.vocab:
                x_arr.append(w2v.vocab[w].index)
        s_l = len(x_arr)
        if s_l <= 1:
            continue
        if in_max_len == 0: # can be specific max len
            if s_l > max_len:
                max_len = s_l
        
        input_x.append(np.asarray(x_arr))
        input_y.append(np.asarray(label))
        s_len.append(s_l)

    # add paddings
    max_len = max(in_max_len, max_len)
    x_padding = []
    for i in range(len(input_x)):
        if (max_len < s_len[i]):
            x_padding.append(input_x[i][0:max_len])
            continue
        tmp = np.append(input_x[i], np.zeros((max_len - s_len[i],), dtype=np.int64))
        x_padding.append(tmp)

    x_padding = np.asarray(x_padding)    
    input_y = np.asarray(input_y)
    s_len = np.asarray(s_len)
    
    return x_padding, input_y, s_len, max_len

def process_label(intents, w2v,class_id_startpoint=0):
    """ pre process class labels
        input: class label file name, w2v model
        output: class dict and label vectors
    """
    class_dict = {}
    label_vec = []
    class_id = class_id_startpoint
    for line in intents:
        # check whether all the words in w2v dict
        line=line[0]
        label = line.split(' ')
        for w in label:
            if not w in w2v.vocab:
                print('not in w2v dict', w)

        # compute label vec
        label_sum = np.sum([w2v[w] for w in label], axis = 0)
        label_vec.append(label_sum)
        # store class names => index
        class_dict[' '.join(label)] = class_id
        class_id = class_id + 1
    return class_dict, np.asarray(label_vec)

def get_label(Ybase):
    sample_num = Ybase.shape[0]
    labels = np.unique(Ybase)
    class_num = labels.shape[0]
    
    # get label index
    ind = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        ind[np.hstack(Ybase == labels[i]), i] = 1;
    return ind

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
#%% Settings
          
data_path='data/dataSNIP.txt'
w2v_path='data/wiki.en.vec'
test_mode=0

config={}
config['hidden_size']=16
config['learning_rate']=0.01
config['batch_size']=50
config['nlayers']=2
config['keep_prob']=0.5
config['num_epochs']=1

if test_mode==0:
    test_intrain_prob=0
else:
    test_intrain_prob=0.3

labelorder=[['search'], ['movie'], ['music'], ['weather'], ['restaurant'],['playlist'], ['book']]
unseen_class=[['playlist'], ['book']]
seen_class=[x for x in labelorder if x not in unseen_class]

#%% input data and split datset
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
emb = w2v.syn0    
emb= norm_matrix(emb)

config['sc_num']=len(seen_class)
config['us_num']=len(unseen_class)
config['vocab_size']=emb.shape[1]
config['emb_size']=emb.shape[0]


max_len = 0
x, y, s_lens, max_len= load_vec(data_path, w2v, max_len)

ind_tr=[]
ind_te=[]
for i in range(len(seen_class)):
    ind_temp = [indx for indx, j in enumerate(y) if j == seen_class[i][0]]
    np.random.shuffle(ind_temp)
    no_sample_temp=int(len(ind_temp)*test_intrain_prob)
    ind_te_temp=ind_temp[0:no_sample_temp]
    ind_tr_temp=ind_temp[no_sample_temp:]
    ind_te.extend(ind_te_temp)
    ind_tr.extend(ind_tr_temp)

x_tr=x[ind_tr,:]
y_tr=y[ind_tr,:]
s_len=s_lens[ind_tr]

x_te=x[ind_te,:]
y_te=y[ind_te,:]
u_len=s_lens[ind_te]

y_tr_ind=get_label(y_tr)

ind_zste=[]
for i in range(len(unseen_class)):
    ind_zste.extend([indx for indx, j in enumerate(y) if j == unseen_class[i][0]])
x_zste=x[ind_zste,:]
y_zste=y[ind_zste,:]
zs_len=s_lens[ind_zste]

# preprocess labels
class_id_startpoint=0
sc_dict, sc_vec = process_label(seen_class, w2v,class_id_startpoint)
uc_dict, uc_vec = process_label(unseen_class, w2v,class_id_startpoint)

y_tr=np.ndarray.tolist(y_tr[:,0])
y_tr=np.asarray([sc_dict[i] for i in y_tr])   
y_te=np.ndarray.tolist(y_te[:,0])
y_te=np.asarray([uc_dict[i] for i in y_te])   

#%% LSTM for utterances
class myLSTM(nn.Module):
    def __init__(self,config,pretrained_embedding = None):
        super(myLSTM, self).__init__()
        
        self.hidden_size = config['hidden_size']
        self.vocab_size = config['vocab_size']
        self.word_emb_size = config['emb_size']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.nlayers=config['nlayers']

        self.word_embedding = nn.Embedding(config['vocab_size'], config['emb_size'])
        self.bilstm = nn.LSTM(config['emb_size'], config['hidden_size'],
                              config['nlayers'], bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(config['keep_prob'])
        self.predict =nn.Linear(config['hidden_size']*2,config['sc_num'])

    def forward(self, input,len,embedding):
        self.s_len=len
        input = input.transpose(0,1) 
        
        if (embedding.nelement() != 0):
            self.word_embedding = nn.Embedding.from_pretrained(embedding)

        emb = self.word_embedding(input)
        packed_emb = pack_padded_sequence(emb, len)

        #Initialize hidden states
        h_0 = torch.zeros(self.nlayers*2, input.shape[1], self.hidden_size)
        c_0 = torch.zeros(self.nlayers*2, input.shape[1], self.hidden_size)
        if torch.cuda.is_available():
            h_0=h_0.cuda()
            c_0=c_0.cuda()
            
        outp = self.bilstm(packed_emb, (h_0, c_0))[0] ## [bsz, len, d_h * 2]
        self.H = pad_packed_sequence(outp)[0].transpose(0,1).contiguous()
        self.pred=F.softmax(self.predict(self.drop(self.H)))
        
    def init_weight(self):
        nn.init.xavier_uniform_(self.predict.weight)
        self.predict.requires_grad_(True)

#%% Train LSTM
config['sample_num']=x_tr.shape[0]
batch_num = int(config['sample_num'] / config['batch_size']+1)
x_tr_ = torch.from_numpy(x_tr)
y_tr_ = torch.from_numpy(y_tr)
y_tr_ind_ = torch.from_numpy(y_tr_ind)
s_len_ = torch.from_numpy(s_len)
emb_ = torch.from_numpy(emb)

model1=myLSTM(config)
optimizer = optim.Adam(model1.parameters(), lr=config['learning_rate'])

for epoch in range(config['num_epochs']):
    
    print('---epoch: ',epoch ,'---')
    model1.train()
    
    for batch in range(batch_num):
        batch_index = generate_batch(config['sample_num'], config['batch_size'])
        batch_x = x_tr_[batch_index]
        batch_y_id = y_tr_[batch_index]
        batch_len = s_len_[batch_index]
        batch_ind = y_tr_ind_[batch_index]
        
        batch_x, batch_y_id, batch_len, batch_ind = sort_batch(batch_x, batch_y_id, batch_len, batch_ind)
        output=model1.forward(batch_x,batch_len,emb_)
