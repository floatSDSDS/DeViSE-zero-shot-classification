""" input data preprocess.
"""
import numpy as np
import tool
from gensim.models.keyedvectors import KeyedVectors
import math
import scipy.io as sio

def load_w2v(file_name):
    """ load w2v model
        input: model file name
        output: w2v model
    """
    w2v = KeyedVectors.load_word2vec_format(file_name, binary=False)
    return w2v

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

def load_vec(file_path, w2v, in_max_len, unSeen=True):
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
    ind_del = []
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
            ind_del.append(len(input_y)-1)
            if unSeen:
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
    
    return x_padding, input_y, class_dict, s_len, max_len, ind_del

def get_label(Ybase):
    sample_num = Ybase.shape[0]
    labels = np.unique(Ybase)
    class_num = labels.shape[0]
    
    # get label index
    ind = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        ind[np.hstack(Ybase == labels[i]), i] = 1;
    return ind

def read_datasets(dataSetting):
    if dataSetting['dataset_name']=='dataSMP18.txt':      
        labelorder=[["联络"],["股票"],["健康"],["app"],["电台"],["翻译"],
                    ["飞机"],["电话"],["谜语"],["小说"],["公交"],["新闻"],["抽奖"],
                    ["音乐"],["电影"],["视频"],["日程"],["网站"],["计算"],["短信"],
                    ["地图"],["比赛"],["诗歌"],["火车"],["时间"],["天气"],["email"],
                    ["节目"],["电视频道"],["食谱"]]
    elif dataSetting['dataset_name']=='dataSNIP.txt':
        labelorder=[['search'], ['movie'], ['music'], ['weather'], ['restaurant'],
                    ['playlist'], ['book']]

    unseen_class=dataSetting['unseen_class']
    seen_class=[x for x in labelorder if x not in unseen_class]
    
    data_path = dataSetting['data_prefix'] + dataSetting['dataset_name']
    word2vec_path = dataSetting['data_prefix'] + dataSetting['wordvec_name']

    print("------------------read datasets begin-------------------")
    data = {}
    # load word2vec model
    print('------------------load word2vec begin-------------------')
    w2v = load_w2v(word2vec_path)
    print("------------------load word2vec end---------------------")

    # load normalized word embeddings
    
    data['w2v']=w2v
    embedding = w2v.syn0    
    data['embedding'] = tool.norm_matrix(embedding)
    
    max_len = 0
    if dataSetting['test_mode']==1:
        x, y, class_set, s_lens, max_len, ind_del= load_vec(data_path, w2v, max_len,False)
    else:
        x, y, class_set, s_lens, max_len, ind_del= load_vec(data_path, w2v, max_len,True)
    
# split training set and test set
    
    label_len=len(class_set)    
    no_class_tr = math.ceil(label_len*dataSetting['training_prob'])
    if dataSetting['random_class']:
        np.random.shuffle(class_set)
        seen_class = class_set[0:no_class_tr]
        unseen_class = class_set[no_class_tr:]
    
    ind_te = []
    for i in range(len(unseen_class)):
        ind_te.extend([indx for indx, j in enumerate(y) if j == unseen_class[i][0]])
    
    ind_tr = []        
    if dataSetting['test_mode']==1:
        for i in ind_del:
            ind_tr=[ind for ind in ind_tr if ind!=i]
            ind_te=[ind for ind in ind_te if ind!=i]        
            
    # split samples with seen class into trainingset and test set
        for i in range(len(seen_class)):
            ind_temp = [indx for indx, j in enumerate(y) if j == seen_class[i][0]]
            np.random.shuffle(ind_temp)
            no_sample_temp=int(len(ind_temp)*dataSetting['test_intrain_prob'])
            ind_te_temp=ind_temp[0:no_sample_temp]
            ind_tr_temp=ind_temp[no_sample_temp:]
            ind_te.extend(ind_te_temp)
            ind_tr.extend(ind_tr_temp)
    else: 
        for i in range(len(seen_class)):
            ind_tr.extend([indx for indx, j in enumerate(y) if j == seen_class[i][0]])

    
    x_tr=x[ind_tr,:]
    y_tr=y[ind_tr,:]
    s_len=s_lens[ind_tr]
    data['label_tr']=y_tr
    
    x_te=x[ind_te,:]
    y_te=y[ind_te,:]
    u_len=s_lens[ind_te]
    data['label_te']=y_te
        
    if dataSetting['test_mode']==1:    
        x=np.delete(x,np.array(ind_del),0)      
        y=np.delete(y,np.array(ind_del),0)      
        s_lens=np.delete(s_lens,np.array(ind_del),0)
        
    # pre process seen and unseen labels    
    class_id_startpoint=0
    sc_dict, sc_vec = process_label(seen_class, w2v,class_id_startpoint)
    
    if dataSetting['test_mode']==1:
        uc_dict, uc_vec = process_label(unseen_class, w2v,class_id_startpoint+len(sc_dict))
        uc_dict=dict(sc_dict,**uc_dict)
        uc_vec = np.concatenate([sc_vec,uc_vec],axis=0)  
    else:
        uc_dict, uc_vec = process_label(unseen_class, w2v,class_id_startpoint)

    y_tr=np.ndarray.tolist(y_tr[:,0])
    y_tr=np.asarray([sc_dict[i] for i in y_tr])   
    y_te=np.ndarray.tolist(y_te[:,0])
    y_te=np.asarray([uc_dict[i] for i in y_te])   
    
    data['x_tr'] = x_tr
    data['y_tr'] = y_tr

    data['s_len'] = s_len # number of training examples 
    data['sc_vec'] = sc_vec
    data['sc_dict'] = sc_dict

    data['x_te'] = x_te
    data['y_te'] = y_te

    data['u_len'] = u_len # number of testing examples 
    data['uc_vec'] = uc_vec
    data['uc_dict'] = uc_dict

    data['max_len'] = max_len

    ind = get_label(data['y_tr'])
    data['s_label'] = ind # [0.0, 0.0, ..., 1.0, ..., 0.0]
    
    data['seen_class']=seen_class #' '.join(list(tool.flatten(seen_class)))
    data['unseen_class']=unseen_class #' '.join(list(tool.flatten(unseen_class)))
    data['untrain_classlen'] = len(unseen_class) #XIAOTONG
    print("------------------read datasets end---------------------")
    return data