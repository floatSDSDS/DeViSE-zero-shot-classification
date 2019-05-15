import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

"""
    Parameters:
        Vocab size: T 
        hidden_dim: hidden dimension
        input_dim (D_W): pre-trained using skip-gram model (300)     
"""

class myLSTM(nn.Module):
    def __init__(self, config, pretrained_embedding = None):
        super(myLSTM, self).__init__()
        
        self.cuda_id=config['cuda_id']
        self.hidden_size = config['hidden_size']
        self.vocab_size = config['vocab_size']
        self.word_emb_size = config['word_emb_size']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.s_cnum = config['s_cnum']
        self.nlayers = config['nlayers']
        

        self.word_embedding = nn.Embedding(config['vocab_size'], config['word_emb_size'])
        self.bilstm = nn.LSTM(config['word_emb_size'], config['hidden_size'],
                              config['nlayers'], bidirectional=True, batch_first=True)
        
        self.ws1 = nn.Linear(config['hidden_size'] * 2, config['d_a'], bias=False)
        self.ws2 = nn.Linear(config['d_a'], config['r'], bias=False)
        if torch.cuda.is_available() and False:
            self.ws1=self.ws1.cuda(self.cuda_id)
            self.ws2=self.ws2.cuda(self.cuda_id)
        self.tanh = nn.Tanh()
        
        self.drop = nn.Dropout(config['keep_prob'])
        self.predict =nn.Linear(config['r']*config['hidden_size']*2,config['s_cnum'])
        



    def forward(self, input,len, embedding):
        self.s_len = len
        input = input.transpose(0,1) #(Bach,Length,D) => (L,B,D)
        # Attention
        if (embedding.nelement() != 0):
            self.word_embedding = nn.Embedding.from_pretrained(embedding)

        emb = self.word_embedding(input)
        packed_emb = pack_padded_sequence(emb, len)

        #Initialize hidden states
        h_0 = torch.zeros(self.nlayers*2, input.shape[1], self.hidden_size)
        c_0 = torch.zeros(self.nlayers*2, input.shape[1], self.hidden_size)
        if torch.cuda.is_available():
            h_0=h_0.cuda(self.cuda_id)
            c_0=c_0.cuda(self.cuda_id)
        
        
        outp, (final_hidden_state,final_cell_state) = self.bilstm(packed_emb, (h_0, c_0))## [bsz, len, d_h * 2]
        self.H = pad_packed_sequence(outp)[0].transpose(0,1).contiguous()
        size = self.H.size()
        compressed_embeddings = self.H.view(-1, size[2])  # [bsz * len, d_h * 2]
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))
        alphas = F.softmax(self.ws2(hbar).view(size[0], size[1], -1),dim=1)  # [bsz, len, hop]
        self.attention = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        self.sentence_embedding = torch.bmm(self.attention, self.H)
        
#        self.hh=torch.cat((final_hidden_state[-1],final_hidden_state[-2]),1)
        self.hh=torch.reshape(self.sentence_embedding,[self.sentence_embedding.shape[0],-1])
        final_output=self.predict(self.hh)
        self.final_output=final_output
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.ws1.weight)
        nn.init.xavier_uniform_(self.ws2.weight)
        nn.init.xavier_uniform_(self.predict.weight)

        self.ws1.weight.requires_grad_(True)
        self.ws2.weight.requires_grad_(True)
        self.predict.weight.requires_grad_(True)