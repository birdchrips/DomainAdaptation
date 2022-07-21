import torch
from torch import nn
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd


class NGIDS_LSTM(nn.Module):
    def __init__(self, hidden_dimension, word_vecs, n_hidden, n_step, dropout_p=0.2):
        super(NGIDS_LSTM, self).__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(word_vecs, dtype=torch.float), freeze=True)
        self.lstm = nn.LSTM(hidden_dimension, n_hidden, n_step, batch_first=True, dropout = dropout_p)
        self.batch_norm = nn.BatchNorm1d(n_hidden)
        
    def forward(self, batch):
        l, batch_x = batch
        emb_out = self.emb(batch_x)
        pack = nn.utils.rnn.pack_padded_sequence(emb_out, l, batch_first =True, enforce_sorted=True)
        packed_out, _ = self.lstm(pack)
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        lstm_out = []
        for idx, l in enumerate(lens_unpacked) :
            lstm_out.append(seq_unpacked[idx, l-1])
        lstm_out = torch.stack(lstm_out)

        return self.batch_norm(lstm_out)

class NGIDS_LINEAR(nn.Module):
    def __init__(self, n_hidden, nn_hidden=16, nnn_hidden = 8):
        super(NGIDS_LINEAR, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_hidden, nn_hidden),
            nn.LeakyReLU(),
            nn.Linear(nn_hidden, nnn_hidden),
            nn.LeakyReLU(),
            nn.Linear(nnn_hidden, 2))
        
    def forward(self, batch):
        out = self.model(batch)
        return out
    
class NGIDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, word_to_index):
        self.data = data
        self.label = label
        self.w2i = word_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        out = []

        for k in self.data[i]:
            out.append(self.w2i[k])
               
        return torch.tensor(out).cuda(), torch.tensor(self.label[i][0], dtype=torch.long).cuda()
    
    
class NGIDS_Reader():
    def __init__(self):

        NGIDS = pd.read_csv('./dataset/NGIDS_host_log_1-99.csv')
        dropna_NGIDS = NGIDS.dropna(subset=['sys_call', 'label'])

        sentence = np.array(dropna_NGIDS['sys_call'].to_list())
        label = np.array(dropna_NGIDS['label'].to_list())

        tmp = label[0]
        idx = 0
        self.sentences = []
        self.labels = []

        slice_size = 1000

        for i in range(len(sentence)) :
            if tmp != label[i] or i-idx >= slice_size:
                tmp = label[i]

                string_sentence = [sentence[j].astype(int) for j in range(idx, i)]
                self.sentences.append(string_sentence)
                self.labels.append(label[idx:i].astype(int).tolist())
                idx = i

        string_sentence = [sentence[j].astype(int) for j in range(idx, len(sentence))]
        self.sentences.append(string_sentence)
        self.labels.append(label[idx:i].astype(int).tolist())
        
        
    def data_split(self) :
        
        X_train, X_test, y_train, y_test = train_test_split(self.sentences, self.labels, test_size=0.4, random_state=42)
        X_vali, X_test, y_vali, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
        return X_train, y_train, X_vali, y_vali, X_test, y_test
        