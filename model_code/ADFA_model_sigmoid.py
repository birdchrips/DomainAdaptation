import torch
from torch import nn
from sklearn.model_selection import train_test_split
import os
import numpy as np



class ADFA_LSTM(nn.Module):
    def __init__(self, hidden_dimension, word_vecs, n_hidden, n_step, dropout_p=0.2):
        super(ADFA_LSTM, self).__init__()
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

class ADFA_LINEAR(nn.Module):
    def __init__(self, n_hidden, nn_hidden=16, nnn_hidden = 8):
        super(ADFA_LINEAR, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_hidden, nn_hidden),
            nn.LeakyReLU(),
            nn.Linear(nn_hidden, nnn_hidden),
            nn.LeakyReLU(),
            nn.Linear(nnn_hidden, 1),
            nn.Sigmoid())
        
    def forward(self, batch):
        out = self.model(batch)
        return out

class ADFA_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, word_to_index):
        self.data = data
        self.label = label
        self.w2i = word_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        out = []

        for k in self.data[i]:
            out.append(self.w2i[int(k)])       
        
        return torch.tensor(out).cuda(), torch.tensor([self.label[i]], dtype = torch.float32).cuda()
   
    
class ADFA_Reader():
    
    def __init__(self):

        adfa_train = self.adfa_read('./Dataset/ADFA-LD/Training_Data_Master/')
        adfa_valid = self.adfa_read('./Dataset/ADFA-LD/Validation_Data_Master/')

        path = './Dataset/ADFA-LD/Attack_Data_Master/'
        attack_path = os.listdir(path)
        
        adfa_attack = self.adfa_attack_read(attack_path[0:60], path)
        self.adfa_list = np.concatenate((adfa_train, adfa_valid, adfa_attack))
        
        self.adfa_label = np.concatenate((np.full(len(adfa_train), 0),
                                          np.full(len(adfa_valid), 0),
                                          np.full(len(adfa_attack), 1),))

    
    slice_size=1000
    
    def adfa_slice(self, slice_size) :
        
        tmp_list = []
        label_list = []
        
        for idx, sentence in enumerate(self.adfa_list) :
            
            k = math.ceil(len(sentence)/slice_size)
            label = self.adfa_label[idx]
            start_idx = 0
            
            for i in range(k):
                if i == k - 1:
                    tmp_list.append(sentence[start_idx:])
                    label_list.append(label)
                else :
                    tmp_list.append(sentence[start_idx:(i+1)*slice_size])
                    label_list.append(label)
                    start_idx = (i+1)*slice_size
        
        
        self.adfa_list = np.concatenate((tmp_list[0:2], tmp_list[2:]))
        self.adfa_label = np.concatenate((label_list[0:2], label_list[2:]))
        
        
    def adfa_read(self, path):
        path_file = os.listdir(path)
        adfa_array = []
        for file in path_file:
            f = open(path + file, 'r').readline()
            adfa_array.append(f.split(" ")[:-1])
        return adfa_array

    def adfa_attack_read(self, attack_path, path):
        for i, at_path in enumerate(attack_path):
            if i == 0:
                adfa_array = self.adfa_read(path + at_path + "/")
            else:
                adfa_array = np.concatenate((adfa_array, self.adfa_read(path + at_path + "/")))
        return adfa_array
    
    def data_split(self) :
        
        X_train, X_test, y_train, y_test = train_test_split(self.adfa_list, self.adfa_label, test_size=0.4, random_state=42)
        X_vali, X_test, y_vali, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
        return X_train, y_train, X_vali, y_vali, X_test, y_test
    