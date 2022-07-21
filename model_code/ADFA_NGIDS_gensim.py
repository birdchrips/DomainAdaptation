import numpy as np
import pandas as pd
import os

from gensim.models import Word2Vec

class NGIDS_word2vec():
    def __init__(self, NGIDS_path = './dataset/NGIDS_host_log_1-99.csv'):
        NGIDS = pd.read_csv(NGIDS_path)
                
        dropna_NGIDS = NGIDS.dropna(subset=['sys_call', 'label'])
        sentence = np.array(dropna_NGIDS['sys_call'].to_list())
        label = np.array(dropna_NGIDS['label'].to_list())
                
        label = np.array(dropna_NGIDS['label'].to_list())

        tmp = label[0]
        idx = 0
        sentences = []
        labels = []

        for i in range(len(sentence)) :
            if tmp != label[i] :
                tmp = label[i]

                string_sentence = [sentence[j].astype(int) for j in range(idx, i)]
                
                sentences.append(string_sentence)
                labels.append(label[idx:i].astype(int).tolist())
                
                idx = i


        string_sentence = [sentence[j] for j in range(idx, len(sentence))]
        sentences.append(string_sentence)
        labels.append(label[idx:len(sentence)])

        self.sentences = sentences


    def make_vector(self, vector_size = 15, window = 1, save_path = 'NGIDS_word2vec.model'):
                
        model = Word2Vec(sentences=self.sentences, vector_size = vector_size, window = window, min_count = 1, workers = 4 , sg = 1)

        model.save(save_path)


class ADFA_word2vec():
    def __init__(self, ADFA_path = './dataset/ADFA-LD/'):

        adfa_train = self.data_read(ADFA_path + 'Training_Data_Master/')
        adfa_valid = self.data_read(ADFA_path + 'Validation_Data_Master/')

        path = ADFA_path + 'Attack_Data_Master/'
        attack_path = os.listdir(path)
        adfa_attack = self.adfa_attack_read(attack_path, path)

        adfa_list = np.concatenate((adfa_train, adfa_valid, adfa_attack))

        adfa_label = np.concatenate((np.full(len(adfa_train), 0),
                             np.full(len(adfa_valid), 0),
                             np.full(len(adfa_attack), 1)
                            ))
        adfa_list = np.asarray(adfa_list).tolist()
        adfa_label = np.asarray(adfa_label)
        sentences = []

        for x in adfa_list :
            string_sentence = [int(x[j]) for j in range(len(x))]
            sentences.append(string_sentence)
        
        self.sentences = sentences


    def data_read(self, path):
        path_file = os.listdir(path)
        data_array = []
        for file in path_file:
            f = open(path + file, 'r').readline()
            data_array.append(f.split(" ")[:-1])

        return data_array

    def adfa_attack_read(self, attack_path, path):
        for i, at_path in enumerate(attack_path):
            if i == 0:
                adfa_array = self.data_read(path + at_path + "/")
            else:
                adfa_array = np.concatenate((adfa_array, self.data_read(path + at_path + "/")))
        return adfa_array


    def make_vector(self, vector_size = 15, window = 1, save_path = 'ADFA_word2vec.model'):
                
        model = Word2Vec(sentences=self.sentences, vector_size = vector_size, window = window, min_count = 1, workers = 4 , sg = 1)
        model.save(save_path)
        
        


class both_word2vec():
    def __init__(self, NGIDS_path = './dataset/NGIDS_host_log_1-99.csv', ADFA_path = './dataset/ADFA-LD/'):
        self.sentences = self.NGIDS_data_load(NGIDS_path)
        self.sentences += self.ADFA_data_load(ADFA_path)
        
    def NGIDS_data_load(self, NGIDS_path):
        NGIDS = pd.read_csv(NGIDS_path)
                
        dropna_NGIDS = NGIDS.dropna(subset=['sys_call', 'label'])
        sentence = np.array(dropna_NGIDS['sys_call'].to_list())
        label = np.array(dropna_NGIDS['label'].to_list())
                
        label = np.array(dropna_NGIDS['label'].to_list())

        tmp = label[0]
        idx = 0
        sentences = []
        labels = []

        for i in range(len(sentence)) :
            if tmp != label[i] :
                tmp = label[i]

                string_sentence = [sentence[j].astype(int) for j in range(idx, i)]
                
                sentences.append(string_sentence)
                labels.append(label[idx:i].astype(int).tolist())
                
                idx = i


        string_sentence = [sentence[j] for j in range(idx, len(sentence))]
        sentences.append(string_sentence)
        labels.append(label[idx:len(sentence)])

        return sentences
        
    def ADFA_data_load(self, ADFA_path):
        adfa_train = self.data_read(ADFA_path + 'Training_Data_Master/')
        adfa_valid = self.data_read(ADFA_path + 'Validation_Data_Master/')

        path = ADFA_path + 'Attack_Data_Master/'
        attack_path = os.listdir(path)
        adfa_attack = self.adfa_attack_read(attack_path, path)

        adfa_list = np.concatenate((adfa_train, adfa_valid, adfa_attack))

        adfa_label = np.concatenate((np.full(len(adfa_train), 0),
                             np.full(len(adfa_valid), 0),
                             np.full(len(adfa_attack), 1)
                            ))
        adfa_list = np.asarray(adfa_list).tolist()
        adfa_label = np.asarray(adfa_label)
        sentences = []

        for x in adfa_list :
            string_sentence = [int(x[j]) for j in range(len(x))]
            sentences.append(string_sentence)
       
        return sentences

    def data_read(self, path):
        path_file = os.listdir(path)
        data_array = []
        for file in path_file:
            f = open(path + file, 'r').readline()
            data_array.append(f.split(" ")[:-1])

        return data_array

    def adfa_attack_read(self, attack_path, path):
        for i, at_path in enumerate(attack_path):
            if i == 0:
                adfa_array = self.data_read(path + at_path + "/")
            else:
                adfa_array = np.concatenate((adfa_array, self.data_read(path + at_path + "/")))
        return adfa_array


    def make_vector(self, vector_size = 15, window = 1, save_path = 'word2vec.model'):
                
        model = Word2Vec(sentences=self.sentences, vector_size = vector_size, window = window, min_count = 1, workers = 4 , sg = 1)
        model.save(save_path)

