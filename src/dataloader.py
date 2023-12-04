import csv, re, random
import numpy as np
from sklearn.model_selection import train_test_split
import torch

stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours', 
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their', 
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once', 
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be', 
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself', 
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn', 
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about', 
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn', 
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

class DataLoader:
    def __init__(self, data_path, word2vec_path, max_length, data_size, batch_size, embedding_method):
        self.max_length = max_length
        self.data_size = data_size 
        self.batch_size = batch_size

        self.word2index, embeddings = self.__readWord2Vec(word2vec_path, embedding_method)
        embeddings = np.array(embeddings)
        pad_emb = np.zeros((1,embeddings.shape[1])) 
        unk_emb = np.mean(embeddings,axis=0,keepdims=True)    
        self.embeddings = np.vstack((embeddings, pad_emb, unk_emb))
        self.word2index['<pad>'] = len(self.word2index)
        self.word2index['<unk>'] = len(self.word2index)
        
        self.data = self.__readData(data_path)
        # print(type(self.data))
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2)
        self.train_data, self.train_label = self.__convertData(self.train_data)
        self.test_data, self.test_label = self.__convertData(self.test_data)
        # self.train_data_len = len(self.train_data)
        # self.test_data_len = len(self.test_data)
        self.n_batches = int(len(self.train_data) / self.batch_size) 
        print('data loaded')

    def __readData(self, path):
        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            cnt = 0
            for item in reader:
                cnt += 1
                if cnt == 1:
                    continue
                data.append({
                    "overall" : int(float(item[0])) - 1,
                    "review" : re.findall(r'[a-zA-Z]+', item[5].lower())
                })
        random.shuffle(data)
        count = [self.data_size // 5 for i in range(5)]
        new_data = []
        for item in data:
            if count[item['overall']] == 0:
                continue
            count[item['overall']] -= 1
            new_data.append(item)

        return new_data

    def __readWord2Vec(self, path, embedding_method):
        word2index = {}
        embeddings = [] 
        with open(path, 'r') as f:
            id = 0
            for line in f:
                line = line.split()
                word2index[line[0]] = id
                if embedding_method == 'glove':
                    embeddings.append([float(val) for val in line[1:]])
                elif embedding_method == 'random':
                    embeddings.append([random.random() for val in range(50)])

                id += 1
        return word2index, embeddings

    def __convertSentence(self, sentence):
        res = []
        for word in sentence:
            if word not in stoplist:
                res.append(self.word2index[word] if word in self.word2index else self.word2index['<unk>'])
            if len(res) == self.max_length:
                return np.array(res)
        if len(res) < self.max_length:
            res += [self.word2index['<pad>']] * (self.max_length - len(res))
        return np.array(res)
    
    def __convertData(self, origin_data):
        data, label = [], []
        for item in origin_data:
            data.append(self.__convertSentence(item['review']))
            label.append(item['overall'])
        return data, label

    def getBatch(self, test=False):
        batch_data, batch_label = [], []
        data = self.train_data if not test else self.test_data
        label = self.train_label if not test else self.test_label

        n_batches = int(len(data) / self.batch_size) 
        for i in range(n_batches):
            batch_data = data[i * self.batch_size : (i+1) * self.batch_size]
            batch_label = label[i * self.batch_size : (i+1) * self.batch_size]
            yield torch.from_numpy(np.array(batch_data)).long(), torch.from_numpy(np.array(batch_label)).long()
