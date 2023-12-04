import argparse
import random
import torch
import numpy as np 
from model import *
from dataloader import DataLoader
from train import *

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
# from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', type = str, help = 'input path')
parser.add_argument('-s' ,'--data_size', type = int, default = 50000, help = 'data size')
parser.add_argument('--max_length', type = int, default = 50, help = 'max length of sentence')
parser.add_argument('--channel_num', type = int, default = 2, help = 'channel number')
parser.add_argument('--task', type = str, default = 'classification', help = 'regression or classification')
parser.add_argument('--kernal_size', nargs = '+', type = int, default = [3,4,5], help = 'kernal size')
parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
parser.add_argument('--dropout', type = float, default = 0.5, help = 'dropout rate')
parser.add_argument('--epoch', type = int, default = 20, help = 'epoch')
parser.add_argument('--model', type = str, default = 'CNN', help = 'CNN or RNN')
parser.add_argument('--hidden_size', type = int, default = 100, help = 'hidden size')
parser.add_argument('--num_layers', type = int, default = 2, help = 'number of layer')
parser.add_argument('--embedding_method', type = str, default = 'glove', help = 'glove or random')

args = vars(parser.parse_args())
print(args)



def print_measure(y_true, y_pred):
    print("MAE: ", mean_absolute_error(y_true, y_pred))
    print("MSE: ", mean_squared_error(y_true, y_pred))
    print("ACC: ", accuracy_score(y_true, y_pred))

if __name__ == '__main__':

    data_loader = DataLoader('ML-Exp4/train.csv', 'ML-Exp4/glove.6B.50d.txt', args['max_length'], args['data_size'], args['batch_size'], args['embedding_method'])

    if args['task'] == 'classification':
        loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
    elif args['task'] == 'regression':
        loss_function = torch.nn.MSELoss(reduction='sum')
    
    if args['model'] == 'CNN':
        model = TextCNN(args['channel_num'], args['kernal_size'], args['dropout'], args['task'], len(data_loader.embeddings), args['max_length'], data_loader.embeddings)
    elif args['model'] == 'RNN':
        model = RNN(args['hidden_size'], args['num_layers'], args['dropout'], args['task'], len(data_loader.embeddings), data_loader.embeddings)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, data_loader, optimizer, loss_function, args['epoch'], args['task'])
