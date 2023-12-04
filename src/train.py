from itertools import count
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data_loader, optimizer, loss_function, epoch, task, is_eval=True):
    for epoch_idx in range(epoch):
        model.train()
        batch_idx = 0
        for data, target in data_loader.getBatch():
            data, target = data.to(device), target.to(device)
            # print(data[:3])
            optimizer.zero_grad()
            output = model(data)
            if task == "regression":
                target = target.to(torch.float)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                    epoch_idx, batch_idx * len(data), len(data_loader.train_data),
                    100. * batch_idx / data_loader.n_batches, loss.item()))
            batch_idx += 1
        if is_eval:
            evaluate(model, data_loader, loss_function, task)

def countValue(pred):
    count = [0 for i in range(6)]
    for i in range(len(pred)):
        count[pred[i]] += 1
    print(count)

def evaluate(model, data_loader, loss_function, task):
    model.eval()
    test_loss = 0
    correct = 0
    # print("Evaluating...")
    with torch.no_grad():
        for data, target in data_loader.getBatch(test=True):
            data, target = data.to(device), target.to(device)
            # print(data.shape)
            output = model(data)
            test_loss += loss_function(output, target).item() # sum up batch loss
            if task == 'classification':
                pred = output.argmax(dim=1, keepdim=True)
            elif task == 'regression':
                pred = output.round()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(data_loader.test_data)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.test_data),
        100. * correct / len(data_loader.test_data)))
