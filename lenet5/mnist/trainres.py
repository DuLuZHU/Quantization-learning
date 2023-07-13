import time
from resnet18 import resnet18

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from torchvision import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=False)

test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=transforms.ToTensor(),
                              download=False)


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

net = resnet18(10)
# print(net)
# print(net(torch.randn([1,1,28,28])))

NUM_EPOCHS = 10

model = resnet18(num_classes=10)

model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

valid_loader = test_loader

def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100, cross_entropy/num_examples
    

start_time = time.time()
train_acc_lst, valid_acc_lst = [], []
train_loss_lst, valid_loss_lst = [], []

for epoch in range(NUM_EPOCHS):
    
    model.train()
    
    for batch_idx, (features, targets) in enumerate(train_loader):
    
        ### PREPARE MINIBATCH
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 300:
            print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                   f'Batch {batch_idx:03d}/{len(train_loader):03d} |' 
                   f' Cost: {cost:.4f}')

    # no need to build the computation graph for backprop when computing accuracy
    model.eval()
    with torch.set_grad_enabled(False):
        train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
        valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
        train_acc_lst.append(train_acc)
        valid_acc_lst.append(valid_acc)
        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
        print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
              f' | Validation Acc.: {valid_acc:.2f}%')
        
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')
  
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

# plt.plot(range(1, NUM_EPOCHS+1), train_loss_lst, label='Training loss')
# plt.plot(range(1, NUM_EPOCHS+1), valid_loss_lst, label='Validation loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross entropy')
# plt.xlabel('Epoch')
# plt.show()

# plt.plot(range(1, NUM_EPOCHS+1), train_acc_lst, label='Training accuracy')
# plt.plot(range(1, NUM_EPOCHS+1), valid_acc_lst, label='Validation accuracy')
# plt.legend(loc='upper left')
# plt.ylabel('Cross entropy')
# plt.xlabel('Epoch')
# plt.show()

model.eval()
with torch.set_grad_enabled(False): # save memory during inference
    test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, DEVICE)
    print(f'Test accuracy: {test_acc:.2f}%')

torch.save(model, './model/resnet18-mnist.pth')

def Convert_ONNX():
    
    model.eval() 

    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True)
    torch.onnx.export(model, dummy_input, "resnet18-mnist.onnx")

    print('Model has been converted to ONNX')

if __name__ == "__main__": 

    model = resnet18(10)
    # model.load_state_dict(torch.load("./model/model-mnist.pth"))
    torch.load("./model/resnet18-mnist.pth")
 
    # Conversion to ONNX 
    Convert_ONNX()