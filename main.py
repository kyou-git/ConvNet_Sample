import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ①data preparetion
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

# ②obtain datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# ③dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ④class definication
'''
convolution network class
'''
class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        
        '''
        convolution layer 1
        input  : 1 channel
        output : 6 channels
        filter : 3x3
        '''
        self.conv1 = nn.Conv2d(1, 6, 3)
        '''
        convolution layer 2
        input  : 6 channel
        output : 16 channels
        filter : 3x3
        '''
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #convolution → relu → pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #convolution → relu → pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #flatten
        x = x.view(-1, self.num_flat_features(x))
        #Linear → relu
        x = F.relu(self.fc1(x))
        #Linear → relu
        x = F.relu(self.fc2(x))
        #Linear → relu
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# ⑤train data   
#instance of convolution network   
net = Convnet()
#creates a loss function in PyTorch
criterion = nn.CrossEntropyLoss()
#create optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

num_epochs = 10

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# ⑥test data
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Test Accuracy: {100*correct/total:.2f}%')