import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
net = Convnet()
params = list(net.parameters())
print(len(params))