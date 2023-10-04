import torch as t
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary


batch_size = 32
num_classes = 10
learning_rate = 0.001
num_epochs = 10
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


#Import dataset and load it

train = datasets.MNIST(root='./data',
                           train=True,
                           download=True,
                           transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

dataset = DataLoader(train, batch_size=batch_size, shuffle=True)

testdataset = DataLoader(test, batch_size=batch_size, shuffle=True)




class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d(1, 6, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 0), #conv layer 1 (input layer)
            nn.ReLU(), #activation function
            nn.AvgPool2d(2, 2) #pooling layer 1
        )
        # Conv2d(6, 16, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5,1,0),  #conv layer 2
            nn.ReLU(), #activation function
            nn.AvgPool2d(2, 2) #pooling layer 2
        )
        # Linear(16*5*5, 120)
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120), #fully connected layer 1
            nn.ReLU() #activation function
        )
        # Linear(120, 84)
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84), #fully connected layer 2
            nn.ReLU() #activation function
        )
        # Linear(84, 10)
        self.fc3 = nn.Linear(84, num_classes) #fully connected layer 3 (output layer)

    def forward(self, x):
        # Conv2d(1, 6, 5)
        x = self.conv1(x)
        # Conv2d(6, 16, 5)
        x = self.conv2(x)
        # Linear(16*5*5, 120)
        x = x.view(x.size()[0], -1) #reshape tensor into (batch_size, n=16*5*5) i think
        x = self.fc1(x)
        # Linear(120, 84)
        x = self.fc2(x)
        # Linear(84, 10)
        x = self.fc3(x)
        return x


model = LeNet().to(device)

cost = nn.CrossEntropyLoss() #cost function
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate) #optimizer

def train():
    total_step = len(dataset)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataset):
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)
            loss = cost(outputs, labels)

            #backwards pass and optimizer step (learning)
            optimizer.zero_grad() #zeroes out the gradients, removing exisitng ones to avoid accumulation
            loss.backward() #gradient of loss, how much each parameter contributed to the loss
            optimizer.step() #adjusts parameters based on results to minimize loss

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

def test():
    with t.no_grad(): #no gradient descent
        correct, total = 0, 0
        for images, labels in testdataset:
            images = images.to(device) #images
            labels = labels.to(device) #true labels
            outputs = model(images)# passes through images through model
            _, predicted = t.max(outputs.data, 1) #gets max values and their indices
            total += labels.size(0)
            correct += (predicted == labels).sum().item() #gets tensor of boolean, counting all the Trues and the extracting the # of trues as a scalar

        print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))




train()
test()



#summary(model=model, input_size=(1, 1, 28, 28), col_width=20,col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], verbose=0)