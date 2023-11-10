import torch as t
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 32
num_classes = 10
learning_rate = 0.001
num_epochs = 13
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# Import dataset and load it

train = datasets.CIFAR10(root='./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                                     transforms.Normalize((.5071,.4865,.4409), (.2675,.2565,.2761))]))

test = datasets.CIFAR10(root='./data',
                      train=False,
                      download=True,
                      transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                                    transforms.Normalize((.4914,.4822,.4465), (.2023,.1994,.2010))]))

dataset = DataLoader(train, batch_size=batch_size, shuffle=True)

testdataset = DataLoader(test, batch_size=batch_size, shuffle=True)


class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(BasicBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels))
    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu2(x)
        return x
class ResNet18(nn.Module):
    def __init__(self):
        super (ResNet18, self).__init__()

        self.conv1 =  nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layers(BasicBlock,64,2,stride=1)
        self.layer2 = self.make_layers(BasicBlock,128,2,stride=2)
        self.layer3 = self.make_layers(BasicBlock,256,2,stride=2)
        self.layer4 = self.make_layers(BasicBlock,512,2,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512,num_classes)

    def make_layers(self, block, out_channels, blocks, stride=1):
        layers = [block(64, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)

        return x



num_classes = 10
model = ResNet18().to(device)
cost = nn.CrossEntropyLoss()  # cost function
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer


def train():
    total_step = len(dataset)
    global_lowest_loss = float('inf')
    best_epoch = 1
    for epoch in range(num_epochs):
        avg_loss = 0
        #lowest_loss = float('inf')
        for i, (images, labels) in enumerate(dataset):

            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = cost(outputs, labels)

            # backwards pass and optimizer step (learning)
            optimizer.zero_grad()  # zeroes out the gradients, removing exisitng ones to avoid accumulation
            loss.backward()  # gradient of loss, how much each parameter contributed to the loss
            optimizer.step()  # adjusts parameters based on results to minimize loss
            avg_loss += loss.item()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        avg_loss /= len(dataset)
        if avg_loss < global_lowest_loss:
            global_lowest_loss = avg_loss
            best_epoch = epoch + 1
            t.save(model.state_dict(), 'best_model.pth')
            print("New Lowest Loss in Network: {:.4f} at Epoch [{}/{}]".format(global_lowest_loss, best_epoch, num_epochs))

    print("Lowest Loss in Network {:.4f} at Epoch [{}/{}]".format(global_lowest_loss, best_epoch, num_epochs))


def test():
    with t.no_grad():  # no gradient descent
        correct, total = 0, 0
        for images, labels in testdataset:
            images = images.to(device)  # images
            labels = labels.to(device)  # true labels
            outputs = model(images)  # passes through images through model
            _, predicted = t.max(outputs.data, 1)  # gets max values and their indices
            total += labels.size(0)
            correct += (
                        predicted == labels).sum().item()  # gets tensor of boolean, counting all the Trues and the extracting the # of trues as a scalar

        print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))



if __name__ == '__main__':
    train()
    test()
