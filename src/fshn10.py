import torch as t
import torch.nn as nn
from lenet import LeNet as LeNetOriginal
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 32
num_classes = 10
learning_rate = 0.001
num_epochs = 15
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# Import dataset and load it
train = datasets.FashionMNIST(root='./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))]))

test = datasets.FashionMNIST(root='./data',
                      train=False,
                      download=True,
                      transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))]))


dataset = DataLoader(train, batch_size=batch_size, shuffle=True)
testdataset = DataLoader(test, batch_size=batch_size, shuffle=True)



model = LeNetOriginal()
#model.load_state_dict(t.load('best_model.pth'))
model.load_state_dict(t.load('frozenfc2_model.pth'))
model = model.to(device)


#Freeze Layers
# for name, param in model.named_parameters():
#     if 'fc2' in name:
#         param.requires_grad = False

for name, param in model.named_parameters():
    if not 'fc2' in name:
        param.requires_grad = False


cost = nn.CrossEntropyLoss()  # cost function
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer

def train():
    print("training...")
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
            #t.save(model.state_dict(), 'frozenfc2_model.pth')
            t.save(model.state_dict(), 'fshn_best_model.pth')
            print("New Lowest Loss in Network: {:.4f} at Epoch [{}/{}]".format(global_lowest_loss, best_epoch, num_epochs))

    print("Lowest Loss in Network {: .4f} at Epoch [{}/{}]".format(global_lowest_loss, best_epoch, num_epochs))

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


def dot_product():
    model.load_state_dict(t.load('fshn_best_model.pth'))
    fc3_weights = model.fc3.weight.data
    model.eval()

    # w1= fc3_weights.data[0]
    # w2 = fc3_weights.data[1]
    # mag_w1 = t.norm(w1)
    # mag_w2 = t.norm(w2)
    # cos_similiarity = t.dot(w1, w2) / (mag_w1 * mag_w2)
    # print(cos_similiarity)

    normalized_weights =  fc3_weights  / t.norm(fc3_weights, dim=1, keepdim=True)
    #print(normalized_weights)
    dot_product_tensor = t.mm(normalized_weights, normalized_weights.t())

    t.set_printoptions(precision=2)

    #cosine_similarity_matrix = t.round(dot_product_tensor, decimals=2)
    #dot_array = dot_product_tensor.numpy()


    print(dot_product_tensor)

if __name__ == '__main__':
    dot_product()
    # train()
    # test()