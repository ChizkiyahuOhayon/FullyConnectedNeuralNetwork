# 1. imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 2. create  afully connected neural network
class NN(nn.Module):
    # initialization and structure
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() # class name and instance itself
        self.fc1 = nn.Linear(input_size, 100) # the connection between the first layer and the second layer
        self.fc2 = nn.Linear(100, num_classes) # the connection between the second layer and the third layer

    # data flow
    def forward(self, x): # x.shape: 784
        # only the second layer has activation function
        x = F.relu(self.fc1(x)) # from right to left    pass and processed(by relue)
        x = self.fc2(x)
        return x
# 3. set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4.hyperparameters
input_size = 784 # number of pixels of each mnist image
num_classes = 10 # 10 classes
learning_rate = 0.01
batch_size = 64
num_epochs = 3

# 5. prepare the data
# define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, ), (1, ))
])

# define dataset (ingredients)
data_path_training = "MNIST/train_set/"
data_path_testing = "MNIST/test_set/"
train_set = datasets.MNIST(root=data_path_training, download=True, train=True, transform=transform)
test_set = datasets.MNIST(root=data_path_testing, download=True, train=False, transform=transform)

# define the dataloader(container for the ingredients)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# 6. Initialize the network and define the loss function and  optimizer

model = NN(input_size=input_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 7. Training
for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):
        # move data to cuda explicitly
        data = data.to(device=device)
        labels = labels.to(device=device)

        # [64, 1, 28, 28] -> [64, 784]
        data = data.reshape(data.shape[0], -1)

        # prediction
        scores = model(data) # the first dimension is batch_size by default
        loss = criterion(scores, labels) # calculate loss value according to teh predicted values and real values

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step() # loss get the gradient of all parameters, and optimizer change these parameters according to the gradients

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("***Chech training data accuracy***")
    else:
        print("***Chech testing data accuracy***")
    num_correct = 0
    num_examples = 0
    model.eval() # no batch normalization

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device=device)
            labels = labels.to(device=device)
            features = features.reshape(features.shape[0], -1)
            scores = model(features)
            # scores.shape: [64, 10]
            # scores[0]: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            # we want scores[0] to be: 4(the predicted label)
            _, predictions = scores.max(1) # _ ,sut be 1
            num_correct += (predictions == labels).sum()
            num_examples += predictions.shape[0]
            # demonstration
        print(f"Correct predictions: {num_correct}")
        print(f"Total examples: {num_examples}")
        print(f"Accuracy: {float(num_correct)/float(num_examples) * 100:.2f} %")

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)