import torch
from torchvision.datasets import MNIST
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable

class FashionMNIST(MNIST):
    """`Fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
resize = transforms.Scale(224)

train_dataset = FashionMNIST(root='../data/fashionmnist',
                             train=True,
                             transform=transforms.Compose([
                                 resize,
                                 transforms.ToTensor(),
                                 normalize,
                                 # Repeat grayscale input to all RGB channels, since pretrained model expects RGB image
                                 lambda x: x.repeat(3, 1, 1)
                             ]
                             ),
                             download=True)

test_dataset = FashionMNIST(root='../data/fashionmnist',
                            train=False,
                            transform=transforms.Compose([
                                resize,
                                transforms.ToTensor(),
                                normalize,
                                # Repeat grayscale input to all RGB channels, since pretrained model expects RGB image
                                lambda x: x.repeat(3, 1, 1)
                            ]
                            ),
                            download=True)

print('Number of train samples: ', len(train_dataset))
print('Number of test samples: ', len(test_dataset))

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class ResNetForMNIST(nn.Module):
    def __init__(self):
        super(ResNetForMNIST, self).__init__()

        self.model = models.resnet18(pretrained=True)
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.model(x)
        out = self.fc1(out)
        return out

model = ResNetForMNIST()

if torch.cuda.is_available():
    model.cuda()
print('model', model)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model.train()

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        print('iter ', iter)
        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                #######################
                #  USE GPU FOR MODEL  #
                #######################
                # Total correct predictions
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))