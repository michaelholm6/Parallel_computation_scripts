import torchvision
import torch
import torchvision.transforms as transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

#import cifar10 dataset and preprocess it
cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
cifar10_testing = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

#load the resnet50 model
model = torchvision.models.resnet50(weights = None)

#parallelize the model
model = model.to(device)

#initialize the optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

#convert the dataset to a dataloader
cifar10_training = torch.utils.data.DataLoader(cifar10_training, batch_size=4, shuffle=True)
cifar10_testing = torch.utils.data.DataLoader(cifar10_testing, batch_size=4, shuffle=True)

#update the learning rate
learning_rate = 0.1
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    

#train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(cifar10_training):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2 == 0:
            print (loss.item())
            
        if (epoch+1) % 2 == 0:
            learning_rate /= 3
            update_lr(optimizer, learning_rate)
    

#evaluate the model performance with the cifar10 dataset
model.eval()
with torch.no_grad():
    for images, labels in cifar10_testing:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        print(f'Loss: {loss.item()}')



