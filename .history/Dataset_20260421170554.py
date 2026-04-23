# Import dataset MNIST
from torchvision import datasets, transforms    
from torch.utils.data import DataLoader
def get_mnist(batch_size=128):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    # Download and load the training data
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    # Download and load the test data
    testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader