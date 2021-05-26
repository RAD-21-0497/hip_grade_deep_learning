import torch
import random
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

def get_train_dataloader(config,num_class=7):
    data_path = config['data_path']
    mean_std = config['mean_std'].split(",")
    mean = float(mean_std[0])
    std = float(mean_std[1])
    mean = [mean, mean, mean]
    std = [std, std, std]
    resize = (config['resize'], config['resize'])
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(500),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean,
                                 std)
        ])
    train_data = torchvision.datasets.ImageFolder(data_path+'/train/',
                                                transform=train_transform)
    train_data_samples = train_data.samples
    train_data_samples.sort(key=lambda  d:d[1])
    normal_samples = train_data_samples[0:38561]
    diease_samples = train_data_samples[38561:]
    normal_samples_5000 = random.sample(normal_samples,5000)
    train_samples = normal_samples_5000+diease_samples
    train_data.samples = train_samples
    train_loader = DataLoader(train_data, batch_size=config['batch_size'],shuffle=True)
    if num_class==3:
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    return train_loader

def get_test_dataloader(config, phase='test'):
    data_path = config['data_path']
    mean_std = config['mean_std'].split(",")
    mean = float(mean_std[0])
    std = float(mean_std[1])
    mean = [mean, mean, mean]
    std = [std, std, std]
    resize = (config['resize'], config['resize'])
    test_transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,
                                    std)
            ])
    test_data = torchvision.datasets.ImageFolder(data_path+"/{}/".format(phase),
                                                transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    return test_loader