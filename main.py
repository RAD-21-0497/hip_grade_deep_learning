import os
import argparse
import torch
from torchvision import transforms, utils
import pretrainedmodels
import random
from torch import nn
import torch.distributed as dist
import numpy as np
from dataset import get_train_dataloader, get_test_dataloader
from config import load_config

torch.manual_seed(1)
torch.random.manual_seed(1)
random.seed(232)

parser = argparse.ArgumentParser(description='Hip classification Training')
parser.add_argument('--config', default='train_config.yaml',type=str, help='Path to the YAML config file', required=True)                    

def train_model(epoch, model, train_loader, optimizer, loss_func, config):
    """training model"""
    model.train()
    print("began train model")
    label_count = [0]*config['num_class']
    print("label count",label_count)
    loss_list = []
    print("*"*20,"began train epoch:{}".format(epoch),"*"*20)
    for step,data in enumerate(train_loader):
        img, label = data
        img, label = img.to(config['device']),label.to(config['device'])
        pred = model(img)
        loss = loss_func(pred, label)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("step:{}, loss:{:.4f}".format(step, loss.item()))
        label_count = [label_count[i]+torch.sum(label==i).item() for i in range(0,config['num_class'])]
        #print("label count:",label_count)
    print("epoch:{}, loss:{:.4f}, label_count:{}".format(epoch, np.mean(loss_list), label_count))
    torch.save(model.state_dict(),config['model_save_path']+'/epoch_{}.pth'.format(epoch))

def test_model(model, test_dataloader, config, phase='test'):
    """evaluating model"""
    print("*"*20,"validate model","*"*20)
    model.eval()
    acc_list = []
    class_correct = list(0. for i in range(config['num_class']))
    class_total = list(0. for i in range(config['num_class']))
    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            imgs,labels = data
            imgs, labels = imgs.to(config['device']), labels.to(config['device'])
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(config['num_class']):
        acc = class_correct[i] / class_total[i]
        acc_list.append(acc)
        print('{} acc: {:.3f}'.format(config['class_name'][i], acc))
    print("{} acc mean:{}".format(phase, np.mean(acc_list)))


if __name__ == '__main__':
    args = parser.parse_args()
    config = load_config(args.config)
    max_epochs = config['trainer']['max_num_epochs']
    print(config)
    print(max_epochs)
    #wandb    # construct model 
    print("-------------------use  xception-------------------")
    model = pretrainedmodels.__dict__['xception'](num_classes=1000,
                                                      pretrained='imagenet')
    num_fc = model.last_linear.in_features
    model.last_linear = nn.Linear(num_fc, config['num_class'])
    model.to(config['device'])

    if not os.path.exists(config['model_save_path']):
        os.mkdir(config['model_save_path'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    class_num = config['class_num']
    #class_weight = [sum(class_num)/num for num in class_num]
    class_weight = config['class_weight']
    print("class weight:", class_weight)
    class_weight = torch.FloatTensor(class_weight).to(config['device'])
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weight)

    # get dataloader
    train_loader = get_train_dataloader(config['dataloaders'], num_class=config['num_class'])
    val_loader = get_test_dataloader(config['dataloaders'], phase='val')
    test_loader = get_test_dataloader(config['dataloaders'], phase='test')
    print("sample data shape: ",next(iter(train_loader))[0].shape)
    for epoch in range(0, max_epochs):
        train_model(epoch, model, train_loader, optimizer, loss_func, config)
        test_model(model, val_loader, config, phase='val')
        test_model(model, test_loader, config, phase='test')
