# -*- coding: utf-8 -*-  
#%matplotlib inline
import sys
import pretrainedmodels
import numpy as np
import seaborn as sn
import torchvision
from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys 
from sklearn.metrics import confusion_matrix, roc_curve, auc, multilabel_confusion_matrix

from utils import get_ci_auc
import pandas as pd
import shutil

def test_model(model, test_dataloader, labels_name, num_class=7, phase='test', device=torch.device("cuda:0")):
    """evaluating model"""
    print("*"*20,"validate model","*"*20)
    model.eval()
    acc_list = []
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))
    data_samples = test_dataloader.dataset.samples
    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            imgs,labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            # if not c.item():
            #     img_file = data_samples[step][0]
            #     print("data samples:",data_samples[step],"pred:",c.item())
            #     shutil.move(img_file, img_file.replace("/val/","/val_bad/"))
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(num_class):
        acc = class_correct[i] / class_total[i]
        acc_list.append(acc)
        print('{} acc: {:.3f}'.format(labels_name[i], acc))
    print("{} acc mean:{}".format(phase, np.mean(acc_list)))


def get_confusion_matrix(net,test_loader,labels_name,num_cls=7,gpu_id=0):
    net.eval()
    y_test = []
    test_size = len(test_loader.dataset)
    print("---------------get ground truth label--------")
    for step ,data in enumerate(tqdm(test_loader)):
        batch_x,batch_y = data
        y_test.append(batch_y.numpy()[0])
    y_pred_prob = np.zeros([test_size,num_cls])
    # test dataset
    print("---------------use model predict-------------")
    for step ,data in enumerate(tqdm(test_loader)):
        batch_x,batch_y=data
        batch_x = batch_x.type(torch.FloatTensor).cuda(gpu_id)
        batch_y = batch_y.type(torch.LongTensor).cuda(gpu_id)
        logit =net(batch_x)
        h_x= F.softmax(logit, dim=1).data.squeeze()
        y_pred_prob[step] = h_x.cpu().numpy()
    cm = confusion_matrix(y_test, np.argmax(y_pred_prob,1),)
    multi_class_cm = multilabel_confusion_matrix(y_test, np.argmax(y_pred_prob,1))
    print("multi class confusion matrix:", multi_class_cm)
    cm_norm = np.zeros([num_cls,num_cls])
    for i in range(0,num_cls):
        for j in range(0,num_cls):
            cm_norm[i,j] = cm[i,j]/np.sum(cm[i])
    cm_norm = np.around(cm_norm,decimals=2)
    fig = sn.heatmap(cm_norm,annot=True,xticklabels=labels_name,
    yticklabels=labels_name,cmap='Purples')
    heatmap_fig = fig.get_figure()
    heatmap_fig.savefig("../result_pic_7cls_20210519/confusion_matrix.png", dpi = 400)
    return cm_norm

def plot_auc_curve(model, test_loader, labels_name, num_class=7, gpu_id=0):
    model.eval()
    print("began plot auc curve")
    test_size = len(test_loader)
    #load model
    y_pred_prob = np.zeros([test_size,num_class])
    # test dataset
    print("-----use model predict------")
    y_test = []
    with torch.no_grad():
        for step ,data in enumerate(tqdm(test_loader)):
            batch_x,batch_y=data
            y_test.append(batch_y.numpy()[0])
            batch_x = batch_x.type(torch.FloatTensor).cuda(gpu_id)
            batch_y = batch_y.type(torch.LongTensor).cuda(gpu_id)
            logit =model(batch_x)
            h_x= F.softmax(logit, dim=1).data.squeeze()
            y_pred_prob[step] = h_x.cpu().numpy()
            probs, idx = h_x.sort(0, True)
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()
    cm = confusion_matrix(y_test,np.argmax(y_pred_prob, 1),)
    print("cm:\n", cm)
    # compute auc
    y_test = np.array(y_test)
    y_predict_proba = y_pred_prob
    plt.style.use('ggplot')
    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lower = {0:0.916,1:0.842,2:0.962}
    upper = {0:0.944,1:0.878,2:0.978}
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(num_class):
        y_test_i = y_test == i
        y_test_i = y_test_i.astype(int)
        #y_test_i = y_test[lambda x: 1 if x == i else 0]
        all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
        all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
        fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
        lower_bound,upper_bound = get_ci_auc(y_test_i, y_predict_proba[:, i])
        lower[i] = lower_bound
        upper[i] = upper_bound
        roc_auc[i] = auc(fpr[i], tpr[i])
        print("{}, auc:{}".format(labels_name[i], roc_auc[i]))
    # get doctor anno
    oa_anno_df = pd.read_excel("../data/doctor_anno/oa_roc.xlsx")
    onfh_anno_df = pd.read_excel("../data/doctor_anno/onfh_roc.xlsx")
    print("oa_anno_df:", oa_anno_df)
    print("onfh_anno_df:", onfh_anno_df)

    # began plot auc curve
    label_dict = {0:"Normal",1:"PHOA_I",2:"PHOA_II",3:"PHOA_III",4:"ONFH_II",5:"ONFH_III",
    6:"ONFH_IV"}
    font = {'family':'Times New Roman','weight':'normal','size':25,}
    font_legend = {'family':'Times New Roman','weight':'normal','size':18,}
    for i in range(1,7):
        k=i
        df = oa_anno_df
        if i>3:
            k = i-2 
            df = onfh_anno_df
        plt.figure(figsize=(10,10))# plt.plot(fpr["average"], tpr["average"],
        plt.plot(fpr[i], tpr[i], lw=2,color='green',
                label='ROC curve of {0} (area = {1:0.3f})\n(95% CI:{2:0.3f}-{3:0.3f})'
                ''.format(label_dict[i], roc_auc[i],lower[i],upper[i]))
        plt.plot(df.iloc[5]['1-'+str(k)+'_spec'], df.iloc[5][str(k)+'_sens'],'*',marker = "$\\bigotimes$",markersize=20,color = 'red',label='Average individual orthopedic surgeons(>=5yr)')  # 绘制紫红色的圆形的点
        plt.plot(df.iloc[11]['1-'+str(k)+'_spec'], df.iloc[11][str(k)+'_sens'],'*',marker = "$\\bigotimes$",markersize=20,color = 'blue',label='Average individual orthopedic surgeons(<5yr)')  # 绘制紫红色的圆形的点
        plt.plot(df.iloc[12]['1-'+str(k)+'_spec'], df.iloc[12][str(k)+'_sens'],'*',marker = "$\\bigotimes$",markersize=20,color = 'purple',label='Average orthopedic surgeons')  # 绘制紫红色的圆形的点
        #doctor point
        plt.plot(df.iloc[0]['1-'+str(k)+'_spec'], df.iloc[0][str(k)+'_sens'],'*',marker = "$\\times$",markersize=14,color = 'green',label='individual orthopedic surgeons')  # 绘制紫红色的圆形的点
        for j in range(0,11):
            if j==5:
                continue
            plt.plot(df.iloc[j]['1-'+str(k)+'_spec'], df.iloc[j][str(k)+'_sens'],'*',marker = "$\\times$",markersize=14,color = 'green')  # 绘制紫红色的圆形的点
        plt.xlim([-0.005, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity',font)
        plt.ylabel('Sensitivity',font)
        plt.title('Class '+label_dict[i],font)
        plt.legend(loc="lower right",prop=font_legend)
        plt.tick_params(labelsize=23)
        plt.savefig('../result_pic/roc_fig'+str(i)+'_0518.jpg')
        plt.show()



if __name__ == "__main__":
    gpu_id = 0
    resize_size=(256,256)
    mean = [0.605,0.605,0.605]
    std = [0.156,0.156,0.156]
    test_data_path = '../data/512_2-1_train_3cls/test/'
    #test_data_path = "/home/gdp/luckie/data/512_2-1_train_3cls_new_20210517/val/"
    label_dict={0:"normal",1:"oa_I",2:"oa_II",3:"oa_III",4:"onfh_II",5:"onfh_III",
        6:"onfh_IV"} 
    test_data = torchvision.datasets.ImageFolder(test_data_path,
                                                transform=transforms.Compose(
                                                    [
                                                        transforms.Resize(resize_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean,std)
                                                    ]))

    test_loader=DataLoader(test_data,batch_size=1,shuffle=False)
    test_size = len(test_data)
    print(test_data.class_to_idx)
    print("test_size:",test_size)
    #labels_name=["normal","oa_I","oa_II","oa_III","onfh_II","onfh_III","onfh_IV"]
    labels_name = ["Normal", "OA", "ONFH"]

    #epoch38
    #epoch39
    for epoch in  range(66,67):
        print("*"*20,"epoch:",epoch,"*"*20)
        model_path ='./models/hip_classification_3cls_resize256/epoch_{}.pth'.format(epoch)
        model = pretrainedmodels.__dict__['xception'](num_classes=1000,pretrained='imagenet')
        num_fc = model.last_linear.in_features
        model.last_linear = nn.Linear(num_fc, 3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda: 0")))
        model= model.cuda(gpu_id)
        test_model(model, test_loader, num_class=3, labels_name=labels_name)
        cm_norm = get_confusion_matrix(model, test_loader, labels_name=labels_name, num_cls=3)
        print("confusion matrix:\n", cm_norm)
        plot_auc_curve(model, test_loader, labels_name, num_class=3)
