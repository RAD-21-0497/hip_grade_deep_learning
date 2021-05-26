import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from visdom import Visdom
import torch
from torch.autograd import Variable
import numpy as np
np.random.seed(1234)
rng=np.random.RandomState(1234)
import torch.nn.functional as F
import seaborn as sn
from sklearn.metrics import confusion_matrix
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')



def plot_curve(model_id, train_loss, train_acc, test_acc, test_acc_class_list):
    x1 = range(0, len(train_loss))
    y1 = train_loss
    x2 = range(0, len(train_acc))
    y2 = train_acc
    x3 = range(0, len(test_acc))
    y3 = test_acc
    y4 = test_acc_class_list[0]
    x4 = range(0, len(y4))
    y5 = test_acc_class_list[1]
    x5 = range(0, len(y5))
    y6 = test_acc_class_list[2]
    x6 = range(0, len(y6))
    plt.subplot(6, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.subplot(6, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Trian accuracy')
    plt.subplot(6, 1, 3)
    plt.plot(x3, y3, '.-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(6, 1, 4)
    plt.plot(x4, y4, '.-')
    plt.title('Test class0 accuracy vs. epoches')
    plt.ylabel('Test class0 accuracy')
    plt.subplot(6, 1, 5)
    plt.plot(x5, y5, '.-')
    plt.title('Test class1 accuracy vs. epoches')
    plt.ylabel('Test class1 accuracy')
    plt.subplot(6, 1, 6)
    plt.plot(x6, y6, '.-')
    plt.title('Test class2 accuracy vs. epoches')
    plt.ylabel('Test class2 accuracy')
    plt.savefig("../loss_curve/"+model_id + ".jpg")
    plt.show()

def process_label(batch_y):
    label_dict_1={0:0,1:1,2:1,3:1,4:2,5:2,6:2}
    label_dict_2={0:0,1:1,2:1,3:1,4:1,5:1,6:1}
    #label_dict_2={0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:3}
    np_y = batch_y.numpy()
    batch_size = len(batch_y)
    batch_y_1 = torch.zeros(batch_size).type(torch.LongTensor)
    batch_y_2 = torch.zeros(batch_size).type(torch.LongTensor)
    for i,x in enumerate(np_y):
        batch_y_1[i] = label_dict_1[x]
        batch_y_2[i] = label_dict_2[x]
    return batch_y_1,batch_y_2

def get_ci_auc( y_true, y_pred): 
    
    from scipy.stats import sem
    from sklearn.metrics import roc_auc_score 
   
    n_bootstraps = 1500   
    bootstrapped_scores = []   
   
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
#         print("indices:",indices)
#         print(len(indices))
       
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
#         print("score:",score)
        bootstrapped_scores.append(score)   
 
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

   # 90% c.i.
#     confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
#     confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
 
   # 95% c.i.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
   
    return confidence_lower,confidence_upper

def plot_auc_curve(test_loader, test_size, n_classes, net, name, gpu_id,process_label_ID):
    y_test = []
    net.eval()
    print('---------get y_test--------')
    for step ,data in enumerate(test_loader):
        batch_x,batch_y = data
        if process_label_ID > 0:
            batch_y_list = process_label(batch_y)
            batch_y = batch_y_list[process_label_ID-1]
        #_,batch_y = process_label(batch_y)
        y_test.append(batch_y.numpy()[0])
    #load model
    y_pred_prob = np.zeros([test_size,n_classes])
    # test dataset
    print("-----use model predict------")

    for step ,data in enumerate(test_loader):
        #batch_x = torch.tensor(cv2.imread('heatmap7_no_onfh.jpg'))
        batch_x,batch_y=data
        if process_label_ID > 0:
            batch_y_list = process_label(batch_y)
            batch_y = batch_y_list[process_label_ID-1]
        batch_x = batch_x.type(torch.FloatTensor).cuda(gpu_id)
        batch_y = batch_y.type(torch.LongTensor).cuda(gpu_id)
        batch_x,batch_y=Variable(batch_x),Variable(batch_y)
        logit =net(batch_x)
        h_x= F.softmax(logit, dim=1).data.squeeze()
        y_pred_prob[step] = h_x.cpu().numpy()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
    print("-------began plot auc curve------")
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
    for i in range(n_classes):
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

    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])


    # Plot average ROC Curve
    plt.figure(figsize=(10,10))
    plt.plot(fpr["average"], tpr["average"],
             label='Average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["average"]),
             color='deeppink', linestyle=':', linewidth=2)

    #Plot each individual ROC curve
    print("roc auc:",roc_auc)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                label='ROC curve of class {0} (area = {1:0.3f})\n(95% CI:{2:0.3f}-{3:0.3f})'
                ''.format(i, roc_auc[i],lower[i],upper[i]))

    plt.plot([0, 5], [0, 5], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' Hip x onfh grade roc curve')
    plt.legend(loc="lower right")
    plt.savefig(name + '.png')
    plt.show()

def get_confusion_matrix(net,test_loader,labels_name=["oa_I","oa_II","oa_III","onfh_II","onfh_III",
"onfh_IV","ddh_I","ddh_II","ddh_III","ddh_IV"],num_cls=10,gpu_id=0,process_label_id = 0):
    y_test = []
    test_size = len(test_loader.dataset)
    for step ,data in enumerate(test_loader):
        batch_x,batch_y = data
        if process_label_id >0:
            batch_y_list = process_label(batch_y)
            batch_y = batch_y_list[process_label_id-1]
        y_test.append(batch_y.numpy()[0])
    y_pred_prob = np.zeros([test_size,num_cls])
    # test dataset
    print("-----use model predict------")
    net.eval()
    for step ,data in enumerate(test_loader):
        batch_x,batch_y=data
        if process_label_id >0:
            batch_y_list = process_label(batch_y)
            batch_y = batch_y_list[process_label_id-1]
        batch_x = batch_x.type(torch.FloatTensor).cuda(gpu_id)
        batch_y = batch_y.type(torch.LongTensor).cuda(gpu_id)
        logit =net(batch_x)
        h_x= F.softmax(logit, dim=1).data.squeeze()
        y_pred_prob[step] = h_x.cpu().numpy()
        if(step%100 == 0 and step > 0):
            print(step)
    cm = confusion_matrix(y_test, np.argmax(y_pred_prob,1),)
    cm_norm = np.zeros([num_cls,num_cls])
    for i in range(0,num_cls):
        for j in range(0,num_cls):
            cm_norm[i,j] = cm[i,j]/np.sum(cm[i])
    cm_norm = np.around(cm_norm,decimals=2)
    # sn.heatmap(cm_norm,annot=True,xticklabels=labels_name,
    # yticklabels=labels_name,cmap='Purples')
    return cm_norm

def get_class_specific_set(fea,batch_y,args):
    label_set = {0,1,2,3,4,5,6}
    label_to_indices = {label: np.where(batch_y.cpu().numpy() == label)[0]
                         for label in label_set}
    class_specific_batch_y = batch_y[np.hstack((label_to_indices[3],label_to_indices[6]))]
    class_specific_fea = fea[np.hstack((label_to_indices[3],label_to_indices[6]))]
    return class_specific_fea,class_specific_batch_y

def model_ensemble(net_list,net_weight_list,test_loader,gpu_id=0):
    print("test size:",test_size)
    for net in net_list:
        net = net.cuda(gpu_id)
        net.eval()
    for step,data in enumerate(test_loader):
        batch_x,batch_y = data
        batch_x = batch_x.type(torch.FloatTensor).cuda(gpu_id)
        batch_y = batch_y.type(torch.LongTensor).cuda(gpu_id)
        batch_x,batch_y=Variable(batch_x),Variable(batch_y)
        logit_lit=[]
        for net,weight in zip(net_list,net_weight_list):
            logit =net(batch_x)
            h_x += weight * F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        pred = torch.max(out, 1)[1]
            #         print("train label:",batch_y)
            #         print("train pred:",pred)
        res = pred == batch_y
        train_correct = (pred == batch_y).sum()
        for label_idx in range(len(batch_y)):
            label_single = batch_y[label_idx]
            correct[label_single] += res[label_idx].item()
            total[label_single] += 1

        acc_str = ''
        train_acc_2 =0.0
        if iteration%50 == 0:
            for acc_idx in range(num_class):
                try:
                    acc = correct[acc_idx] / total[acc_idx]
                    plotter.plot('acc'+str(acc_idx), 'train', 'Class'+str(acc_idx)+'Accuracy', iteration, acc)
                    train_acc_2 +=acc
                except:
                    acc = 0
                finally:
                    acc_str += ' classID%d acc:%.4f ' % (acc_idx + 1, acc)
            #         print("pred:",pred)
            #         print("batch_y:",batch_y)
            #         print("train_correct:",train_correct)
            train_acc_2 =train_acc_2/float(num_class)
            train_acc += train_correct.cpu().numpy()
    #//TODO