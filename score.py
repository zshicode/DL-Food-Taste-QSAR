import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,accuracy_score,matthews_corrcoef,roc_auc_score,roc_curve
import seaborn
seaborn.set(style='whitegrid',font_scale=2.0)
from rdkit import DataStructs
from dataprepare import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--data', type=int, default=1,
                    help='Dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='Model')
parser.add_argument('--taste', type=str, default='bitter',
                    help='Taste')               

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

set_seed(args.seed,args.cuda)

def sensitivity(y_true,y_pred):
    return recall_score(y_true,y_pred)

def specificity(y_true,y_pred):
    tn,fp = confusion_matrix(y_true,y_pred)[0]
    if tn+fp!=0: return tn/(tn+fp)
    else: return 1

path = './Data%d/' % args.data
df = pd.read_csv(path+'food-compound.csv')
f = pd.read_csv(path+'food.csv')
c = pd.read_csv(path+'compound.csv')
n = len(f)
m = len(c)
x = np.loadtxt(path+'x.txt')
if path == './Data1/': x = 0.1*(np.log10(x+1e-5)+5)
#s = normalized(np.loadtxt(path+'sim.txt')) # Tanimoto similarity
s = np.loadtxt(path+'fps.txt')
#x = x.dot(s)
tst = pd.read_csv('taste.csv')
fp = tst['smiles'].apply(ecfp)
fpLength = len(fp[0])
fps = np.zeros((len(fp),fpLength))
for i in range(len(fp)):
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp[i],arr)
    fps[i] = arr

y = tst[args.taste].values
kf = StratifiedKFold(n_splits=2,shuffle=True)

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.pool1 = nn.AvgPool1d(4) # d/4 dim
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=1,
            padding=1,
            dilation=2
        )
        self.pool2 = nn.AvgPool1d(4) # d/16 dim
        self.conv = nn.Sequential(
            self.conv1,self.pool1,nn.Tanh(),
            self.conv2,self.pool2,nn.Tanh()
        )
        self.lin = nn.Linear(fpLength//16,1)
    
    def forward(self,x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.lin(x).squeeze()
        x = F.sigmoid(x) 
        return x

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet,self).__init__()
        self.conv1 = nn.LSTM(
            input_size=fpLength,
            hidden_size=args.hidden,
        )
        self.conv2 = nn.LSTM(
            input_size=args.hidden,
            hidden_size=args.hidden,
        )
        self.lin = nn.Linear(args.hidden,1)
    
    def forward(self,x):
        x = F.tanh(self.conv1(x)[0])
        x = F.tanh(self.conv2(x)[0])
        x = x.squeeze()
        x = F.sigmoid(self.lin(x).squeeze())
        return x

predict = np.zeros_like(y)
s = torch.from_numpy(s).float().unsqueeze(1)
fps = torch.from_numpy(fps).float().unsqueeze(1)
y = torch.from_numpy(y).float()

for train,test in kf.split(fps,y):
    yt = y[train]
    yv = y[test]
    if args.model == 'CNN': clf = CNNNet()
    else: clf = LSTMNet()
    if args.cuda:
        clf = clf.cuda()
        fps = fps.cuda()
        s = s.cuda()
        yt = yt.cuda()
    
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    clf.train()
    for e in range(args.epochs):     
        z = clf(fps)
        loss = F.mse_loss(z[train],yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))
    
    clf.eval()
    z = clf(fps)
    if args.cuda:
        z = z.cpu().detach().numpy()
    else:
        z = z.detach().numpy()
    
    predict = z[test]
    predict -= predict.min()
    predict /= predict.max()
    fpr,tpr,th = roc_curve(yv,predict)
    pred = np.ones(len(predict))
    for i in range(len(predict)):
        if predict[i]<th[np.argmax(tpr-fpr)]: pred[i] = 0.0

    confusion = confusion_matrix(yv,pred)
    print("AUROC Sn Sp Pre Acc F1 Mcc")
    res = [
        roc_auc_score(yv,predict),
        sensitivity(yv,pred),
        specificity(yv,pred),
        precision_score(yv,pred),
        accuracy_score(yv,pred),
        f1_score(yv,pred),
        matthews_corrcoef(yv,pred)
    ]
    print(res)
    plt.figure()
    seaborn.heatmap(confusion,annot=True,cbar=False,fmt='d',
        xticklabels=[0,1],
        yticklabels=[0,1])
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.ylim(2,0)
    #plt.tight_layout()
    plt.show()

score = clf(s)
if args.cuda:
    score = score.cpu().detach().numpy()
else:
    score = score.detach().numpy()

predict = np.dot(x,score)/(x.sum(axis=-1)+1e-5)
predict -= predict.min()
predict /= predict.max()
f['score'] = predict
f.to_csv(args.taste+'-pred.csv',index=False)