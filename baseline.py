import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,accuracy_score,matthews_corrcoef,roc_auc_score,roc_curve
import seaborn
seaborn.set(style='whitegrid',font_scale=2.0)
from rdkit import DataStructs
from dataprepare import *
import argparse
from sklearn.svm import SVR
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
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
predict = np.zeros_like(y)

for train,test in kf.split(fps,y):
    xt = fps[train]
    xv = fps[test]
    yt = y[train]
    yv = y[test]
    if args.model == 'SVM': clf = SVR()
    else: clf = lgb.LGBMRegressor(max_depth=5,n_estimators=50)
    clf.fit(xt,yt)
    predict = clf.predict(xv)
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

score = clf.predict(s)
f['score'] = np.dot(x,score)/(x.sum(axis=-1)+1e-5)
f.to_csv(args.taste+'-pred.csv',index=False)