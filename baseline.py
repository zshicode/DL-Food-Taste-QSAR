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
parser.add_argument('--frag', type=str, default='alcoholic',
                    help='Fragrance')
parser.add_argument('--all-frag', type=bool, default=False,
                    help='Predicting all fragrance')

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
frag = pd.read_csv('fragrance.csv')

def test(tst,q):
    fp = tst['smiles'].apply(ecfp)
    fpLength = len(fp[0])
    fps = np.zeros((len(fp),fpLength))
    kf = StratifiedKFold(n_splits=2,shuffle=True)
    for i in range(len(fp)):
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp[i],arr)
        fps[i] = arr
    
    y = tst[q].values
    predict = np.zeros(len(y))
    for train,test in kf.split(fps,y):
        xt = fps[train]
        xv = fps[test]
        yt = y[train]
        if args.model == 'SVM': clf = SVR()
        else: clf = lgb.LGBMRegressor(max_depth=5,n_estimators=50,n_jobs=-1)
        clf.fit(xt,yt)
        predict[test] = clf.predict(xv)
    
    predict -= predict.min()
    predict /= predict.max()
    fpr,tpr,th = roc_curve(y,predict)
    pred = np.ones(len(predict))
    for i in range(len(predict)):
        if predict[i]<th[np.argmax(tpr-fpr)]: pred[i] = 0.0

    res = [
        roc_auc_score(y,predict),
        sensitivity(y,pred),
        specificity(y,pred),
        precision_score(y,pred),
        accuracy_score(y,pred),
        f1_score(y,pred),
        matthews_corrcoef(y,pred)
    ]
    if args.all_frag:
        return [q]+res,predict
    else:
        confusion = confusion_matrix(y,pred)
        print("AUROC Sn Sp Pre Acc F1 Mcc")
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
        f.to_csv(q+'-pred.csv',index=False)

#test(tst,args.taste)
#test(frag,args.frag)
if args.all_frag:
    txt = np.loadtxt('frag_dict.txt',dtype=str)
    frags = frag
    df = pd.DataFrame(columns=['Name','AUROC','Sn','Sp','Pre','Acc','F1','Mcc'])
    for t in txt:
        res,score = test(frag,t)
        df.loc[len(df)] = res
        frags[t] = score
    
    ddf = pd.DataFrame(columns=['Name','Rank1','Rank2','Rank3','Rank4','Rank5'])
    for i in range(len(frags)):
        ddf.loc[i] = [frags.loc[i]['name']]+list(frags.loc[i][4:].sort_values(ascending=False).index[:5])
    
    df.loc[len(df)] = ['Average']+list(df.mean().values)
    #df.to_csv('metric.csv',index=False,float_format='%.3f')
    ddf.to_csv('description.csv',index=False)