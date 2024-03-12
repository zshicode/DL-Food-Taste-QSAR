import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,accuracy_score,matthews_corrcoef,roc_auc_score,roc_curve
import seaborn
seaborn.set(style='whitegrid',font_scale=2.0)
from rdkit import DataStructs
from dataprepare import *
from sklearn.svm import SVR
import lightgbm as lgb

def sensitivity(y_true,y_pred):
    return recall_score(y_true,y_pred)

def specificity(y_true,y_pred):
    tn,fp = confusion_matrix(y_true,y_pred)[0]
    if tn+fp!=0: return tn/(tn+fp)
    else: return 1

path = './Data2/'
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

def binary(y,predict,q):
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
    # confusion = confusion_matrix(y,pred)
    # print("AUROC Sn Sp Pre Acc F1 Mcc")
    # print(res)
    # plt.figure()
    # seaborn.heatmap(confusion,annot=True,cbar=False,fmt='d',
    #     xticklabels=[0,1],
    #     yticklabels=[0,1])
    # plt.xlabel('Pred')
    # plt.ylabel('True')
    # plt.ylim(2,0)
    # plt.show()
    return [q]+res

def test(tst,q,frag):
    fp = tst['smiles'].apply(ecfp)
    fpLength = len(fp[0])
    fps = np.zeros((len(fp),fpLength))
    kf = StratifiedKFold(n_splits=2,shuffle=True)
    for i in range(len(fp)):
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp[i],arr)
        fps[i] = arr
    
    y = tst[q].values
    predict = np.zeros((y.shape[0],y.shape[1]))
    for train,test in kf.split(fps,y[:,0]):
        xt = fps[train]
        xv = fps[test]
        yt = y[train]
        clf = lgb.LGBMRegressor(max_depth=5,n_estimators=50,n_jobs=-1)
        clf = MultiOutputRegressor(clf)
        clf.fit(xt,yt)
        predict[test] = clf.predict(xv)
    
    df = pd.DataFrame(columns=['Name','AUROC','Sn','Sp','Pre','Acc','F1','Mcc'])
    for i in range(len(q)):
        df.loc[len(df)] = binary(y[:,i],predict[:,i],q[i])
    
    df.loc[len(df)] = ['Average']+list(df.mean().values)
    score = clf.predict(s)
    score = minmax_scale(score,axis=1)
    c[q] = score
    fq = np.dot(x,score)
    f[q] = minmax_scale(fq,axis=1)
    row = tst[['name','smiles']]
    row[q] = minmax_scale(predict,axis=1)
    if frag:
        df.to_csv('frag-metric.csv',index=False)
        rr = []
        cr = []
        fr = []
        for i in range(len(row)):
            rr.append(row.loc[i][q].sort_values(ascending=False).index[:5])

        for i in range(len(c)):
            cr.append(c.loc[i][q].sort_values(ascending=False).index[:5])
        
        for i in range(len(f)):
            fr.append(f.loc[i][q].sort_values(ascending=False).index[:5])
        
        rank = ['Rank1','Rank2','Rank3','Rank4','Rank5']
        row[rank] = rr
        c[rank] = cr
        f[rank] = fr
        row.to_csv('frag-row-pred.csv',index=False)
        c.to_csv('compound-pred.csv',index=False)
        f.to_csv('food-pred.csv',index=False)
    else:
        df.to_csv('taste-metric.csv',index=False)
        row['MainTaste'] = row[q].idxmax(axis=1)
        c['MainTaste'] = c[q].idxmax(axis=1)
        f['MainTaste'] = f[q].idxmax(axis=1)
        row.to_csv('taste-row-pred.csv',index=False)
        c.to_csv('compound-pred.csv',index=False) 
        f.to_csv('food-pred.csv',index=False)

q = ['bitter','sweet','umami','kokumi','salty','sour','tasteless']
txt = np.loadtxt('frag_dict.txt',dtype=str).tolist()
test(tst,q,False)
test(frag,txt,True)