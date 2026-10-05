import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def recall_at_k(sim, labels, ks=(1,5,10)):
    labels=np.asarray(labels); out={}
    order=np.argsort(-sim,axis=1)
    for k in ks:
        hits=[]
        for i in range(len(labels)):
            cand=order[i,:k]
            cand=cand[cand!=i]
            hits.append(labels[i] in labels[cand])
        out[f'Recall@{k}']=float(np.mean(hits))
    return out

def verification_metrics(scores, y):
    scores=np.asarray(scores); y=np.asarray(y)
    auc=roc_auc_score(y,scores)
    fpr,tpr,thr=roc_curve(y,scores)
    i=np.nanargmin(np.abs(fpr-(1-tpr)))
    eer=float((fpr[i]+(1-tpr[i]))/2)
    def tar_at_far(target):
        valid=np.where(fpr<=target)[0]
        return float(tpr[valid[-1]]) if len(valid) else 0.0
    return {'ROC-AUC':float(auc),'EER':eer,'TAR@FAR=1%':tar_at_far(0.01),'TAR@FAR=0.1%':tar_at_far(0.001)}
