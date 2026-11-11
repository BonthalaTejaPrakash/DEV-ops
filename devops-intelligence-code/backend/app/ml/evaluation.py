import numpy as np
def evaluate(y_true, y_prob):
    y_true=np.array(y_true); y_prob=np.array(y_prob); pred=(y_prob>=0.5).astype(int)
    tp=int(((y_true==1)&(pred==1)).sum()); fp=int(((y_true==0)&(pred==1)).sum()); tn=int(((y_true==0)&(pred==0)).sum()); fn=int(((y_true==1)&(pred==0)).sum())
    acc=(tp+tn)/max(1,len(y_true)); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn); f1=2*prec*rec/max(1e-9,prec+rec)
    order=np.argsort(y_prob); ranks=np.empty(len(y_prob)); ranks[order]=np.arange(1,len(y_prob)+1); n_pos=int(y_true.sum()); n_neg=len(y_true)-n_pos; auc=(ranks[y_true==1].sum()-n_pos*(n_pos+1)/2)/max(1,n_pos*n_neg)
    return {'accuracy':round(float(acc),4),'precision':round(float(prec),4),'recall':round(float(rec),4),'f1':round(float(f1),4),'confusion':{'tp':tp,'fp':fp,'tn':tn,'fn':fn},'roc_auc':round(float(auc),4)}
