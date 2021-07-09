from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
import torch
import numpy as np

def wilcoxon_rank_test(pop):
    postive_pop = pop[(pop >= 1 - np.sqrt(0.16) * 2) & (pop <= 1 + np.sqrt(0.16) * 2)]
    negative_pop = pop[(pop >= -1 - np.sqrt(0.16) * 2) & (pop <= -1 + np.sqrt(0.16) * 2)]
    total_pop = len(postive_pop) + len(negative_pop)
    if total_pop == 0:
        return 1.0
    else:
        w, p = wilcoxon(np.concatenate((postive_pop, negative_pop)), alternative='greater')
        return p
       
def evaluation(model, val_loaders):
    val_acc = 0
    pos_acc = 0
    neg_acc = 0
    true_label = []
    pred_label = []
    
    model.eval()
    for pos_img in val_loaders['pos']:
        pos_img = pos_img.cuda().squeeze(0)
        pos_output = model(pos_img)
        pop = pos_output.cpu().flatten().detach().numpy()
        #ratio = torch.sum(pos_output > 0).item() / len(pos_output) 
        p_value = wilcoxon_rank_test(pop)
        if p_value < 0.01:
            pos_acc += 1
            pred_label.append(1)
        else:
            pred_label.append(0)    
        true_label.append(1)
        
    for neg_img in val_loaders['neg']:
        neg_img = neg_img.cuda().squeeze(0)
        neg_output = model(neg_img)
        #ratio = torch.sum(neg_output > 0).item() / len(neg_output)
        pop = neg_output.cpu().flatten().detach().numpy()
        p_value = wilcoxon_rank_test(pop)
        if p_value < 0.01:
            pred_label.append(1)
        else:
            neg_acc += 1
            pred_label.append(0)
        true_label.append(0)
        
    val_acc = pos_acc + neg_acc
    val_acc /= (len(val_loaders['pos']) + len(val_loaders['neg']))
    pos_acc /= len(val_loaders['pos'])
    neg_acc /= len(val_loaders['neg'])
    val_f1 = f1_score(true_label, pred_label, average='macro')
    
    print('Val F1: {:.6f} \tVal Acc: {:.6f} \tPos Acc: {:.6f} \tNeg Acc: {:.6f}'.format(
            val_f1, val_acc, pos_acc, neg_acc))
    return val_f1
