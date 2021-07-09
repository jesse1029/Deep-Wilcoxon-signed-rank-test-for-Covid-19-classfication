from data import get_loader
from args import get_args
from tqdm import tqdm
from sklearn.metrics import f1_score
import os

from models import SwinTransformer
from evaluation import evaluation

from torch.optim import AdamW
import torch.nn as nn
import torch

args = get_args()
train_loaders, val_loaders = get_loader(args)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

model = SwinTransformer(img_size=224,
                        patch_size=4,
                        in_chans=3,
                        num_classes=1,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.2,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False)

model.load_state_dict(torch.load('weights/swin_tiny_patch4_window7_224.pt'), strict=False)
model = nn.DataParallel(model)
model = model.cuda()


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

pos_dis = torch.distributions.Normal(1, 0.2)
neg_dis = torch.distributions.Normal(-1, 0.2)

for _ in range(args.epoch):
    train_loss = 0.0
    train_pos = 0.0
    train_neg = 0.0
    
    model.train()
    for pos_img, neg_img in tqdm(zip(train_loaders['pos'], train_loaders['neg'])):
        ct_b, img_b, c, h, w = pos_img.size()
        
        pos_img = pos_img.reshape(-1, c, h, w).cuda()
        neg_img = neg_img.reshape(-1, c, h, w).cuda()
        
        optimizer.zero_grad()
        
        pos_output = model(pos_img)
        neg_output = model(neg_img)

        pos_target = pos_dis.sample(sample_shape=torch.Size([pos_output.shape[0], 1])).cuda()
        neg_target = neg_dis.sample(sample_shape=torch.Size([neg_output.shape[0], 1])).cuda()

        pos_loss = criterion(pos_output, pos_target)
        neg_loss = criterion(neg_output, neg_target)
        
        Loss = pos_loss + neg_loss
        
        Loss.backward()
        optimizer.step()
        
        train_loss += Loss.item()
        train_pos += pos_loss.item()
        train_neg += neg_loss.item()
    
    train_loss /= min(len(train_loaders['pos']), len(train_loaders['neg']))
    train_pos /= min(len(train_loaders['pos']), len(train_loaders['neg']))
    train_neg /= min(len(train_loaders['pos']), len(train_loaders['neg']))
    print('Epoch: ', _)    
    print('Training Loss: {:.6f} \tPos Loss: {:.6f} \tNeg Loss: {:.6f}'.format(
            train_loss, train_pos, train_neg))
    if _ % 5 == 0:
        val_f1 = evaluation(model, val_loaders)    
        if val_f1 > 0.9:
            save_path = 'weight_s0.4/w_mse_epoch_' + str(_) + '_f1_' + str(round(val_f1, 4)) + '.pt'
            torch.save(model.module.state_dict(), save_path)      
            
    #torch.save(model.module.state_dict(), 'w_mse.pt')