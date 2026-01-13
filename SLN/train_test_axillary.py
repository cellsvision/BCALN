import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict as edict
import time
import os
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import pandas as pd

from monai.data import DataLoader, decollate_batch
from monai.transforms import (
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    RandFlipd,
    SpatialPadd,
)
from monai.networks.nets import BasicUNet,HighResNet,DenseNet121,DenseNet169,EfficientNetBN


from dataloader import LymDataset
from transforms import LoadPickle, CombineSeq

import torch_optimizer as optim

import torch.utils.data as data

device = torch.device("cuda")

from cfg_a import C
from m import fetch


if C.mode == 'train':
    model = fetch(mod='a')
elif C.mode == 'continue_train':
    model = fetch(ckpt=C.ckpt, strict=True, dp=True, mod='a')
elif C['mode'] == 'val':
    model = fetch(ckpt=C.ckpt, strict=True, dp=True, mod='a')
else:
    raise NotImplementedError



test_ds = LymDataset(
    cfgs = C,
    ph='test',
    cr=0,
    nw=16,
    exc=[], 
)
print('test length',len(test_ds))

test_loader = DataLoader(test_ds, batch_size=C.bz, shuffle=True, num_workers=8)


if (C.mode == 'train') or (C.mode == 'continue_train'):
    train_ds = LymDataset(
        cfgs = C,
        ph='train',
        cr=0.0,
        nw=16,
    )
    print('train length',len(train_ds))
    
    train_loader = DataLoader(train_ds, batch_size=C.bz, shuffle=True, num_workers=8)

    loss_function = C.loss
    optimizer = C.opt(model.parameters(), C.lr) 
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.eps)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=120,T_mult=1)

    best_metric = -1
    best_metric_epoch = -1
    total_start = time.time()
    writer = SummaryWriter(log_dir=C.workdir+'/'+C.log, flush_secs=20)
    for epoch in range(C.eps):
        epoch_start = time.time()
        print("-" * 10 + f"\nepoch {epoch + 1}/{C.eps}")
        model.train()
        epoch_loss = 0
        step = 0

        lables_all = []
        probs_all = []

        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            axillary, labels = (
                batch_data["axillary"].to(device),
                batch_data["gt"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(axillary)
            loss = loss_function(outputs, labels.float())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            outputs = nn.Softmax(dim=1)(outputs)[:,1]
            lables_all.extend(labels.cpu().numpy()[:,1])
            probs_all.extend(outputs.detach().cpu().numpy())
            try:
                epoch_auc = metrics.roc_auc_score(lables_all,probs_all)
            except Exception as e:
                print(lables_all)
                epoch_auc = 0.1
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", train_auc: {epoch_auc:.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
            writer.add_scalar('step/train_loss', loss, epoch*len(train_ds)//train_loader.batch_size + step)
            writer.add_scalar('step/train_auc', epoch_auc, epoch*len(train_ds)//train_loader.batch_size + step)
        
        lr_scheduler.step()
        writer.add_scalar('epoch_metrics/train_auc', epoch_auc, epoch)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar('epoch_Loss/tain', epoch_loss, epoch)

        torch.save(
            model.state_dict(),
            os.path.join(C.workdir, f"{C.log}_epoch_{epoch}.pth"),
        )


        if epoch>=0:
            model.eval()
            lables_all = []
            probs_all = []
            hos_all = []
            with torch.no_grad():
                step = 0
                for val_data in tqdm(test_loader):
                    step +=1

                    axillary, val_labels, hos = (
                        val_data["axillary"].to(device),
                        val_data["gt"].to(device),
                        val_data["hos"],
                    )
                    hos_all.extend(hos)
                    val_outputs = model(axillary)
                    val_outputs = nn.Softmax(dim=1)(val_outputs)[:,1]
                    lables_all.extend(val_labels.cpu().numpy()[:,1])
                    probs_all.extend(val_outputs.detach().cpu().numpy())     

                try:
                    epoch_auc = metrics.roc_auc_score(lables_all,probs_all)
                except Exception as e:
                    epoch_auc = 0.0
                metric = epoch_auc

                cm = metrics.confusion_matrix(lables_all,np.round(probs_all))
                print(cm)
                sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
                spcificity = cm[0,0] / (cm[0,0] + cm[0,1])

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(C.workdir, f"{C.log}_best_{metric}.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {metric:.4f}"
                    f"\nsensitivity: {sensitivity:.4f} spcificity: {spcificity:.4f}"
                    f"\nbest AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}")
                writer.add_scalar('epoch_metrics/auc', metric, epoch)
                writer.add_scalar('epoch_metrics/sensitivity', sensitivity, epoch)
                writer.add_scalar('epoch_metrics/spcificity', spcificity, epoch)

                lables_all = np.array(lables_all)
                probs_all = np.array(probs_all)
                pred_round = np.round(probs_all)
                hos_all = np.array(hos_all)
                for i in set(hos_all):
                    tmp_lables_all = lables_all[np.where(hos_all==i)]
                    tmp_probs_all = probs_all[np.where(hos_all==i)]
                    tmp_pred_round = pred_round[np.where(hos_all==i)]

                    try:
                        epoch_auc = metrics.roc_auc_score(tmp_lables_all,tmp_probs_all)
                    except Exception as e:
                        epoch_auc = 0.0
                    cm = metrics.confusion_matrix(tmp_lables_all,tmp_pred_round)
                    print(cm)
                    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
                    spcificity = cm[0,0] / (cm[0,0] + cm[0,1])
                    

                for i,content in {
                        'cohort_internal':['center1','center2'],
                        'cohort_external_1':['center3','center4'],
                        'cohort_external_2':['center5','center6'],
                        'cohort_external_3':['center7','center8','center9','center10'],
                        }.items():
                    tmp_lables_all = lables_all[np.where(np.isin(hos_all,content))]
                    tmp_probs_all = probs_all[np.where(np.isin(hos_all,content))]
                    tmp_pred_round = pred_round[np.where(np.isin(hos_all,content))]

                    try:
                        epoch_auc = metrics.roc_auc_score(tmp_lables_all,tmp_probs_all)
                    except Exception as e:
                        epoch_auc = 0.0
                    cm = metrics.confusion_matrix(tmp_lables_all,tmp_pred_round)
                    print(cm)
                    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
                    spcificity = cm[0,0] / (cm[0,0] + cm[0,1])

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start

else:
    model.eval()
    lables_all = []
    probs_all = []
    hos_all = []
    ID_all = []
    with torch.no_grad():
        step = 0
        for val_data in tqdm(test_loader):
            axillary, val_labels, hos, ID = (
                val_data["axillary"].to(device),
                val_data["gt"].to(device),
                val_data["hos"],
                val_data["ID"],
            )
            hos_all.extend(hos)
            ID_all.extend(ID)
            val_outputs = model(axillary)
            val_outputs = nn.Softmax(dim=1)(val_outputs)[:,1]
            lables_all.extend(val_labels.cpu().numpy()[:,1])
            probs_all.extend(val_outputs.detach().cpu().numpy())     

        
        result_df = pd.DataFrame({
            'ID':ID_all,
            'hos':hos_all,
            'prob':probs_all,
            'gt':lables_all,
            'pred':np.round(probs_all),
        })
        result_df.to_csv('./sample_data/tmp_result.csv',index=False)
        
        
        try:
            epoch_auc = metrics.roc_auc_score(lables_all,probs_all)
        except Exception as e:
            epoch_auc = 0.0
        metric = epoch_auc

        cm = metrics.confusion_matrix(lables_all,np.round(probs_all))
        print(cm)
        sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
        spcificity = cm[0,0] / (cm[0,0] + cm[0,1])

        print(
            f"current AUC: {metric:.4f}"
            f"\nsensitivity: {sensitivity:.4f} spcificity: {spcificity:.4f}")


        lables_all = np.array(lables_all)
        probs_all = np.array(probs_all)
        pred_round = np.round(probs_all)
        hos_all = np.array(hos_all)
        for i in set(hos_all):
            tmp_lables_all = lables_all[np.where(hos_all==i)]
            tmp_probs_all = probs_all[np.where(hos_all==i)]
            tmp_pred_round = pred_round[np.where(hos_all==i)]

            try:
                epoch_auc = metrics.roc_auc_score(tmp_lables_all,tmp_probs_all)
            except Exception as e:
                epoch_auc = 0.0
            cm = metrics.confusion_matrix(tmp_lables_all,tmp_pred_round)
            print(cm)
            sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
            spcificity = cm[0,0] / (cm[0,0] + cm[0,1])

            print(
                f"======= {i} ======="
                f"\n CM: \n{cm}"
                f"\nsensitivity: {sensitivity:.4f} spcificity: {spcificity:.4f}  total: {np.sum(cm)}"
                f"\n AUC: {epoch_auc:.4f}")

            

        for i,content in {
                'cohort_internal':['center1','center2'],
                'cohort_external_1':['center3','center4'],
                'cohort_external_2':['center5','center6'],
                'cohort_external_3':['center7','center8','center9','center10'],
                }.items():
            tmp_lables_all = lables_all[np.where(np.isin(hos_all,content))]
            tmp_probs_all = probs_all[np.where(np.isin(hos_all,content))]
            tmp_pred_round = pred_round[np.where(np.isin(hos_all,content))]

            try:
                epoch_auc = metrics.roc_auc_score(tmp_lables_all,tmp_probs_all)
            except Exception as e:
                epoch_auc = 0.0
            cm = metrics.confusion_matrix(tmp_lables_all,tmp_pred_round)
            print(cm)
            sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
            spcificity = cm[0,0] / (cm[0,0] + cm[0,1])

            print(
                f"================ {i} ==============="
                f"\n CM: \n{cm}"
                f"\nsensitivity: {sensitivity:.4f} spcificity: {spcificity:.4f}  total: {np.sum(cm)}"
                f"\n AUC: {epoch_auc:.4f}")
    
