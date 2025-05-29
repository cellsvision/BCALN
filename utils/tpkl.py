import pickle
import cv2
import numpy as np
import torch

p = './sample_data/ZS2Y11034292.pkl'

with open(p,'rb') as f:
    content = pickle.load(f)


# print(content['roi_info'].keys())
# print(content.keys())
print(content['roi'].keys())

tumor = content['roi']['tumor'][0].numpy()
tumor = tumor/np.max(tumor[0])*255.
print(tumor[0].shape)
print(np.max(content['roi']['tumor_mask'][0].numpy()),np.min(content['roi']['tumor_mask'][0].numpy()))
cv2.imwrite('./tmp_data/im.png',tumor[0,:,:,25])
a = content['roi']['tumor_mask'][0].numpy()[:,:,25]* content['roi']['tumor'][0].numpy()[:,:,:,25]
cv2.imwrite('./im_mask.png',a[0])