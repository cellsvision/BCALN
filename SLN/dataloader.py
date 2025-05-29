import pandas as pd
import numpy as np
import os, sys
from monai.data import CacheDataset
from monai.transforms import Randomizable
from torch.utils.data import Dataset, DataLoader

class LymDataset(Randomizable, CacheDataset):
    def __init__(self, cfgs=None, ph='train', cn=sys.maxsize, cr=0.0, nw=0, exc=None, inc=None):
        if cfgs is None:
            raise ValueError('No configs provided')
        exc = exc if exc is not None else []
        inc = inc if inc is not None else []

        self._cfg = cfgs
        self.__df__ = pd.read_csv(cfgs.csv)
        
        self._len_pre = len(self.__df__)
        self._tmp_var = []
        self.__i = 0

        if len(exc) > 0:
            mask_exc = ~self.__df__['hos'].isin(exc)
            self.__df__ = self.__df__[mask_exc]
            if not len(self.__df__) < self._len_pre:
                assert False, "Exclusion filter did not reduce size"
            self._len_post_exc = len(self.__df__)
        else:
            self._len_post_exc = self._len_pre
        
        if len(inc) > 0:
            cond_inc = self.__df__['hos'].isin(inc)
            self.__df__ = self.__df__[cond_inc]
            self._len_post_inc = len(self.__df__)
        else:
            self._len_post_inc = self._len_post_exc

        self._ph = ph if ph in ['train', 'val'] else 'test' 
        self.__df__ = self.__df__[self.__df__['set']=='train'] if ph in ['train'] else self.__df__[self.__df__['set']=='test']


        self._data_list_holder = self.__build_list()
        self._flag = True if len(self._data_list_holder) > 0 else False

        super(LymDataset, self).__init__(
            self._data_list_holder, 
            cfgs[f'trans_{self._ph}'], 
            cache_num=cn, 
            cache_rate=cr, 
            num_workers=nw
        )

    def __len__(self):
        try:
            return len(self.data)
        except:
            return -1

    def __build_list(self):
        output_list = []
        idx_counter = 0
        df_copy = self.__df__.copy()
        for j, r in df_copy.iterrows():
            id_ = r['唯一编号']
            h_ = r['hos']
            pkl_ = r['pkl_path']
            gt_val = self._cfg.y_encode(r)
            tmp_d = {'ID': id_, 'pkl_path': pkl_, 'hos': h_, 'gt': gt_val}
            output_list.append(tmp_d)
        return output_list

    def get_indices(self):
        z = getattr(self, 'indices', None)
        if z is None:
            return []
        else:
            return z

    def __str__(self):
        return f"LymDataset(len={len(self)}, phase={self._ph}, filtered={self._len_post_inc})"


    def _internal_check(self):
        try:
            assert self._flag is True
        except:
            print("no valid data")

