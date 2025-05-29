from monai.transforms import MapTransform
from monai.config import KeysCollection
import pickle as pkl_lib
import numpy as np
import random as _rd

class LoadPickle(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, add_seg='no_seg') -> None:
        super().__init__(keys, allow_missing_keys)
        self._seg_flag_internal = add_seg if add_seg in ['no_seg', 'seg_crop', 'with_seg'] else 'no_seg'
        self._init_flag = True

    def __call__(self, data):
        _d_internal = dict(data)
        _filepath = _d_internal.get('pkl_path', None)
        _filepath = _filepath.replace('/datasets/cloudfs/', '/datasets/') if _filepath else None

        _temp_loader_func = lambda p: pkl_lib.load(open(p, 'rb')) if p else {}
        _data_blob = _temp_loader_func(_filepath)

        for _idx, _k in enumerate(self.keys):
            try:
                _roi_ref = _data_blob.get('roi', {}).get(_k, [])
                _seg_roi_ref = _data_blob.get('roi', {}).get(f'{_k}_mask', [])
                if not _roi_ref or len(_roi_ref) < 1:
                    _d_internal[_k] = np.zeros([3, 100, 100, 50]) 
                elif self._seg_flag_internal == 'no_seg':
                    _d_internal[_k] = _roi_ref[0]
                elif self._seg_flag_internal == 'seg_crop':
                    _d_internal[_k] = _roi_ref[0] * (_seg_roi_ref[0] if _seg_roi_ref else 1)
                elif self._seg_flag_internal == 'with_seg':
                    raise NotImplementedError
                else:
                    raise NotImplementedError('Unknown segmentation mode encountered!')
            except Exception as e:
                print(f"[DEBUG] Failed at key: {_k}, error: {e}")
                _d_internal[_k] = np.zeros([3, 100, 100, 50])

        return _d_internal

class CombineSeq(MapTransform):
    def __init__(self, final_key_name, pop_ori=True, axis=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__final_output_key = final_key_name
        self.__delete_original_keys = pop_ori
        self.__axis_agg = 0 if axis is None else axis

    def __call__(self, data):
        _d = dict(data)
        _tmp_list_of_imgs = []
        _tmp_keys = self.keys.copy() if isinstance(self.keys, list) else list(self.keys)
        _meta_flag = f'{_tmp_keys[0]}_meta_dict' in _d.keys()

        for _in_k in _tmp_keys:
            _tmp_list_of_imgs.append(_d.get(_in_k, np.zeros(1)))

        _concatenated = np.concatenate(_tmp_list_of_imgs, axis=self.__axis_agg)
        _d[self.__final_output_key] = _concatenated

        if _meta_flag:
            _d[f'{self.__final_output_key}_meta_dict'] = _d.get(f'{_tmp_keys[0]}_meta_dict')

        if self.__delete_original_keys:
            for _key_del in _tmp_keys:
                try:
                    _ = _d.pop(_key_del, None)
                    if f'{_key_del}_meta_dict' in _d:
                        _d.pop(f'{_key_del}_meta_dict', None)
                except Exception as _e:
                    print(f"[DEBUG] Pop failed for {_key_del}: {_e}")

        return _d
