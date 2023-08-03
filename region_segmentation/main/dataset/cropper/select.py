from typing import List, Dict, Any
import os
import json
import math
import bisect
import numpy as np

from basic import config, join
from utils import magic_iter, Rotator
from .interface import Cropper
from ..adaptor import Adaptor
from ..loader import Loader


# 小图选择器，基于与 ActiveCropper 同样的倾向，但针对图片均为小图的情况，此种情况下要求小图尺寸一致
class SelectCropper(Cropper):
    def __init__(self, adaptor: Adaptor):
        self.adaptor = adaptor
        self.patch_size = config['cropper.patch_size']
        self.weights = config['dataset.active_weight']
        self.cache = {}

    # 加载小图选择预备数组
    def load(self, loader: Loader) -> Dict[str, Any]:
        # 如果内存中有, 直接返回
        if loader.part_id in self.cache:
            return self.cache[loader.part_id]
        # 否则检查硬盘
        cache_path = join('~/cache/select_sample_weights', str(self.weights), f'{loader.part_id}.json')
        # 如果硬盘中有, 加载到内存并返回
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                result = json.load(f)
            self.cache[loader.part_id] = result
            return result
        # 否则就需要读图处理一遍了
        print(f'# -> Select Cropper Initing Data {loader.part_id}')
        l, u, r, d = loader.box

        piece = loader.label({
            'x': l,
            'y': u,
            'degree': 0,
            'scaling': 1,
            'w': r-l,
            'h': d-u,
        })
        # 像素的权重和除以像素数得到的就是当前小图理论上的重现倍率
        weight = sum((piece == i).sum() * p for i, p in enumerate(self.weights))
        result = {
            'weight': weight,
            'multiple_rate': math.ceil(weight / (r - l) / (d - u)),
        }
        # 保存下来
        self.cache[loader.part_id] = result
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=4)
        return result

    def sample(self, loader: Loader, num: int = -1):
        # 理想区域
        l, u, r, d = self.adaptor.get_box(loader)
        result = []
        # 固定遍历
        if num == -1:
            num = self.cache[loader.part_id]['multiple_rate']

        degrees = self.adaptor.get_rotate_degrees()
        result = []
        while len(result) < num:
            for degree in degrees:
                rotate_size_rate = Rotator.size_rate(degree)
                w, h = loader.size
                scaling = max(
                    w * rotate_size_rate / self.patch_size,
                    h * rotate_size_rate / self.patch_size,
                )
                result.append({
                    'part_id': loader.part_id,
                    'name': loader.name,
                    'i': 0,
                    'j': 0,
                    # (x, y) 记录中点坐标
                    'x': w//2,
                    'y': h//2,
                    'w': self.patch_size,
                    'h': self.patch_size,
                    'degree': degree,
                    'scaling': scaling,
                })
        np.random.shuffle(result)
        return result[:num]
