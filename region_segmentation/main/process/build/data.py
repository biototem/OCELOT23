import numpy as np
import json
import cv2
import imagesize

from basic import join, config, os
from utils import Assert, Timer, Drawer


def build_data(data: dict, visual: bool = False, T: Timer = None):

    drawer = Drawer(root=config['build.visual_root'])
    if visual:
        Assert.not_none(
            visual_root=config['build.visual_root'],
            fill=config['visual.color.fill'],
            outline=config['visual.color.outline'],
        )
        if T: T.track(f' -> visual at {config["build.visual_root"]}')

    lib = {}
    for name, info in data.items():
        if T: T.track(f' -> building {name}')

        # 获取图片宽高信息
        w, h = imagesize.get(info['image'])
        lib[name] = {
            'name': name,                                   # name 是数据源标识
            'image': info['image'],                         # image 表示图片位置（jpg,png,tif,svs等）
            'image_loader': 'image',                        # 图片识别类型：image/numpy/slide
            'image_zoom': 1,                                # zoom 表示数据源的标准缩放倍率（zoom==2表示此图分辨率更高、需要缩小）
            'label': info['label'],                         # label 表示标签位置（png,npy,shape等）
            'label_loader': 'image',                        # 标签识别类型：image/numpy/shape
            'label_zoom': 1,                                # zoom 表示数据源的标准缩放倍率（zoom==2表示此图分辨率更高、需要缩小）
            'mask_loader': 'image',                         # 标签识别类型：image/numpy/shape
            'mask': info['mask'],                           # label 表示标签位置（png,npy,shape等）
            'mask_zoom': 1,                                 # zoom 表示数据源的标准缩放倍率（zoom==2表示此图分辨率更高、需要缩小）
            'sight': [0, 0, w, h],                          # sight 表示可视区域在数据源中的坐标(l, u, w, h)
            'box': [0, 0, w, h],                            # box 表示采集区域在数据源中的坐标(l, u, w, h)
            'size': [w, h],                                 # size 表示数据源的尺寸
        }
        # 数据可视化 - 按下采样倍率执行
        if visual:
            if T: T.track(f' -> visual image {name}')
            image = cv2.imread(info['image'])
            drawer.name(name)
            drawer.image(image=image, box=None)
            label = cv2.imread(info['label'])[:, :, 0]
            label = np.eye(8)[label]
            drawer.label(label=label, box=None)
            drawer.image_label(image=image, label=label, box=None)
    return lib
