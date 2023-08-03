import json
from typing import List

import cv2
import numpy as np

from basic import join
from utils import PPlot


def main():
    source_root = r'/media/predator/totem/jizheng/ocelot2023/'
    # visual_root = r'D:\jassorRepository\OCELOT_Dataset\visual'

    with open(r'/media/predator/totem/jizheng/ocelot2023/metadata.json') as f:
        meta = json.load(f)
    sample_pairs = meta['sample_pairs']

    for index in range(1, 401):
        code = str(index).zfill(3)

        print(join(source_root, rf'tissue/image/{code}.jpg'))

        # 分割图
        image_tissue = cv2.imread(join(source_root, rf'tissue/image/{code}.jpg'))
        image_tissue = cv2.cvtColor(image_tissue, cv2.COLOR_BGR2RGB)
        label_tissue = cv2.imread(join(source_root, rf'tissue/label_origin/{code}.png'), 0)

        # 细胞在分割图上的位置
        info = sample_pairs[code]
        left = int((info['patch_x_offset'] - 0.125) * 1024)
        top = int((info['patch_y_offset'] - 0.125) * 1024)
        image_tissue_cell = cv2.resize(
            image_tissue[top: top+256, left: left+256, :],
            (1024, 1024)
        )
        label_tissue_cell = cv2.resize(
            label_tissue[top: top+256, left: left+256],
            (1024, 1024),
            interpolation=cv2.INTER_NEAREST
        )

        # 细胞图
        image_cell = cv2.imread(join(source_root, rf'cell/image/{code}.jpg'))
        image_cell = cv2.cvtColor(image_cell, cv2.COLOR_BGR2RGB)
        with open(join(source_root, rf'cell/label_origin/{code}.csv'), 'r+') as f:
            label_cell = f.readlines()

        # 可视化
        label_tissue_visual = trans_tissue(label_tissue)
        label_tissue_cell_visual = trans_tissue(label_tissue_cell)
        label_cell_visual = trans_cell(image_cell, label_cell)

        PPlot().add(
            image_tissue, image_tissue_cell, image_cell,
            label_tissue_visual, label_tissue_cell_visual, label_cell_visual
        ).show()


COLORS = np.asarray([
    [255, 255, 255],
    [255, 255,   0],
    [255,   0,   0],
], dtype=np.uint8)


def trans_tissue(label: np.ndarray) -> np.ndarray:
    label[label == 255] = 0
    return COLORS[label]


def trans_cell(image: np.ndarray, label: List[str]) -> np.ndarray:
    visual = image.copy()
    for line in label:
        line = line.lstrip()
        if not line:
            continue
        x, y, tp = map(int, line.split(','))
        if tp == 1:
            color = [255, 255,   0]
        elif tp == 2:
            color = [255,   0,   0]
        else:
            raise Exception('鬼知道这里发生了什么')
        cv2.circle(img=visual, center=(x, y), radius=15, color=color, thickness=-1)
    return visual


main()
