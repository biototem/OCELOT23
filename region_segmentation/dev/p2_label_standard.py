import json
from typing import List

import cv2
import numpy as np

from basic import join
from utils import PPlot


def main():
    source_root = r'/media/predator/totem/jizheng/ocelot2023/'

    MASK_COLORS = np.asarray([
        [  0,   0,   0],
        [255, 255, 255],
    ], dtype=np.uint8)

    LABEL_COLORS = np.asarray([
        [255, 255,   0],
        [255,   0,   0],
    ], dtype=np.uint8)

    for index in range(1, 401):
        code = str(index).zfill(3)

        print(join(source_root, rf'tissue/image/{code}.jpg'))

        # 分割图
        image = cv2.imread(join(source_root, rf'tissue/image/{code}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(join(source_root, rf'tissue/label_origin/{code}.png'), 0)

        mask = (label != 255).astype(np.uint8)
        label = (label - 1) * mask

        # 保存
        cv2.imwrite(join(source_root, rf'tissue/label/{code}.png'), label)
        cv2.imwrite(join(source_root, rf'tissue/mask/{code}.png'), mask)

        # 可视化
        # mask_visual = MASK_COLORS[mask]
        # label_visual = LABEL_COLORS[label]

        # PPlot().add(
        #     image, mask_visual, label_visual
        # ).show()


main()
