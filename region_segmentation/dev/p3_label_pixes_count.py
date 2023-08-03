import json
from typing import List

import cv2
import numpy as np

from basic import join


def main():
    source_root = r'/media/predator/totem/jizheng/ocelot2023/'
    # visual_root = r'D:\jassorRepository\OCELOT_Dataset\visual'

    all_counts = np.zeros(shape=(3,), dtype=np.int64)
    for index in range(1, 401):
        code = str(index).zfill(3)

        print(join(source_root, rf'tissue/image/{code}.jpg'))

        # 分割图
        label_tissue = cv2.imread(join(source_root, rf'tissue/label_origin/{code}.png'), 0)
        tps, counts = np.unique(label_tissue, return_counts=True)

        # print(tps, counts)
        for tp, count in zip(tps, counts):
            if tp == 255:
                tp = 0
            all_counts[tp] += count

    print(all_counts)


main()
