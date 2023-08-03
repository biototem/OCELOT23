from random import random

from albumentations import ImageOnlyTransform

from basic import join

import os
import numpy as np
import cv2
from tqdm import tqdm
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization


class Conver(ImageOnlyTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target_number = '/media/totem_data_backup/totem/tiger-training-data/wsirois/roi-level-annotations/tissue-cells/images/122S_[27872, 11652, 29086, 12830].png'
        target_tc = '/media/totem_data_backup/totem/tiger-training-data/wsirois/roi-level-annotations/tissue-cells/images/TC_S01_P000054_C0001_B101_[27671, 88267, 28946, 89493].png'
        self.target_number = cv2.cvtColor(cv2.imread(target_number), cv2.COLOR_BGR2RGB)
        self.target_tc = cv2.cvtColor(cv2.imread(target_tc), cv2.COLOR_BGR2RGB)

    def apply(self, img, **params):
        p = random()
        if p < 0.33:
            return self.convert(img, self.target_number)
        elif p < 0.66:
            return self.convert(img, self.target_tc)
        return img

    def convert(self, img, target):

        stain_unmixing_routine_params = {
            'stains': ['hematoxylin', 'eosin'],
            'stain_unmixing_method': 'macenko_pca',
        }
        tissue_rgb_normalized = deconvolution_based_normalization(
            img,
            im_target=target,
            stain_unmixing_routine_params=stain_unmixing_routine_params
        )

        source_change = cv2.cvtColor(tissue_rgb_normalized, cv2.COLOR_RGB2BGR)

        return source_change
