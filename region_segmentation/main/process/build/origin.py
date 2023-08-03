import json
import os
import random

import imagesize

from basic import join


def build_origin() -> dict:

    data = {
        code: {
            'image': f'/media/predator/totem/jizheng/ocelot2023/tissue/image/{code}.jpg',
            'label': f'/media/predator/totem/jizheng/ocelot2023/tissue/label/{code}.png',
            'mask': f'/media/predator/totem/jizheng/ocelot2023/tissue/mask/{code}.png',
        } for code in map(lambda index: str(index).zfill(3), range(1, 401))
    }

    for code, info in data.items():
        assert os.path.exists(info['image']), f'{code} image not exists'
        assert os.path.exists(info['label']), f'{code} label not exists'
        assert os.path.exists(info['mask']), f'{code} mask not exists'

    return data
