import json
from typing import List
import torch
import numpy as np

from basic import source, join, config
from dataset import Trainset
from utils import Timer
from utils.table.my_table import Table
from .predict import predict
from .draw import draw


def do_calculate(model_names: List[str], root: str, T: Timer = None, visual: bool = False):
    CLASS_NAMES = config['evaluate.class_names']
    # 数据集
    # 难样本特殊操作
    hard = [
        17, 199, 216, 239, 243, 250, 259, 280, 300, 310, 346, 351, 354, 368,
        26, 61, 74, 89, 135, 187, 203, 236, 237, 251, 270, 272, 281, 285, 294,
        295, 304, 317, 319, 332, 359, 362, 364, 373, 393, 399
    ]
    hard = [str(index).zfill(3) for index in hard]
    if T: T.track(' -> initializing dataset')
    target_datas = []
    for group in config['evaluate.target_group']:
        if group == 'hard':
            target_datas.extend(hard)
        else:
            target_datas.extend(source['group'][group])

    # target_datas = [d for d in target_datas if d not in hard]

    dataset_dict = {
        name: Trainset(
            names=[name],
            length=-1,
            cropper_type='simple',
            shuffle=False,
            transfer={'level': 'off'},
            return_label=False,
            return_subset=False,
            return_pos=True,
            cropper_rotate='list[0]',
            cropper_scaling='list[1]',
        ) for name in source['group']['available'] if name in target_datas
    }
    # 图集分包信息
    key_set = {
        **{key: 'hard' for key in hard},
        **{key: 'train' for key in source['group']['train'] if key not in hard},
        **{key: 'valid' for key in source['group']['valid'] if key not in hard},
    }
    # key_set.update(key_set_3)
    # 确认数据集的合法性
    for data_key in dataset_dict.keys():
        assert data_key in key_set, f'something goes wrong -> {data_key}'
    # 统计表单：
    # axis 1 -> 模型名称
    # axis 2 -> 图片名称
    # axis 3 -> 评估矩阵 -> 标签端
    # axis 4 -> 评估矩阵 -> 预测端
    # 元素值类型均为 int64，表示统计像素数
    table = Table(
        model_names, list(dataset_dict.keys()), CLASS_NAMES, CLASS_NAMES,
        dtype=np.int64, key_sep='#', k_v_sep=':')
    with torch.no_grad():
        for model_name in model_names:
            if T: T.track(f' -> initializing model -> {model_name}')
            m = torch.load(join(root, f'{model_name}.pth'), 'cpu')
            m.to(config['evaluate.device'])
            m = m.eval()
            for data_key, dataset in dataset_dict.items():
                if T: T.track(f' -> predicting {data_key}')
                # 获得预测结果
                argmaxed = predict(m, data_key, dataset)
                # 打印预测结果 -> 目录结合 keyset 确定
                if visual:
                    draw(argmaxed, data_key, dataset, root=join(root, model_name, key_set[data_key]))
                # 统计样本评估的原始结果
                table[model_name, data_key, :, :] = calculate_single(argmaxed, data_key, dataset, n=len(CLASS_NAMES))

    # 保存原始结果
    table.save(path=join(root, 'calculate.txt'))
    # 保存数据集分组信息
    key_set = json.dumps(key_set, indent=4)
    with open(join(root, 'key_set.txt'), 'w') as f:
        f.writelines(key_set)


def calculate_single(argmaxed: np.ndarray, data_key: str, dataset: Trainset, n: int) -> np.ndarray:
    l, u, r, d = dataset.loaders[data_key].box
    label = dataset.loaders[data_key].label({
        'x': (l + r) // 2,
        'y': (u + d) // 2,
        'w': (r - l),
        'h': (d - u),
        'degree': 0,
        'scaling': 1,
    })
    mask = dataset.loaders[data_key].mask({
        'x': (l + r) // 2,
        'y': (u + d) // 2,
        'w': (r - l),
        'h': (d - u),
        'degree': 0,
        'scaling': 1,
    })
    result = np.empty(shape=(n, n), dtype=int)
    for i in range(n):
        gt = (label == i) & mask.astype(bool)
        for j in range(n):
            pr = (argmaxed == j) & mask.astype(bool)
            # result[i, j] 表示 gt == i 且 pr == j 的像素的个数，因此：
            # result.sum(axis=1) 表示 gt 的像素数
            # result.sum(axis=0) 表示 pr 的像素数
            # result.diagonal() 表示 tp 的像素数
            result[i, j] = (gt & pr).sum()
    return result
