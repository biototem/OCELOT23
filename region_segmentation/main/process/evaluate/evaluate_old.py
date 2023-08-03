import json
from typing import Tuple

import numpy as np

from basic import join
from utils.table.my_table import Table

# 分类名称列
CLASS_NAMES = [
    'bg',  # 背景，值序：0，染色：浅灰
    'tm1',  # 浸润性肿瘤，值序：1，染色：亮红
    'st1',  # 肿瘤相关间质，值序：2，染色：亮黄
    'tm2',  # 原位肿瘤，值序：3，染色：暗红
    'nm',  # 健康腺体，值序：4，染色：深绿
    'nc',  # 非原位坏死，值序：5，染色：纯黑
    'st2',  # 炎症间质，值序：6，染色：橙黄
    'rs',  # 其它组织，值序：7，染色：纯白
]


def do_evaluate(root: str):
    # 加载统计信息
    data_table = Table.load(path=join(root, 'calculate.txt'), dtype=np.int64)
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = data_table.keys[0]
    # 图片标签
    data_keys = data_table.keys[1]
    # 评价指标：
    # mdice: 除 label==0 之外的全类型平均 dice
    # dice: 除 label==0 之外的全类型 dice 列表
    # score: 以下两个评价指标间的 mdice
    # dice1: 浸润性肿瘤的 dice
    # dice26: 两种间质的混叠 dice
    metrics = ['mdice', 'dice', 'score', 'dice1', 'dice26']
    # 以下开始进入评测流程
    score_table = Table(
        model_names, data_keys + ['train', 'valid', 'test'], metrics,
        k_v_sep=data_table.k_v_sep, key_sep=data_table.key_sep,
    )
    # 逐模型
    for model_name in model_names:
        # 分类统计汇总
        collect = {
            'train': {
                'dice': {'pr': np.zeros(8, int), 'gt': np.zeros(8, int), 'tp': np.zeros(8, int)},
                'dice26': {'pr': 0, 'gt': 0, 'tp': 0},
            },
            'valid': {
                'dice': {'pr': np.zeros(8, int), 'gt': np.zeros(8, int), 'tp': np.zeros(8, int)},
                'dice26': {'pr': 0, 'gt': 0, 'tp': 0},
            },
            'test': {
                'dice': {'pr': np.zeros(8, int), 'gt': np.zeros(8, int), 'tp': np.zeros(8, int)},
                'dice26': {'pr': 0, 'gt': 0, 'tp': 0},
            },
        }
        # 逐图片
        for data_key in data_keys:
            # 图片归类
            data_type = key_set[data_key]
            # 开始打分
            dice, dice26, dice_val, dice26_val = evaluate_dice(data_table[model_name, data_key])
            # 存储长程统计数据
            collect[data_type]['dice']['pr'] += dice_val['pr']
            collect[data_type]['dice']['gt'] += dice_val['gt']
            collect[data_type]['dice']['tp'] += dice_val['tp']
            collect[data_type]['dice26']['pr'] += dice26_val['pr']
            collect[data_type]['dice26']['gt'] += dice26_val['gt']
            collect[data_type]['dice26']['tp'] += dice26_val['tp']
            # 存储单图评分
            score_table[model_name, data_key, 'mdice'] = sum(dice) / len(dice)
            score_table[model_name, data_key, 'dice'] = dice
            score_table[model_name, data_key, 'score'] = (dice[1] + dice26) / 2
            score_table[model_name, data_key, 'dice1'] = dice[1]
            score_table[model_name, data_key, 'dice26'] = dice26

        # 长程统计 - train
        # 长程统计数据
        dice_val = collect['train']['dice']
        pr, gt, tp = dice_val['pr'], dice_val['gt'], dice_val['tp']
        dice26_val = collect['train']['dice26']
        pr26, gt26, tp26 = dice26_val['pr'], dice26_val['gt'], dice26_val['tp']
        # 长程统计评分
        dice = 2 * tp / (pr + gt + 1e-17)
        dice = dice.tolist()
        dice26 = 2 * tp26 / (pr26 + gt26 + 1e-17)
        score_table[model_name, 'train', 'mdice'] = sum(dice) / len(dice)
        score_table[model_name, 'train', 'dice'] = dice
        score_table[model_name, 'train', 'score'] = (dice[1] + dice26) / 2
        score_table[model_name, 'train', 'dice1'] = dice[1]
        score_table[model_name, 'train', 'dice26'] = dice26

        # 长程统计 - valid
        # 长程统计数据
        dice_val = collect['valid']['dice']
        pr, gt, tp = dice_val['pr'], dice_val['gt'], dice_val['tp']
        dice26_val = collect['valid']['dice26']
        pr26, gt26, tp26 = dice26_val['pr'], dice26_val['gt'], dice26_val['tp']
        # 长程统计评分
        dice = 2 * tp / (pr + gt + 1e-17)
        dice = dice.tolist()
        dice26 = 2 * tp26 / (pr26 + gt26 + 1e-17)
        score_table[model_name, 'valid', 'mdice'] = sum(dice) / len(dice)
        score_table[model_name, 'valid', 'dice'] = dice
        score_table[model_name, 'valid', 'score'] = (dice[1] + dice26) / 2
        score_table[model_name, 'valid', 'dice1'] = dice[1]
        score_table[model_name, 'valid', 'dice26'] = dice26

        # 长程统计 - test
        # 长程统计数据
        dice_val = collect['test']['dice']
        pr, gt, tp = dice_val['pr'], dice_val['gt'], dice_val['tp']
        dice26_val = collect['test']['dice26']
        pr26, gt26, tp26 = dice26_val['pr'], dice26_val['gt'], dice26_val['tp']
        # 长程统计评分
        dice = 2 * tp / (pr + gt + 1e-17)
        dice = dice.tolist()
        dice26 = 2 * tp26 / (pr26 + gt26 + 1e-17)
        score_table[model_name, 'test', 'mdice'] = sum(dice) / len(dice)
        score_table[model_name, 'test', 'dice'] = dice
        score_table[model_name, 'test', 'score'] = (dice[1] + dice26) / 2
        score_table[model_name, 'test', 'dice1'] = dice[1]
        score_table[model_name, 'test', 'dice26'] = dice26

    # 保存评估结果
    score_table.save(path=join(root, 'evaluate.txt'))


def evaluate_dice(score_table: Table) -> Tuple[list, float, dict, dict]:
    # 直接根据 CLASS_NAMES 在矩阵上进行数据处理
    data = score_table.data
    # 根据业务需求， gt == 0 的标签不参与评测，因而 gt == 0 的 pr 应当从评分中剔除
    pr = [int(data[1:, i].sum()) for i in range(8)]
    gt = [int(data[i, :].sum()) for i in range(8)]
    tp = [int(data[i, i]) for i in range(8)]
    pr26 = int(pr[2] + pr[6])
    gt26 = int(gt[2] + gt[6])
    tp26 = int(data[2, 2] + data[6, 6] + data[2, 6] + data[6, 2])
    dice = [2 * t / (p + g + 1e-17) for t, p, g in zip(tp, pr, gt)]
    dice26 = 2 * tp26 / (pr26 + gt26 + 1e-17)
    return dice, dice26, {'pr': pr, 'gt': gt, 'tp': tp}, {'pr': pr26, 'gt': gt26, 'tp': tp26}
