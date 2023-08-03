import json
from typing import Tuple, Dict
import numpy as np

from basic import join, config
from utils.table.my_table import Table


def do_evaluate(root: str):
    CLASS_NAMES = config['evaluate.class_names']
    # 加载统计信息
    data_table = Table.load(path=join(root, 'calculate.txt'), dtype=np.int64)
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = data_table.keys[0]
    # 图片标签
    data_keys = [key for key in data_table.keys[1] if key != '122S_[41083, 29393, 42269, 30544]']
    # 评价指标：
    # mdice: 除 label==0 之外的全类型平均 dice
    # dice: 除 label==0 之外的全类型 dice 列表
    # score: 以下两个评价指标间的 mdice
    # dice1: 浸润性肿瘤的 dice
    # dice26: 两种间质的混叠 dice
    metrics = ['wdice', 'weight', 'dice', 'dice_mix']
    # 以下开始进入评测流程
    score_table = Table(
        model_names, data_keys + config['evaluate.target_group'], metrics,
        k_v_sep=data_table.k_v_sep, key_sep=data_table.key_sep,
    )
    # 逐模型
    for model_name in model_names:
        # 计算单图评分
        for data_key in data_keys:
            # 开始打分
            result = evaluate_dice(matrix_table=data_table[model_name, data_key], class_num=len(CLASS_NAMES))
            # 存储单图评分 -> 考虑剔除死线数据
            # 剔除方式有两种：
            # mdice -> 只剔除 gt == 0 的死线
            # wdice -> 按 gt 像素数赋权来决定均占比
            score_table[model_name, data_key, 'wdice'] = sum([d * g for d, g in zip(result['dice'], result['ws'])]) / sum(result['ws'])
            score_table[model_name, data_key, 'weight'] = result['ws']
            # print(ws)
            score_table[model_name, data_key, 'dice'] = result['dice']
            score_table[model_name, data_key, 'dice_mix'] = result['dice_mix']

        # 计算组评分
        for group in config['evaluate.target_group']:
            # 获取分组信息
            group_keys = list(filter(lambda key: key_set[key] == group, data_keys))
            assert len(group_keys) >= 2, f'分组数量少于 2 是不可能的，请检查这里的 bug：{group_keys}'
            # 获得统计矩阵 -> 直接在 data_key 这个维度上求和即可
            matrix = data_table[model_name, group_keys].data.sum(axis=0)
            # 将统计矩阵还原为统计表
            matrix_table = Table(
                CLASS_NAMES,
                CLASS_NAMES,
                data=matrix,
            )
            result = evaluate_dice(matrix_table=matrix_table, class_num=len(CLASS_NAMES))
            # 存储单图评分 -> 考虑剔除死线数据
            # 剔除方式有两种：
            # mdice -> 只剔除 gt == 0 的死线
            # wdice -> 按 gt 像素数赋权来决定均占比
            score_table[model_name, group, 'wdice'] = sum([d * g for d, g in zip(result['dice'], result['ws'])]) / sum(result['ws'])
            score_table[model_name, group, 'weight'] = result['ws']
            # print(ws)
            score_table[model_name, group, 'dice'] = result['dice']
            score_table[model_name, group, 'dice_mix'] = result['dice_mix']

    # print(score_table)
    # for pk in data_keys + ['train', 'valid', 'test']:
    #     wdice = score_table['auto-save-epoch-16', pk, 'wdice']
    #     dice134 = score_table['auto-save-epoch-16', pk, 'dice134']
    #     if wdice < 0.8 and dice134 < 0.8:
    #         print(f'{pk}: {wdice}, {dice134}')

    # 保存评估结果
    score_table.save(path=join(root, 'evaluate.txt'))


def evaluate_dice(matrix_table: Table, class_num: int) -> Dict[str, object]:
    # 评估矩阵的 numpy 数组
    matrix = matrix_table.data
    #
    pr = [int(matrix[:, i].sum()) for i in range(class_num)]
    gt = [int(matrix[i, :].sum()) for i in range(class_num)]
    tp = [int(matrix[i, i]) for i in range(class_num)]
    dice = [2 * t / (p + g + 1e-17) for t, p, g in zip(tp, pr, gt)]

    # dice_mix 是不去分具体类型的综合 dice 计算
    pr_mix = int(sum(pr[:]))
    gt_mix = int(sum(gt[:]))
    tp_mix = int(sum(tp[:]))
    dice_mix = 2 * tp_mix / (pr_mix + gt_mix + 1e-17)

    # ws 表示当前图片的标签有效性权重，其中 -0.003465735 = np.log(0.5) / 200
    # 即：gt 像素数每增加 200，其”无效性“减半
    # 同时， label == 0 始终无效
    # ws = [1 - np.exp(-0.003465735 * g) for g in gt]
    ws = [1 - np.exp(-6.931471805599453e-05 * g) for g in gt]

    return {
        'ws': ws,
        'dice': dice,
        'dice_mix': dice_mix,
    }
