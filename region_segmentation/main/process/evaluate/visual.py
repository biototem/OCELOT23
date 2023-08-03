import json
import numpy as np
import matplotlib.pyplot as plt

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


def do_visual(root: str):
    score_table = Table.load(path=join(root, 'evaluate.txt'), dtype=object)
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = score_table.keys[0]
    # 图片标签
    data_keys = score_table.keys[1]

    # 逐模型
    idx = 0
    for model_name in model_names:
        for index, cls_name in enumerate(CLASS_NAMES):
            idx += 1
            plt.subplot(len(model_names), len(CLASS_NAMES), idx)
            plt.title(f'{model_name}-{cls_name}')
            # All Divides
            trains = [k for k in data_keys[:-3] if key_set[k] == 'train']
            valids = [k for k in data_keys[:-3] if key_set[k] == 'valid']
            tests = [k for k in data_keys[:-3] if key_set[k] == 'test']

            # datas = score_table[model_name, data_keys[:-3], metric].data.tolist()
            trains = score_table[model_name, trains, 'dice'].data.tolist()
            valids = score_table[model_name, valids, 'dice'].data.tolist()
            tests = score_table[model_name, tests, 'dice'].data.tolist()

            trains = [x[index] for x in trains]
            valids = [x[index] for x in valids]
            tests = [x[index] for x in tests]

            trains.sort(reverse=True)
            valids.sort(reverse=True)
            tests.sort(reverse=True)

            train = score_table[model_name, 'train', 'dice'][index]
            valid = score_table[model_name, 'valid', 'dice'][index]
            test = score_table[model_name, 'test', 'dice'][index]

            print(len(trains), len(valids), len(tests))
            plt.plot(np.arange(len(trains)) / (len(trains) - 1), trains, color='g')
            plt.plot(np.arange(len(valids)) / (len(valids) - 1), valids, color='y')
            plt.plot(np.arange(len(tests)) / (len(tests) - 1), tests, color='r')

            # plt.scatter(np.arange(len(trains)) / (len(trains) - 1), trains, color='g', s=1)
            # plt.scatter(np.arange(len(valids)) / (len(valids) - 1), valids, color='y', s=1)
            # plt.scatter(np.arange(len(tests)) / (len(tests) - 1), tests, color='r', s=1)

            plt.hlines(y=train, xmin=0, xmax=1, colors='g', label='train')
            plt.hlines(y=valid, xmin=0, xmax=1, colors='y', label='valid')
            plt.hlines(y=test, xmin=0, xmax=1, colors='r', label='test')
    plt.show()


def do_visual2(root: str):
    # Load Origin Data
    # data_table = Table.load(path=join(root, 'calculate.txt'), dtype=np.int64)
    # Load Evaluate Data
    score_table = Table.load(path=join(root, 'evaluate.txt'), dtype=object)
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = score_table.keys[0]
    # 图片标签
    data_keys = score_table.keys[1]
    # 评价指标：
    # mdice: 除 label==0 之外的全类型平均 dice
    # dice: 除 label==0 之外的全类型 dice 列表
    # score: 以下两个评价指标间的 mdice
    # dice1: 浸润性肿瘤的 dice
    # dice26: 两种间质的混叠 dice
    metrics = score_table.keys[2]
    # Visualizing Metrics

    # 逐模型
    idx = 0
    plt.title('Green: train | Yellow: valid | Red: test')
    for model_name in model_names:
        for metric in metrics:
            if metric == 'dice':
                continue
            idx += 1
            plt.subplot(len(model_names), len(metrics) - 1, idx)
            plt.title(f'{model_name}-{metric}')
            # All Divides
            trains = [k for k in data_keys[:-3] if key_set[k] == 'train']
            valids = [k for k in data_keys[:-3] if key_set[k] == 'valid']
            tests = [k for k in data_keys[:-3] if key_set[k] == 'test']

            # datas = score_table[model_name, data_keys[:-3], metric].data.tolist()
            trains = score_table[model_name, trains, metric].data.tolist()
            valids = score_table[model_name, valids, metric].data.tolist()
            tests = score_table[model_name, tests, metric].data.tolist()

            trains.sort(reverse=True)
            valids.sort(reverse=True)
            tests.sort(reverse=True)

            train = score_table[model_name, 'train', metric]
            valid = score_table[model_name, 'valid', metric]
            test = score_table[model_name, 'test', metric]

            plt.plot(np.arange(len(trains)) / (len(trains) - 1), trains, color='g')
            plt.plot(np.arange(len(valids)) / (len(valids) - 1), valids, color='y')
            plt.plot(np.arange(len(tests)) / (len(tests) - 1), tests, color='r')

            # plt.scatter(np.arange(len(trains)) / (len(trains) - 1), trains, color='g', s=1)
            # plt.scatter(np.arange(len(valids)) / (len(valids) - 1), valids, color='y', s=1)
            # plt.scatter(np.arange(len(tests)) / (len(tests) - 1), tests, color='r', s=1)

            plt.hlines(y=train, xmin=0, xmax=1, colors='g', label='train')
            plt.hlines(y=valid, xmin=0, xmax=1, colors='y', label='valid')
            plt.hlines(y=test, xmin=0, xmax=1, colors='r', label='test')
    plt.show()


def do_visual1(root: str):
    # Load Origin Data
    # data_table = Table.load(path=join(root, 'calculate.txt'), dtype=np.int64)
    # Load Evaluate Data
    score_table = Table.load(path=join(root, 'evaluate.txt'), dtype=object)
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = score_table.keys[0]
    # 图片标签
    data_keys = score_table.keys[1]
    # 评价指标：
    # mdice: 除 label==0 之外的全类型平均 dice
    # dice: 除 label==0 之外的全类型 dice 列表
    # score: 以下两个评价指标间的 mdice
    # dice1: 浸润性肿瘤的 dice
    # dice26: 两种间质的混叠 dice
    metrics = score_table.keys[2]
    # Visualizing Metrics

    # 逐模型
    idx = 0
    plt.legend()
    for model_name in model_names:
        for metric in metrics:
            if metric == 'dice':
                continue
            idx += 1
            plt.subplot(len(model_names), len(metrics) - 1, idx)
            plt.title(f'{model_name}-{metric}')
            # All in Group
            datas = score_table[model_name, data_keys[:-3], metric].data.tolist()
            train = score_table[model_name, 'train', metric]
            valid = score_table[model_name, 'valid', metric]
            test = score_table[model_name, 'test', metric]
            plt.scatter(range(len(datas)), datas, color='c', s=1)
            plt.hlines(y=train, xmin=0, xmax=len(datas), colors='g', label='train')
            plt.hlines(y=valid, xmin=0, xmax=len(datas), colors='y', label='valid')
            plt.hlines(y=test, xmin=0, xmax=len(datas), colors='r', label='test')
            # plt.legend()
    plt.show()
