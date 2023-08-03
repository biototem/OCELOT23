import json
import matplotlib.pyplot as plt

from basic import join, config
from utils.table.my_table import Table


def visual(root: str):
    CLASS_NAMES = config['evaluate.class_names']
    # 加载评分表
    score_table = Table.load(path=join(root, 'evaluate.txt'), dtype=object)
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = score_table.keys[0]
    # 图片 ID
    data_keys = score_table.keys[1]
    # 评价 ID
    metric_keys = score_table.keys[2]

    idx = 0
    # plt.legend()
    # 逐评价指标
    for metric in metric_keys:
        if metric not in [
            'wdice',
            'dice_mix'
        ]: continue
        idx += 1
        plt.subplot(2, 1, idx)
        # 逐分组
        for group, color in zip(config['evaluate.target_group'], ['g', 'y', 'r']):
            plt.grid(visible=True, axis='y')
            ys = score_table[model_names, group, metric].data.tolist()
            xs = [i + 1 for i in range(len(model_names))]
            plt.title(f'{metric}')
            plt.plot(xs, ys, color=color, label=group)
        plt.yticks([i / 20 for i in range(21)])
    plt.show()

    idx = 0
    plt.legend()
    # # 逐评价指标
    for j, c in enumerate(CLASS_NAMES):
        idx += 1
        plt.subplot(2, 1, idx)
        # 逐分组
        for group, color in zip(config['evaluate.target_group'], ['g', 'y', 'r']):
            plt.grid(visible=True, axis='y')
            ys = score_table[model_names, group, 'dice'].data.tolist()
            ys = [y[j] for y in ys]
            xs = [i + 1 for i in range(len(model_names))]
            plt.title(f'{c}')
            plt.plot(xs, ys, color=color, label=group)
        plt.yticks([i / 40 for i in range(41)])
    plt.show()
