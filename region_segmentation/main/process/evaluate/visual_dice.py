import json
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from basic import join, config
from utils.table.my_table import Table


def visual(root: str, names: List[str] = None):
    CLASS_NAMES = config['evaluate.class_names']
    # 加载评分表
    score_table = Table.load(path=join(root, 'evaluate.txt'), dtype=object)
    if names:
        score_table = score_table[names, ]
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = score_table.keys[0]
    # 图片 ID
    data_keys = score_table.keys[1]

    # 逐模型
    idx = 0
    for model_name in model_names:
        # 逐类型
        for index, cls_name in enumerate(CLASS_NAMES):
            idx += 1
            plt.subplot(len(model_names), len(CLASS_NAMES), idx)
            plt.title(f'{model_name.replace("auto-save-epoch-", "")}-{cls_name}')
            for group, color in zip(config['evaluate.target_group'], ['g', 'y', 'r']):
                plt.grid(visible=True, axis='y')
                # 图片各自分类
                samples = [k for k in data_keys[:-len(config['evaluate.target_group'])] if key_set[k] == group]

                samples = score_table[model_name, samples, 'dice'].data.tolist()

                samples = [x[index] for x in samples]

                samples.sort(reverse=True)

                result = score_table[model_name, group, 'dice'][index]
                plt.plot(np.arange(len(samples)) / (len(samples) - 1), samples, color=color)

                # plt.scatter(np.arange(len(trains)) / (len(trains) - 1), trains, color='g', s=1)

                plt.hlines(y=result, xmin=0, xmax=1, colors=color, label=group)

            # plt.axis('equal')
            # plt.xticks([])
            plt.yticks([i / 40 for i in range(41)])
            # if index == 1:
            #     plt.yticks([i / 10 for i in range(11)])
            # else:
            #     plt.yticks([])

    plt.show()
