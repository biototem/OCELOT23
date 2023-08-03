import json
import numpy as np
import json
from typing import List, Tuple

import cv2
import numpy as np

from basic import join


def create_pix_counts():

    with open('/media/predator/totem/jizheng/ocelot2023/metadata.json', 'r+') as f:
        meta = json.load(f)

    source_root = r'/media/predator/totem/jizheng/ocelot2023/'

    all_counts = {}
    for index in range(1, 401):
        if index == 152:
            continue
        code = str(index).zfill(3)

        print(join(source_root, rf'tissue/image/{code}.jpg'))

        # 分割图
        label_tissue = cv2.imread(join(source_root, rf'tissue/label_origin/{code}.png'), 0)
        tps, count = np.unique(label_tissue, return_counts=True)

        # print(tps, counts)
        counts = [0, 0]
        for tp, c in zip(tps, count):
            if tp == 255:
                continue
            counts[tp - 1] = int(c)
        all_counts[code] = {
            'counts': counts,
            'organ': meta['sample_pairs'][code]['organ'],
        }

    with open('/media/totem_disk/totem/jizheng/ocelot2023/resource/manual/pix_count.json', 'w+') as f:
        json.dump(all_counts, f, indent=4)


def watch_data():

    with open('/media/totem_disk/totem/jizheng/ocelot2023/resource/manual/pix_count.json', 'r+') as f:
        all_counts = json.load(f)

    statistics = {}
    for code, info in all_counts.items():
        if info['organ'] not in statistics:
            statistics[info['organ']] = {
                'count': 0,
                'bg_pix': 0,
                'ca_pix': 0,
            }
        statistics[info['organ']]['count'] += 1
        statistics[info['organ']]['bg_pix'] += info['counts'][0]
        statistics[info['organ']]['ca_pix'] += info['counts'][1]
    print(json.dumps(statistics, indent=4))


def main():

    with open('/media/totem_disk/totem/jizheng/ocelot2023/resource/manual/pix_count.json', 'r+') as f:
        all_counts = json.load(f)

    statistics = {}
    for code, info in all_counts.items():
        if info['organ'] not in statistics:
            statistics[info['organ']] = []
        statistics[info['organ']].append((code, info['counts'][0], info['counts'][1]))

    # print(json.dumps(statistics, indent=4))

    all_train = []
    all_valid = []
    print('启动随机数据均衡分组算法')
    print('-------------------------------------------------------------------------------')
    for organ, elements in statistics.items():
        train, valid = group(elements)
        bg_all = sum(bg for code, bg, ca in elements)
        ca_all = sum(ca for code, bg, ca in elements)
        bg_train = sum(bg for code, bg, ca in train)
        ca_train = sum(ca for code, bg, ca in train)
        bg_valid = sum(bg for code, bg, ca in valid)
        ca_valid = sum(ca for code, bg, ca in valid)
        all_train += train
        all_valid += valid

        print(f'器官 {organ} 分类结果展示：')
        print(f'#\t -> 基准比例 [{round(bg_all / (bg_all + ca_all), 3)}, {round(ca_all / (bg_all + ca_all), 3)}]')
        print(f'#\t -> 训练比例 [{round(bg_train / (bg_train + ca_train), 3)}, {round(ca_train / (bg_train + ca_train), 3)}]')
        print(f'#\t -> 验证比例 [{round(bg_valid / (bg_valid + ca_valid), 3)}, {round(ca_valid / (bg_valid + ca_valid), 3)}]')
        print(f'#\t -> BG 占比 [{round(bg_train / bg_all, 3)} : {round(bg_valid / bg_all, 3)}]')
        print(f'#\t -> CA 占比 [{round(ca_train / ca_all, 3)} : {round(ca_valid / ca_all, 3)}]')
        print(f'#\t -> train 名单：{[code for code, _, _ in train]}')
        print(f'#\t -> valid 名单：{[code for code, _, _ in valid]}')
    print('-------------------------------------------------------------------------------')
    print(f'最终汇总 train 名单：{[code for code, _, _ in all_train]}')
    print(f'最终汇总 valid 名单：{[code for code, _, _ in all_valid]}')


def group(elements: List[Tuple[str, int, int]]) -> Tuple[Tuple[str, int, int], Tuple[str, int, int]]:
    # 采用随机法分组，容限依旧是 5%
    # 首先统计数据集比例
    bg_all = sum(bg for code, bg, ca in elements)
    ca_all = sum(ca for code, bg, ca in elements)

    # 目标数据集像素占比
    bg_rate_all = bg_all / (bg_all + ca_all)
    ca_rate_all = ca_all / (bg_all + ca_all)

    # 目标数据集像素数
    target_valid_pix = int((bg_all + ca_all) * 0.2)

    # 最优界
    best_z = 1
    best_train = None
    best_valid = None

    for epoch in range(5000):
        # print('epoch', epoch)
        np.random.shuffle(elements)
        # 依据目标像素数尝试分组
        valid_pix = 0
        i = 0
        while target_valid_pix > valid_pix:
            valid_pix += elements[i][1] + elements[i][2]
            i += 1
        # 统计分组结果
        train = elements[i+1:]
        valid = elements[:i+1]
        bg_train = sum(bg for code, bg, ca in train)
        ca_train = sum(ca for code, bg, ca in train)
        bg_valid = sum(bg for code, bg, ca in valid)
        ca_valid = sum(ca for code, bg, ca in valid)

        bg_rate_train = bg_train / (bg_train + ca_train)
        ca_rate_train = ca_train / (bg_train + ca_train)
        bg_rate_valid = bg_valid / (bg_valid + ca_valid)
        ca_rate_valid = ca_valid / (bg_valid + ca_valid)

        z = max([
            abs(bg_rate_train / bg_rate_all - 1),
            abs(ca_rate_train / ca_rate_all - 1),
            abs(bg_rate_valid / bg_rate_all - 1),
            abs(ca_rate_valid / ca_rate_all - 1),
        ])

        if z <= best_z:
            best_z = z
            best_train = train
            best_valid = valid

        # if z < 0.05:
        #     print('\t\t!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return best_train, best_valid


main()
