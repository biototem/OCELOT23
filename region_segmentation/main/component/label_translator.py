import cv2
import numpy as np
import os
import json

__all__ = ['read_json']

from utils import ComplexPolygon, ComplexMultiPolygon


def read_json(json_path: str):
    """
    :param json_path: 待解析的 json 文件路径
    文件格式说明：
    [{
        "type": "Feature",                  # 固定值，无效
        "geometry": {
            "type": str,                    # MultiPolygon || Polygon
            "coordinates": [],              # 轮廓点坐标，形状见下：
            # --------------------------------
            # Polygon:
            # [outer, inner1, inner2, ...]
            # MultiPolygon:
            # [
            #   [outer, inner1, inner2, ...],
            #   ...
            # ]
            # outer ~ inner ~ [(x, y), ...]
            # --------------------------------
        },
        "properties": {
            "object_type": “annotation”,        # 固定值，无效
            "classification": {
                "name": str,                    # 这个是我们的分类名称
                "colorRGB": int,                # 全是负数，尚未发现取色规则
            },
            "isLocked": bool,                   # 无效项
        },
    }]
    :return: [(sight, box, labels)]
    结果说明：
    本函数的结果只对 json 文件负责，不涉及缩放，
    """
    with open(json_path) as f:
        items = json.load(f)

    # 基于上述探索，我们只取用三个字段：
    # 类别 -> properties.classification.name
    # 轮廓类型 -> geometry.type
    # 轮廓点 -> geometry.coordinates
    boxes = []
    labels = []
    for item in items:
        label_type = item['properties']['classification']['name']
        shape_type = item['geometry']['type']
        coords = item['geometry']['coordinates']
        # 轮廓情况：
        #   存在 Multi 轮廓，但大部分都是一个主轮廓 + 一堆噪点
        #   存在 Complex 轮廓，但大部分也都是由噪声引起的
        #   轮廓之间存在包含现象
        # 业务类型：
        #   ROI: box框， 一张图中可能有一道两个 box
        #   Tumor, Stroma, Necrosis, Normal, Vessel, other: 目标分类
        #   Immune Cells: 视同 Stroma
        #   未标记区域： 视同 Stroma
        # 处置逻辑：
        #   首先将轮廓分为三大类别，box，labels 和 stromas
        #   box 规定了训练集的采样区域
        #   labels 做差集运算，直至无重叠区域为止
        #   然后所有的 labels 对 stromas 做差
        #   最后，stromas 合并 box 内其余区域， 并对组织区域蒙版做差
        #   最终类型数如下：
        #   0: 白背景 | 噪声 | 非组织轮廓
        #   1: Stroma
        #   2: Normal
        #   3: Tumor
        #   4: Necrosis
        #   5: Vessel
        #   6: other
        if shape_type.upper() == 'MULTIPOLYGON':
            singles = [ComplexPolygon(outer, *inners) for outer, *inners in coords]
            shape = ComplexMultiPolygon(singles=singles)
        else:
            shape = ComplexPolygon(coords[0], *coords[1:])

        # 由于 0级图 实在太大，预处理阶段全部采样至 16倍
        # shape /= 16
        if label_type.upper() == 'ROI':
            boxes.append(shape)
        else:
            if label_type.upper() == 'IMMUNE CELLS':
                label_type = 'STROMA'
            labels.append((label_type.upper(), shape))

    # 对 box 向心融合，获得 sight
    sights = []
    sight_labels = []
    for box in boxes:
        sight_label = []
        box = box.copy()
        for cls, label in labels:
            if not box.is_joint(label): continue
            sight_label.append((cls, label))
            box <<= label
        sights.append(box)
        sight_labels.append(sight_label)

    # 获得 sights 坐标
    sights = [sight.bounds for sight in sights]
    # 转换为 l, u, w, h
    sights = [(l, u, r-l, d-u) for l, u, r, d in sights]

    # 获得 box 坐标
    boxes = [box.bounds for box in boxes]
    # 向 sight 对齐
    boxes = [(l-x, u-y, r-x, d-y) for (l, u, r, d), (x, y, _, _) in zip(boxes, sights)]
    # label 向 sight 对齐
    for (x, y, _, _), sight_label in zip(sights, sight_labels):
        for _, label in sight_label:
            label -= (x, y)
    return list(zip(sights, boxes, sight_labels))

    # label_values = {
    #     'STROMA': 1,
    #     'NORMAL': 2,
    #     'TUMOR': 3,
    #     'NECROSIS': 4,
    #     'VESSEL': 5,
    #     'OTHER': 6,
    # }
    #
    # masks = [np.zeros(shape=(w, h), dtype=int) for (_, _, w, h) in sights]
    # for mask in masks:
    #     print(mask.shape, mask.size)
    # for cls, label in labels:
    #     print(label.area / 5703548)
    # for mask, sight_label, sight in zip(masks, sight_labels, sights):
    #     # 先画大轮廓，再画小轮廓，这样就可以解决轮廓间包含关系了
    #     sight_label.sort(key=lambda pair: pair[1].area, reverse=True)
    #     for cls, label in sight_label:
    #         for lb in label.sep_out():
    #             outer, inners = lb.sep_p()
    #             coords = [np.array(inner, dtype=int) for inner in (outer, *inners)]
    #             cv2.fillPoly(mask, coords, label_values[cls])
    #
    # return list(zip(sights, boxes, masks))


def test():
    # result = read_json('/media/totem_disk/totem/jizheng/lung_area_seg/datasource/label/fold 2/TCGA-86-8279-01Z-00-DX1.fd12b60e-d181-454b-a655-298a973a849d.geojson')

    root = '/media/totem_disk/totem/jizheng/lung_area_seg/datasource'
    areas = {
        'STROMA': 0,
        'NORMAL': 0,
        'TUMOR': 0,
        'NECROSIS': 0,
        'VESSEL': 0,
        'OTHER': 0,
    }
    box_areas = 0
    sight_areas = 0
    for r, ds, fs in os.walk(root):
        for f in fs:
            name, ext = os.path.splitext(f)
            if ext != '.geojson': continue
            result = read_json(os.path.join(r, f))
            print(f'{name}')
            for sight, box, labels in result:
                print('\t', sight, box)
                for cls, label in labels:
                    x = '%s:%.2f' % (cls, label.area/sight[2]/sight[3])
                    print('\t\t', x)
                    areas[cls] += label.area
                box_areas += (box[2]-box[0]) * (box[3]-box[1])
                sight_areas += sight[2] * sight[3]
    print(areas)
    print(box_areas)
    print(sight_areas)


if __name__ == '__main__':
    test()

'''
    for points, color in zip(single_list, color_list):
        if color == '16711680':
            dr.polygon(points, fill=1, outline=101)
        elif color == "65280":
            dr.polygon(points, fill=2, outline=102)
        elif color == '255':
            dr.polygon(points, fill=3, outline=103)
        elif color == '65535':
            dr.polygon(points, fill=4, outline=104)
        elif color == '32255':
            dr.polygon(points, fill=5, outline=105)
        elif color == '8224125':
            dr.polygon(points, fill=6, outline=106)
        elif color == '16777085':
            dr.polygon(points, fill=7, outline=107)
'''
