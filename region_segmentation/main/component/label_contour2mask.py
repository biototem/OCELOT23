from typing import Tuple, List
from scipy import signal
import cv2
import numpy as np

from utils import Shape


LABEL_DICT = {
    'STROMA': 1,
    'NORMAL': 2,
    'TUMOR': 3,
    'NECROSIS': 4,
    'VESSEL': 5,
    'OTHER': 6,
}


def labels2mask(size: Tuple[int, int], labels=List[Tuple[str, Shape]]):
    w, h = size
    temp = np.zeros(shape=(h, w), dtype=np.uint8)
    # 先画大轮廓后画小轮廓，以此支持“轮廓A在轮廓B中”的需求
    labels = sorted(labels, key=lambda pair: pair[1].area, reverse=True)
    for cls, label in labels:
        cls_value = LABEL_DICT[cls]
        for lb in label.sep_out():
            outer, inners = lb.sep_p()
            coords = [np.array(pts, dtype=int) for pts in (outer, *inners)]
            cv2.fillPoly(temp, coords, [cls_value])
            # cv2.drawContours(temp, coords, None, [cls_value])
            # PPlot().add(temp).show()
    return temp


def image2mask(image: np.ndarray):
    # 阈值蒙版
    # # tim1 -> 滤除纯白区域，要求至少在某一RGB上具有小于210的色值
    # tim1 = np.any(image <= 210, 2).astype(np.uint8)
    # # tim2 -> 滤除灰度区域，要求RGB的相对色差大于18
    # tim2 = (np.max(image, 2) - np.min(image, 2) > 18).astype(np.uint8)

    # 先缩放
    h, w, _ = image.shape
    temp = cv2.resize(image, (w // 4, h // 4))
    tim1 = np.any(temp <= 238, 2).astype(np.uint8)
    # tim2 = (np.max(temp, 2) - np.min(temp, 2) > 2).astype(np.uint8)
    m1 = np.max(temp, 2)
    m2 = np.min(temp, 2)
    m3 = np.mean(temp, 2)
    m4 = m2 / 255 * (255-m1)
    m5 = np.clip(m4, 1, 8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m6 = signal.convolve2d(m5, k, mode='same') / k.sum()
    # print(m1.mean(),m2.mean(),m4.mean(),m5.mean())
    tim2 = (m1 - m2 > m6).astype(np.uint8)
    tim = tim1 * tim2
    # 捕获边缘
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cv2.erode(tim, k, dst=tim, iterations=2)
    # cv2.dilate(tim, k, dst=tim, iterations=4)
    # cv2.erode(tim, k, dst=tim, iterations=2)
    cv2.dilate(tim, k, dst=tim, iterations=2)
    cv2.erode(tim, k, dst=tim, iterations=4)
    cv2.dilate(tim, k, dst=tim, iterations=2)

    tim = cv2.resize(tim, (w, h))
    # from utils import PPlot
    # # PPlot().add(temp, tim1, m1-m2, m5, tim2).show()
    # PPlot().add(temp, tim1, tim2, tim).show()
    return tim


def test1():
    from component import read_json
    import openslide
    from utils import PPlot

    slide = openslide.OpenSlide('/media/totem_disk/totem/Lung Area Segmentation/fold 2/TCGA-86-8279-01Z-00-DX1.fd12b60e-d181-454b-a655-298a973a849d.svs')
    results = read_json('/media/totem_disk/totem/jizheng/lung_area_seg/datasource/label/fold 2/TCGA-86-8279-01Z-00-DX1.fd12b60e-d181-454b-a655-298a973a849d.geojson')

    assert abs(1 - float(slide.properties['openslide.mpp-x']) / float(slide.properties['openslide.mpp-y'])) < 0.05, \
        '警告：横纵分辨率相差过大!'
    source_mpp = float(slide.properties['openslide.mpp-x'])
    # 取得期望缩放比（但并不是真正的缩放比）
    expect_zoom = 0.5 / source_mpp
    # 取得缩放层级(允差 0.05)
    level = slide.get_best_level_for_downsample(expect_zoom + 0.05)
    # 取得层级缩放比
    level_zoom = slide.level_downsamples[level]
    # 取得相对缩放比
    relative_zoom = expect_zoom / level_zoom
    # 开始拆包，生成子图块
    pplt = PPlot()
    for index, (sight, box, labels) in enumerate(results):
        # 边界向 level 取齐
        x, y = map(lambda u: round(u), sight[:2])
        w, h = map(lambda u: round(u / level_zoom), sight[2:])
        print(x, y, w, h)
        # 获得 sight 区域图
        image = slide.read_region(location=(x, y), level=level, size=(w, h))
        image = np.asarray(image)
        # 若相对缩放比小于允差，则无需执行 cv 缩放
        # 若相对缩放比大于允差，则需要执行 cv 缩放
        absolute_zoom = slide.level_downsamples[level]
        if relative_zoom - 1 > 0.05:
            absolute_zoom *= relative_zoom
            # 边界向期望取齐
            x, y, w, h = map(lambda w: round(w / relative_zoom), (x, y, w, h))
            image = cv2.resize(image, (w, h))
        # 获得计算后的 label 信息
        sight = (x, y, w, h)
        box = tuple(map(lambda w: round(w / absolute_zoom), box))
        labels = [(cls, label / absolute_zoom) for cls, label in labels]
        # 至此，label 和 image 信息都完整了， 将它们存放在 cache 目录下即可
        # 思路1: 沿用原 dataset， 先把 label 画出来放好
        label_mask = labels2mask(size=(w, h), labels=labels)
        pplt.add(label_mask / 7)
    pplt.show()


def test2():
    from component import read_json
    import openslide
    from utils import PPlot

    slide = openslide.OpenSlide(
        '/media/totem_disk/totem/Lung Area Segmentation/fold 1/157974_648337001.svs'
    )
    results = read_json(
        '/media/totem_disk/totem/jizheng/lung_area_seg/datasource/label/fold 1/157974_648337001.geojson'
    )

    # image_mask
    # level = slide.get_best_level_for_downsample(2.1)
    # img1 = np.asarray(slide.read_region(
    #     location=(40645, 40305),
    #     level=level,
    #     size=(2786, 2274)
    # ))
    # img2 = np.asarray(slide.read_region(
    #     location=(83457, 39792),
    #     level=level,
    #     size=(2468, 2311)
    # ))
    # image_mask1 = image2mask(img1)
    # image_mask2 = image2mask(img2)

    # label_mask
    assert abs(1 - float(slide.properties['openslide.mpp-x']) / float(slide.properties['openslide.mpp-y'])) < 0.05, \
        '警告：横纵分辨率相差过大!'
    source_mpp = float(slide.properties['openslide.mpp-x'])
    # 取得期望缩放比（但并不是真正的缩放比）
    expect_zoom = 0.5 / source_mpp
    # 取得缩放层级(允差 0.05)
    level = slide.get_best_level_for_downsample(expect_zoom + 0.05)
    # 取得层级缩放比
    level_zoom = slide.level_downsamples[level]
    # 取得相对缩放比
    relative_zoom = expect_zoom / level_zoom
    # 开始拆包，生成子图块
    pplt = PPlot()
    for index, (sight, box, labels) in enumerate(results):
        # 边界向 level 取齐
        x, y = map(lambda u: round(u), sight[:2])
        w, h = map(lambda u: round(u / level_zoom), sight[2:])
        print(x, y, w, h)
        # 获得 sight 区域图
        image = slide.read_region(location=(x, y), level=level, size=(w, h))
        image = np.asarray(image)[:, :, :3]
        # 若相对缩放比小于允差，则无需执行 cv 缩放
        # 若相对缩放比大于允差，则需要执行 cv 缩放
        absolute_zoom = slide.level_downsamples[level]
        # 获得计算后的 label 信息
        sight = (x, y, w, h)
        box = tuple(map(lambda w: round(w / absolute_zoom), box))
        labels = [(cls, label / absolute_zoom) for cls, label in labels]
        # 至此，label 和 image 信息都完整了， 将它们存放在 cache 目录下即可
        # 思路1: 沿用原 dataset， 先把 label 画出来放好
        if index == 0:
            label_mask1 = labels2mask(size=(w, h), labels=labels)
            img1 = image
            image_mask1 = image2mask(img1)
        if index == 1:
            label_mask2 = labels2mask(size=(w, h), labels=labels)
            img2 = image
            image_mask2 = image2mask(img2)

    mask1 = label_mask1.copy()
    mask1[(label_mask1 == 0) & (image_mask1 == 1)] = 1
    mask2 = label_mask2.copy()
    mask2[(label_mask2 == 0) & (image_mask2 == 1)] = 1

    PPlot().add(img1, img2, mask1, mask2, image_mask1, image_mask2, label_mask1, label_mask2).show()


if __name__ == '__main__':
    # test1()
    test2()
