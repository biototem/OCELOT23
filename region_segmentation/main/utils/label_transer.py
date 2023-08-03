from typing import Dict
import cv2
import numpy as np
import json

from .shape import ComplexPolygon, ComplexMultiPolygon


def geojson2label(mask: np.ndarray, geojson_path: str, TYPE_MAPPER: Dict[str, int]) -> np.ndarray:

    h, w = mask.shape

    with open(geojson_path) as f:
        geojson = json.load(f)

    # 里面可能是一个标注，则最外层为 dict，也可能时一组标注，则最外层为 list
    if isinstance(geojson, dict):
        geojson = [geojson]
    shapes = []
    names = []
    colors = []
    for i, lb in enumerate(geojson):
        if lb['geometry']['type'].upper() == 'POLYGON':
            outer, *inners = lb['geometry']['coordinates']
            polygon = ComplexPolygon(outer, *inners)
            shapes.append(polygon)
        elif lb['geometry']['type'].upper() == 'LINESTRING':
            outer = lb['geometry']['coordinates']
            polygon = ComplexPolygon(outer, )
            shapes.append(polygon)
        else:
            polygons = []
            for coords in lb['geometry']['coordinates']:
                outer, *inners = coords
                polygon = ComplexPolygon(outer, *inners)
                polygons.append(polygon)
            multi_polygon = ComplexMultiPolygon(singles=polygons)
            shapes.append(multi_polygon)
        colors.append(lb['properties']['classification']['color'])
        names.append(lb['properties']['classification']['name'])

    # temp = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    label = np.zeros(shape=(h, w), dtype=np.uint8) + 2

    for shape, color, name in zip(shapes, colors, names):
        tp = TYPE_MAPPER[name.lower()]
        singles = shape.sep_out()
        for single in singles:
            outer, inners = single.sep_p()
            coords = [np.array(outer, dtype=int)] + [np.array(inner, dtype=int) for inner in inners]
            # cv2.fillPoly(temp, coords, color)
            cv2.fillPoly(label, coords, tp)
            # cv2.drawContours(temp, coords, None, color, thickness=1)

    label[~mask.astype(bool)] = 0
    return label
