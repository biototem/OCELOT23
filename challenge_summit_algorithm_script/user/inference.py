from __future__ import annotations

from typing import List, Tuple
import numpy as np

from main import detect


class Model:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(self, metadata, developing=False):
        self.metadata = metadata
        self.developing = developing

    def __call__(self, cell_patch: np.ndarray[np.uint8], tissue_patch: np.ndarray[np.uint8], pair_id: str) -> List[Tuple[int, int, int, float]]:

        # 试着加点断言，看看会不会报错，以确定提交数据是否具备一致性
        # tissue - cell 比例关系锁定为四倍
        assert round(self.metadata[pair_id]['tissue']['resized_mpp_x'] / self.metadata[pair_id]['cell']['resized_mpp_x']) == 4
        assert round(self.metadata[pair_id]['tissue']['resized_mpp_y'] / self.metadata[pair_id]['cell']['resized_mpp_y']) == 4
        # 横纵长锁定为 1024
        assert round(
            (self.metadata[pair_id]['tissue']['x_end'] - self.metadata[pair_id]['tissue']['x_start']) *
            self.metadata[pair_id]['mpp_x'] / self.metadata[pair_id]['tissue']['resized_mpp_x']
        ) == 1024
        assert round(
            (self.metadata[pair_id]['tissue']['y_end'] - self.metadata[pair_id]['tissue']['y_start']) *
            self.metadata[pair_id]['mpp_y'] / self.metadata[pair_id]['tissue']['resized_mpp_y']
        ) == 1024
        assert round(
            (self.metadata[pair_id]['cell']['x_end'] - self.metadata[pair_id]['cell']['x_start']) *
            self.metadata[pair_id]['mpp_x'] / self.metadata[pair_id]['cell']['resized_mpp_x']
        ) == 1024
        assert round(
            (self.metadata[pair_id]['cell']['y_end'] - self.metadata[pair_id]['cell']['y_start']) *
            self.metadata[pair_id]['mpp_y'] / self.metadata[pair_id]['cell']['resized_mpp_y']
        ) == 1024

        x = round(self.metadata[pair_id]['patch_x_offset'] * 1024 - 128)
        y = round(self.metadata[pair_id]['patch_y_offset'] * 1024 - 128)

        return detect(
            cell=cell_patch,
            tissue=tissue_patch,
            offset=(x, y),
            cache_code=self.developing and pair_id,
        )
