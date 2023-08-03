import numpy as np


def build_group() -> dict:

    codes = list(map(lambda index: str(index).zfill(3), range(1, 401)))

    # 152 这张图片没有有效组织轮廓，全部像素均为 unknown

    available = [c for c in codes if c != '152']

    np.random.shuffle(codes)
    return {
        'all': codes,
        'available': available,
        'blocked': ['152'],
        'train': available[:360],
        'valid': codes[360:],
        'dev_test': codes[355:365],
    }
