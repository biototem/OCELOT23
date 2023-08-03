00 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 22:52:48 2022
@author: ys2af
"""

from skimage import measure, color
from skimage.measure import label, regionprops
import numpy as np
from copy import deepcopy
from PIL import Image
import cv2
import openslide
from openslide import *
import scipy.signal
from scipy.signal import argrelextrema
from skimage.filters.rank import entropy
from skimage.morphology import disk


# function to apply the entropy mask/filter
def filter_entropy_image(image, filter, disk_radius: int = 3):
    eimage = entropy(image, disk(disk_radius))
    new_picture = np.ndarray(shape=eimage.shape)  # [[False] * image.shape[1]] * image.shape[0]
    for rn, row in enumerate(eimage):
        for pn, pixel in enumerate(row):
            if pixel < filter:
                new_picture[rn, pn] = True
            else:
                new_picture[rn, pn] = False
    return new_picture.astype('b')


# main program to apply to a given image
def main():
    input_path = r'C:\Users\jizhe\Desktop\EntropyMasker-master\images\ERA_CVD_Logo_CMYK.png'
    output_path = r'C:\Users\jizhe\Desktop\EntropyMasker-master\output\ERA_CVD_Logo_CMYK.png'
    print(f"Processing image [{input_path}].")
    ORO = cv2.imread(input_path, 0)
    source = deepcopy(ORO)
    ent = entropy(source, disk(5))
    hist = list(np.histogram(ent, 30))
    minindex = list(argrelextrema(hist[0], np.less))

    thresh_localminimal = 0
    for i in range(len(minindex[0])):
        temp_thresh = hist[1][minindex[0][i]]
        if 1 < temp_thresh < 4:
            thresh_localminimal = temp_thresh

    thresh1 = (255 * filter_entropy_image(ORO, thresh_localminimal)).astype('uint8')
    mask_255 = cv2.bitwise_not(deepcopy(thresh1))

    redolbl = measure.label(np.array(mask_255), connectivity=2)
    redprops = regionprops(redolbl)

    redAreaData = []
    for r in range(len(redprops)):
        redAreaData.append(redprops[r].area)

    max_value = max(redAreaData)
    max_index = redAreaData.index(max_value)

    max_mask = (redolbl == (max_index + 1)) * 255
    print(f"Writing entropy-based masked image at [{output_path}].")

    cv2.imwrite(output_path, max_mask)


# start the main program
if __name__ == "__main__":
    main()

