import cv2
import numpy as np
import torch.nn as nn
import torch
import gc
from torchvision import transforms
from skimage import morphology

from utils import gaussian_kernel


class HebingdouMaskDivider(object):
    def __init__(self, mask_model_path, device='cuda:0', batch: int = 8):
        super().__init__()
        # model_AFFormer = torch.jit.load(mask_model_path, 'cpu')
        # model_AFFormer = model_AFFormer.to(device)
        # model_AFFormer = model_AFFormer.eval()
        self.device = device
        self.mask_model = torch.jit.load(mask_model_path).to(device).eval()
        self.trnsfrms_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )
        self.ksize = 2048
        self.step = 1024
        self.batch = batch
        self.kernel = gaussian_kernel(size=self.ksize, steep=4)
        self.kernel = np.expand_dims(self.kernel, axis=2)

    def __call__(self, *args, **kwargs):
        result = self.forward(*args, **kwargs)
        # gc.collect()
        return result

    def forward(self, tmp_img, contour_area_threshold=10000):
        h, w = tmp_img.shape[:2]
        H = h + (-h % self.ksize)
        W = w + (-w % self.ksize)
        image = cv2.resize(tmp_img, (W, H))
        mask = np.zeros((H, W, 2), dtype=np.float32)
        patches = [(
            [x, y],
            image[y:y+self.ksize, x:x+self.ksize, :]
        ) for x in range(0, W - self.step, self.step) for y in range(0, H - self.step, self.step)]

        for group in range(0, len(patches), self.batch):
            patch = patches[group: min(group + self.batch, len(patches))]
            indexes, inputs = zip(*patch)

            inputs = np.stack([self.trnsfrms_val(ips) for ips in inputs])
            inputs = torch.tensor(inputs, requires_grad=False, device=self.device)
            results = self.mask_model(inputs)
            outputs = results.permute(0, 2, 3, 1).detach().cpu().numpy()

            for (x, y), pre in zip(indexes, outputs):
                mask[y:y+self.ksize, x:x+self.ksize, :] += pre * self.kernel

        tmppp_mask = (np.argmax(mask, axis=2).astype(dtype=np.uint8))
        # pre_mask_rever = morphology.remove_small_objects((tmppp_mask == 1), min_size=contour_area_threshold)
        # tmppp_mask1 = np.uint8(pre_mask_rever)
        #
        # pre_mask_rever = morphology.remove_small_objects((tmppp_mask1 == 0), min_size=contour_area_threshold)
        # tmppp_mask2 = np.uint8(pre_mask_rever)
        # tmppp_mask1[tmppp_mask2 == 0] = 1

        tmppp_mask1 = cv2.resize(tmppp_mask, (w, h))

        return tmppp_mask1
