import numpy as np
import torch.nn as nn
import torch
import gc
from torchvision import transforms
from skimage import morphology


class HebingdouMaskDivider(object):
    def __init__(self, mask_model_path, device='cuda: 0'):
        super().__init__()
        # model_AFFormer = torch.jit.load(mask_model_path, 'cpu')
        # model_AFFormer = model_AFFormer.to(device)
        # model_AFFormer = model_AFFormer.eval()
        self.device = device
        self.mask_model = torch.jit.load(mask_model_path).to(device).eval()
        for param in self.mask_model.parameters():
            param.requires_grad = False
        self.trnsfrms_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )
        self.trnsfrms_val.requires_grad = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, tmp_img, contour_area_threshold=10000):
        with torch.no_grad():
            mask = np.zeros((tmp_img.shape[0], tmp_img.shape[1], 2), dtype=np.float32)
            step = 0.5
            svs_H, svs_W = tmp_img.shape[0], tmp_img.shape[1]
            if min(tmp_img.shape[0], tmp_img.shape[1]) > 2048:
                pred_size = (2048, 2048)
                newimg_list = []
                newimg_index_list = []
                for h in range(0, svs_H, round(pred_size[0] * step)):
                    for w in range(0, svs_W, round(pred_size[1] * step)):
                        title = tmp_img[h:h + pred_size[0], w:w + pred_size[0]]
                        img_index = [h, h + pred_size[0], w, w + pred_size[0]]
                        mask_index = [0, pred_size[0], 0, pred_size[1]]
                        if (title.shape[0] != pred_size[0]) | (title.shape[1] != pred_size[1]):
                            title_tmp = np.zeros((pred_size[0], pred_size[1], 3), dtype=np.uint8)
                            title_tmp[:title.shape[0], :title.shape[1]] = title
                            mask_index = [0, title.shape[0], 0, title.shape[1]]
                            title = title_tmp
                        newimg_index_list.append([img_index, mask_index])
                        newimg_list.append(self.trnsfrms_val(title))
                outpur_mask2_list = []
                batch_size_88 = 8
                for i77 in range(0, len(newimg_list), batch_size_88):
                    t_imt = newimg_list[i77:(i77 + batch_size_88)]
                    outpur_mask0 = self.mask_model(torch.stack(t_imt).to(self.device))
                    outpur_mask00 = outpur_mask0.detach().permute(0, 2, 3, 1).cpu()
                    outpur_mask2_list.append(outpur_mask00)
                # outpur_mask1 = torch.cat(outpur_mask2_list).numpy()
                outpur_mask1 = torch.concat(outpur_mask2_list).numpy()

                # new_transform_img = torch.stack(newimg_list).cuda()
                # outpur_mask0 = self.mask_model(new_transform_img)
                # outpur_mask1 = outpur_mask0.detach().permute(0,2,3,1).cpu().numpy()

                for i977 in range(len(newimg_index_list)):
                    batch_n_mask = outpur_mask1[i977, :, :, :]
                    tmp_y_img_index = newimg_index_list[i977][0]
                    tmp_y_mask_index = newimg_index_list[i977][1]
                    batch_n_mask_real = batch_n_mask[tmp_y_mask_index[0]:tmp_y_mask_index[1], tmp_y_mask_index[2]:tmp_y_mask_index[3]]
                    tmp_mask = mask[tmp_y_img_index[0]:tmp_y_img_index[1], tmp_y_img_index[2]:tmp_y_img_index[3]] + batch_n_mask_real
                    mask[tmp_y_img_index[0]:tmp_y_img_index[1], tmp_y_img_index[2]:tmp_y_img_index[3]] = tmp_mask
            elif min(tmp_img.shape[0], tmp_img.shape[1]) > 1024:
                pred_size = (1024, 1024)
                newimg_list = []
                newimg_index_list = []
                for h in range(0, svs_H, round(pred_size[0] * step)):
                    for w in range(0, svs_W, round(pred_size[1] * step)):
                        title = tmp_img[h:h + pred_size[0], w:w + pred_size[0]]
                        img_index = [h, h + pred_size[0], w, w + pred_size[0]]
                        mask_index = [0, pred_size[0], 0, pred_size[1]]
                        if (title.shape[0] != pred_size[0]) | (title.shape[1] != pred_size[1]):
                            title_tmp = np.zeros((pred_size[0], pred_size[1], 3), dtype=np.uint8)
                            title_tmp[:title.shape[0], :title.shape[1]] = title
                            mask_index = [0, title.shape[0], 0, title.shape[1]]
                            title = title_tmp
                        newimg_index_list.append([img_index, mask_index])
                        newimg_list.append(self.trnsfrms_val(title))
                outpur_mask2_list = []
                batch_size_88 = 8
                for i77 in range(0, len(newimg_list), batch_size_88):
                    t_imt = newimg_list[i77:(i77 + batch_size_88)]
                    outpur_mask0 = self.mask_model(torch.stack(t_imt).to(self.device))
                    outpur_mask00 = outpur_mask0.detach().permute(0, 2, 3, 1).cpu()
                    outpur_mask2_list.append(outpur_mask00)
                # outpur_mask1 = torch.cat(outpur_mask2_list).numpy()
                outpur_mask1 = torch.concat(outpur_mask2_list).numpy()

                # new_transform_img = torch.stack(newimg_list).cuda()
                # outpur_mask0 = self.mask_model(new_transform_img)
                # outpur_mask1 = outpur_mask0.detach().permute(0,2,3,1).cpu().numpy()

                for i977 in range(len(newimg_index_list)):
                    batch_n_mask = outpur_mask1[i977, :, :, :]
                    tmp_y_img_index = newimg_index_list[i977][0]
                    tmp_y_mask_index = newimg_index_list[i977][1]
                    batch_n_mask_real = batch_n_mask[tmp_y_mask_index[0]:tmp_y_mask_index[1], tmp_y_mask_index[2]:tmp_y_mask_index[3]]
                    tmp_mask = mask[tmp_y_img_index[0]:tmp_y_img_index[1], tmp_y_img_index[2]:tmp_y_img_index[3]] + batch_n_mask_real
                    mask[tmp_y_img_index[0]:tmp_y_img_index[1], tmp_y_img_index[2]:tmp_y_img_index[3]] = tmp_mask

        try:
            del tmp_img, tmp_mask, outpur_mask1, outpur_mask2_list, outpur_mask00, title_tmp, title
            gc.collect()
        except:
            pass
        try:
            del tmp_img, tmp_mask, outpur_mask1, outpur_mask2_list, outpur_mask00, title_tmp, title
            gc.collect()
        except:
            pass
        try:
            del tmp_img, tmp_mask, outpur_mask1, outpur_mask2_list, outpur_mask00, title_tmp, title
            gc.collect()
        except:
            pass

        tmppp_mask = (np.argmax(mask, axis=2).astype(dtype=np.uint8))
        pre_mask_rever = morphology.remove_small_objects((tmppp_mask == 1), min_size=contour_area_threshold)
        tmppp_mask1 = np.uint8(pre_mask_rever)

        pre_mask_rever = morphology.remove_small_objects((tmppp_mask1 == 0), min_size=contour_area_threshold)
        tmppp_mask2 = np.uint8(pre_mask_rever)
        tmppp_mask1[tmppp_mask2 == 0] = 1

        return tmppp_mask1
