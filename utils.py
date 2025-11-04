import os
import re
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), \
            f'predict {inputs.size()} & target {target.size()} shape mismatch'
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1):
    """
    通用预测函数（支持 SwinUNet / TransUNet）
    自动保持原 CT 空间信息
    """
    # --- 安全 squeeze，防止多余维度 ---
    image = image.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()

    if len(image.shape) == 3:  # 多层
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            gt_slice = label[ind, :, :]
            x, y = slice.shape
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0]/x, patch_size[1]/y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                out = zoom(out, (x/patch_size[0], y/y_patch[1]), order=0)
            prediction[ind] = out
    else:  # 单层
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().numpy()

    # --- 指标 ---
    metric_list = [calculate_metric_percase(prediction == i, label == i)
                   for i in range(1, classes)]

    # --- 保存 ---
    if test_save_path:
        os.makedirs(test_save_path, exist_ok=True)
        match = re.search(r'\d+', case or '')
        case_id = int(match.group(0)) if match else 0
        save_name = f"GTVp_{case_id:03d}.nii.gz"
        pred_path = os.path.join(test_save_path, save_name)

        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

        ref_path = os.path.join("/home/wusi/SAMdata/20250711_GTVp/datanii/test_nii", case, "image.nii.gz")
        if os.path.exists(ref_path):
            ref_img = sitk.ReadImage(ref_path)
            for itk_img in [img_itk, prd_itk, lab_itk]:
                itk_img.CopyInformation(ref_img)
            print(f"✅ 已继承原始 CT 空间信息: {case}")
        else:
            for itk_img in [img_itk, prd_itk, lab_itk]:
                itk_img.SetSpacing((1, 1, z_spacing))
            print(f"⚠️ 未找到参考 CT, 使用默认 spacing=(1,1,{z_spacing})")

        sitk.WriteImage(prd_itk, pred_path)
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))
        print(f"已保存预测文件: {save_name}")

    return metric_list
