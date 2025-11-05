import os
import cv2
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
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
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1):
    """
    测试单个病例，并自动将预测结果恢复至原始CT尺寸保存为nii.gz。
    """

    # 安全去除 batch 和通道维度
    image = image.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    # 若形状是 [1, D, H, W] 或 [1, 1, D, H, W]
    if image.ndim == 5:
        image = image[0, 0]
        label = label[0, 0]
    elif image.ndim == 4:
        image = image[0]
        label = label[0]

    # ==== 读取原始图像的尺寸信息 ====
    ref_path = os.path.join("/home/wusi/SAMdata/20250711_GTVp/datanii/test_nii", case, "image.nii.gz")
    orig_size, ref_img = None, None
    if os.path.exists(ref_path):
        ref_img = sitk.ReadImage(ref_path)
        orig_shape = sitk.GetArrayFromImage(ref_img).shape  # (D, H, W)
        orig_size = (orig_shape[2], orig_shape[1])  # (W, H)
        print(f"[{case}] 原始尺寸: {orig_shape}")
    else:
        print(f"⚠️ 未找到参考 CT：{ref_path}")

    # ==== 模型推理 ====
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input_tensor = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

            net.eval()
            with torch.no_grad():
                outputs = net(input_tensor)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input_tensor), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    # ==== 计算评估指标 ====
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # ==== 保存结果 ====
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)

        import re
        match = re.search(r'\d+', case)
        case_id = int(match.group(0)) if match else 0
        save_name = f"GTVp_{case_id:03d}.nii.gz"
        pred_path = os.path.join(test_save_path, save_name)

        # ==== 恢复回原始尺寸 ====
        if orig_size is not None:
            d, h, w = prediction.shape
            target_w, target_h = orig_size
            restored_pred = np.zeros((d, target_h, target_w), dtype=np.uint8)
            for i in range(d):
                restored_pred[i] = cv2.resize(prediction[i].astype(np.uint8),
                                              (target_w, target_h),
                                              interpolation=cv2.INTER_NEAREST)
            prediction = restored_pred
            print(f"[{case}] 已恢复至原始尺寸: {prediction.shape[::-1]}")

        # ==== 转回 ITK 图像 ====
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))

        # ==== 空间信息 ====
        if ref_img is not None:
            for itk_img in [img_itk, prd_itk, lab_itk]:
                itk_img.CopyInformation(ref_img)
            print(f"已继承原始 CT 空间信息: {case}")
        else:
            print(f"未找到参考 CT: {case}，使用默认 spacing=(1,1,{z_spacing})")
            for itk_img in [img_itk, prd_itk, lab_itk]:
                itk_img.SetSpacing((1, 1, z_spacing))

        sitk.WriteImage(prd_itk, pred_path)
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))
        print(f"✅ 已保存预测文件: {save_name}")

    return metric_list
