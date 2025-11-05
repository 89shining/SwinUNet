"""
将 nii.gz 转换为 SwinUNet 数据格式
✅ 固定 train/test 文件夹
✅ 窗宽窗位归一化 (0-1)
✅ 生成 train/test list
"""

import os

import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import h5py
import random

# ===================== 参数设置 =====================
root_dir = r"/home/wusi/SAMdata/20250711_GTVp/datanii"
train_dir = os.path.join(root_dir, "train_nii")
test_dir = os.path.join(root_dir, "test_nii")

save_dir = r"/home/wusi/SwinUNet/data/Synapse"
list_dir = r"/home/wusi/SwinUNet/lists/Synapse"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)

train_npz_dir = os.path.join(save_dir, "train_npz")
test_vol_dir = os.path.join(save_dir, "test_vol_h5")
os.makedirs(train_npz_dir, exist_ok=True)
os.makedirs(test_vol_dir, exist_ok=True)

# ------------------ 可选配置 ------------------
make_val = False          # 是否生成验证集
val_ratio = 0.2           # 从 train 中抽出 20% 做验证
RANDOM_SEED = 42          # 固定随机划分
WINDOW_CENTER = 40
WINDOW_WIDTH = 350
# -----------------------------------------------

random.seed(RANDOM_SEED)

# ===================== 工具函数 =====================
def window_normalize(img, center=40, width=350):
    """窗宽窗位 + 归一化到 [0,1]"""
    img = img.astype(np.float32)
    lower = center - width / 2
    upper = center + width / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / width
    img = np.clip(img, 0, 1)
    return img


def save_npz(image, label, save_path):
    """保存为 .npz 格式（2D）"""
    np.savez_compressed(save_path, image=image.astype(np.float32), label=label.astype(np.uint8))


def save_h5(image, label, save_path):
    """保存为 .h5 格式（3D）"""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('image', data=image.astype(np.float32))
        f.create_dataset('label', data=label.astype(np.uint8))


def process_case(pid_path, is_train=True):
    """读取单个病例并保存"""
    pid = os.path.basename(pid_path)
    img_path = os.path.join(pid_path, "image.nii.gz")
    label_path = os.path.join(pid_path, "GTVp.nii.gz")

    if not (os.path.exists(img_path) and os.path.exists(label_path)):
        print(f"⚠️ {pid} 缺少 image 或 GTV 文件，跳过。")
        return []

    img_itk = sitk.ReadImage(img_path)
    label_itk = sitk.ReadImage(label_path)
    img_arr = sitk.GetArrayFromImage(img_itk)  # [D, H, W]
    label_arr = sitk.GetArrayFromImage(label_itk).astype(np.uint8)

    img_arr = window_normalize(img_arr, WINDOW_CENTER, WINDOW_WIDTH)

    names = []
    if is_train:
        for i in range(img_arr.shape[0]):
            img_slice = img_arr[i, :, :]
            label_slice = label_arr[i, :, :]

            # ===== 新增：Resize 到 224×224 =====
            target_size = (224, 224)
            img_slice = cv2.resize(img_slice, target_size, interpolation=cv2.INTER_LINEAR)
            label_slice = cv2.resize(label_slice, target_size, interpolation=cv2.INTER_NEAREST)
            # ====================================

            # # 🚫 跳过完全空层
            # if np.sum(label_slice) == 0:
            #     continue

            save_name = f"{pid}_slice{i:03d}.npz"
            save_path = os.path.join(train_npz_dir, save_name)
            save_npz(img_slice, label_slice, save_path)
            names.append(save_name.replace(".npz", ""))

    else:
        save_name = f"{pid}.npy.h5"
        save_path = os.path.join(test_vol_dir, save_name)
        save_h5(img_arr, label_arr, save_path)
        names.append(pid)

    return names


# ===================== 主流程 =====================
train_patients = sorted([os.path.join(train_dir, d) for d in os.listdir(train_dir) if d.startswith("p_")])
test_patients = sorted([os.path.join(test_dir, d) for d in os.listdir(test_dir) if d.startswith("p_")])

# --- 固定 val 划分 ---
random.shuffle(train_patients)
split_idx = int(len(train_patients) * (1 - val_ratio))
val_patients = train_patients[split_idx:]
train_patients = train_patients[:split_idx]

train_list, val_list, test_list = [], [], []

# --- 处理训练集 ---
for p in tqdm(train_patients, desc="Train set"):
    train_list += process_case(p, is_train=True)

# --- 处理验证集 ---
for p in tqdm(val_patients, desc="Val set"):
    val_list += process_case(p, is_train=True)

# --- 处理测试集 ---
for p in tqdm(test_patients, desc="Test set"):
    test_list += process_case(p, is_train=False)

# ===================== 保存txt列表 =====================
with open(os.path.join(list_dir, "train.txt"), "w") as f:
    f.writelines([f"{x}\n" for x in train_list])

with open(os.path.join(list_dir, "val.txt"), "w") as f:
    f.writelines([f"{x}\n" for x in val_list])

with open(os.path.join(list_dir, "test_vol.txt"), "w") as f:
    f.writelines([f"{x}\n" for x in test_list])

print("✅ 数据准备完成！")
print(f"Train slices: {len(train_list)} | Val slices: {len(val_list)} | Test vols: {len(test_list)}")
print(f"固定随机种子: {RANDOM_SEED}")
