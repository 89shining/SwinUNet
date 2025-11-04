import argparse
import logging
import os
import random
import sys
import numpy as np
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # 自动读取对应 CT 的真实 z_spacing
        ref_path = os.path.join("/home/wusi/SAMdata/20250711_GTVp/datanii/test_nii", case_name, "image.nii.gz")
        if os.path.exists(ref_path):
            ref_img = sitk.ReadImage(ref_path)
            z_spacing = ref_img.GetSpacing()[2]
        else:
            z_spacing = 1.0
        print(f"{case_name} 真实 z_spacing = {z_spacing:.3f}")

        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name,
                                      z_spacing=z_spacing)
        if metric_i is None:
            print(f"⚠️ 跳过病例 {case_name}（返回None）")
            continue

        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %.4f mean_hd95 %.4f' %
                     (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %.4f mean_hd95 %.4f' %
                     (i, metric_list[i - 1][0], metric_list[i - 1][1]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice: %.4f, mean_hd95: %.4f' %
                 (performance, mean_hd95))
    print(f"\n===== 测试完成 =====\nMean Dice: {performance:.4f}, Mean HD95: {mean_hd95:.4f}\n")
    return "Testing Finished!"


if __name__ == "__main__":
    # ============================================================
    # 固定参数（无需命令行输入）
    # ============================================================
    class Args:
        pass

    args = Args()
    args.dataset = 'Synapse'
    args.volume_path = './data/Synapse/test_vol_h5'
    args.list_dir = './lists/lists_Synapse'
    args.num_classes = 2
    args.img_size = 224
    args.batch_size = 16
    args.max_epochs = 100
    args.base_lr = 0.01
    args.seed = 1234
    args.is_pretrain = True
    args.is_savenii = True
    args.cfg = 'configs/swin_tiny_patch4_window7_224_lite.yaml'
    args.Dataset = Synapse_dataset

    # ============================================================
    # 初始化
    # ============================================================
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ============================================================
    # snapshot_path（与训练完全一致）
    # ============================================================
    args.exp = 'TU_' + args.dataset + str(args.img_size)
    snapshot_path = f"./output/{args.exp}/TU_pretrain_swinunet_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}_{args.img_size}"
    print(f"\n📂 Snapshot path: {snapshot_path}\n")

    # ============================================================
    # 模型加载
    # ============================================================
    config = get_config(args)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
    net.load_state_dict(torch.load(snapshot))
    print("✅ Loaded checkpoint:", snapshot)

    # ⚠️ 删除旧的 snapshot_name 逻辑，改成下面这行
    snapshot_name = os.path.basename(snapshot_path)

    # ============================================================
    # 日志路径（与 TransUNet 一致）
    # ============================================================
    log_folder = f'./output/test_log_{args.exp}'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, f'{snapshot_name}.txt'),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # ============================================================
    # 输出目录（完全一致）
    # ============================================================
    if args.is_savenii:
        args.test_save_dir = './output/predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    # ============================================================
    # 推理
    # ============================================================
    inference(args, net, test_save_path)
