import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse


# ============================================================
# 固定参数（无需命令行输入）
# ============================================================
class Args:
    pass


args = Args()
args.dataset = 'Synapse'                           # 数据集名称
args.root_path = './data/Synapse/train_npz'        # 数据路径
args.list_dir = './lists/lists_Synapse'            # 训练/验证划分文件
args.num_classes = 2                               # 分割类别数（GTVp→1；Synapse→2或9）
args.img_size = 224                                # 输入尺寸
args.batch_size = 16
args.max_epochs = 100
args.max_iterations = 30000
args.n_gpu = 1
args.deterministic = 1
args.base_lr = 0.01
args.seed = 1234
args.eval_interval = 1
args.n_class = args.num_classes
args.cfg = 'configs/swin_tiny_patch4_window7_224_lite.yaml'  # 模型配置文件
args.opts = None
args.zip = False
args.cache_mode = 'no'
args.resume = None
args.accumulation_steps = None
args.use_checkpoint = False
args.amp_opt_level = 'O1'
args.tag = None
args.throughput = False
args.eval = False

# ============================================================
# 初始化环境与随机种子
# ============================================================
if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# ============================================================
# 输出路径逻辑（完全复刻 TransUNet）
# ============================================================
args.is_pretrain = True
args.exp = 'TU_' + args.dataset + str(args.img_size)
snapshot_path = "./output/{}/{}".format(args.exp, 'TU')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_swinunet'        # 模型名
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
print("Snapshot path:", snapshot_path)

# ============================================================
# 模型加载与训练
# ============================================================
config = get_config(args)
net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
net.load_from(config)

print(f"\n===== Start Training SwinUNet =====")
print(f"Dataset: {args.dataset}")
print(f"Data root: {args.root_path}")
print(f"Output dir: {snapshot_path}")
print(f"Image size: {args.img_size}")
print(f"Num classes: {args.num_classes}")
print(f"Batch size: {args.batch_size}")
print(f"Max epochs: {args.max_epochs}")
print("====================================\n")

trainer = {'Synapse': trainer_synapse}
trainer[args.dataset](args, net, snapshot_path)
