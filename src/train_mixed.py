import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import wandb
import tqdm
import trimesh
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils.GT_dataloader import GTDataset, convert_geodesic_distances_to_confidence

# from models.models_pointcloud_eq import GT_network_equiv

from models.models_pointcloud import GT_network_equiv

import matplotlib.pyplot as plt
from collections import defaultdict
import pickle as pkl
from scipy.spatial import cKDTree

import datetime


def default_stuff(seed: int):
    torch.backends.cudnn.allow_tf32 = False
    torch.set_printoptions(precision=8, linewidth=10000, edgeitems=10, threshold=5000)
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 10)


def vis_loss(all_epochs_losses, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    for name in all_epochs_losses.keys():
        plt.figure()
        plt.plot(all_epochs_losses[name], label=f"{name} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{name} Loss")
        plt.legend()

        # 动态设置 y 轴范围
        min_loss = 0
        max_n_loss = sorted(all_epochs_losses[name], reverse=True)[
            min(0, len(all_epochs_losses[name]) - 1)
        ]
        margin = (max_n_loss - min_loss) * 0.1  # 增加 10% 的边距
        plt.ylim(min_loss, max_n_loss + margin)
        plt.xlim(0, len(all_epochs_losses[name]))

        # 获取最后一个点的坐标
        last_epoch = len(all_epochs_losses[name]) - 1
        last_loss = all_epochs_losses[name][-1]

        # 标注最后一个点的损失值
        plt.annotate(
            f"{last_loss:.6f}",
            xy=(last_epoch, last_loss),
            xytext=(last_epoch, last_loss + margin * 0.05),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=8,
            ha="center",
        )

        plt.savefig(os.path.join(log_dir, f"{name}.png"))
        plt.close()


def train(args, model, optimizer, train_loader, writer, global_step):
    model.train()

    pbar = tqdm.tqdm(train_loader)
    epoch_losses = defaultdict(float)
    num_batches = 0

    for batch_data in pbar:
        hitpts = batch_data["hitpts"].to(args.device)  # shape(B, N, 3)
        vectors = batch_data["vectors"].to(args.device)  # shape(B, N, 3)
        confidences = batch_data["confidences"].to(args.device)  # shape(B, N, 1)
        labels = batch_data["labels"].to(args.device)  # shape(B, N)
        B = hitpts.shape[0]

        losses = {}

        PRED_ITEMS = [
            # "confidence",
            "direction",
            "magnitude",
        ]  # ["confidence", "direction", "magnitude"]
        results, selected_indexs = model(
            hitpts, pred_items=PRED_ITEMS, direction_mode="standard_vector"
        )

        # Compute Losses
        if "direction" in PRED_ITEMS:
            pred_directions = results["direction"]  # shape(B, K, 3)
            vectors_k = torch.gather(
                vectors, 1, selected_indexs
            )  # shape(B, N, 3) -> shape(B, K, 3)
            cosine_loss = 1 - F.cosine_similarity(vectors_k, pred_directions, dim=-1)
            losses["direction_loss"] = cosine_loss.mean() * args.direction_w

        if "magnitude" in PRED_ITEMS:
            pred_magnitudes = results["magnitude"]  # shape(B, K, 1)
            vectors_k = torch.gather(
                vectors, 1, selected_indexs
            )  # shape(B, N, 3) -> shape(B, K, 3)
            vector_norms = torch.norm(vectors_k, dim=-1, keepdim=True)  # shape(B, K, 1)
            losses["magnitude_loss"] = (
                F.mse_loss(vector_norms * args.scale_magnitude, pred_magnitudes)
                * args.magnitude_w
            )

        if "confidence" in PRED_ITEMS:
            pred_part_labels = results["part_labels"]  # shape(B, K, num_parts)
            pred_confidences = results["confidences"]  # shape (B, K, 1)
            if args.use_dynamic_label_confidence:
                pred_inner_points = (
                    hitpts - pred_directions * pred_magnitudes / args.scale_magnitude
                )  # shape(B, N, 3)
                labels = []
                confidences = []
                for j in range(B):
                    markers_positions_ = batch_data["markers_positions"][
                        j
                    ]  # shape(num_markers, 3)
                    tree = cKDTree(markers_positions_)
                    pred_inner_points_ = (
                        pred_inner_points[j].detach().cpu().numpy()
                    )  # shape(N, 3)
                    dists_, indices_ = tree.query(
                        pred_inner_points_, k=1
                    )  # shape(N,), shape(N,)
                    geodesic_distances_ = dists_.reshape(-1, 1)  # shape(N, 1)
                    labels_ = indices_  # shape(N,)

                    confidences_ = convert_geodesic_distances_to_confidence(
                        geodesic_distances_
                    )  # shape(N, 1)

                    labels.append(
                        torch.from_numpy(labels_).type(dtype=torch.long).to(args.device)
                    )
                    confidences.append(
                        torch.from_numpy(confidences_)
                        .type(dtype=torch.float32)
                        .to(args.device)
                    )

                labels = torch.stack(labels, dim=0)  # shape(B, N)
                confidences = torch.stack(confidences, dim=0)  # shape(B, N, 1)

            labels_k = torch.gather(
                labels, 1, selected_indexs[:, :, 0]
            )  # shape(B, N) -> shape(B, K)
            confidences_k = torch.gather(
                confidences, 1, selected_indexs[:, :, 0].unsqueeze(-1)
            )  # shape(B, N, 1) -> shape(B, K, 1)

            losses["confidence_loss"] = (
                F.mse_loss(pred_confidences, confidences_k) * args.confidence_w
            )
            losses["part_label_loss"] = (
                F.cross_entropy(
                    pred_part_labels.permute(0, 2, 1).contiguous(), labels_k
                )
                * args.part_label_w
            )

        # losses["vector_loss"] = vector_loss(vectors, pred_vectors)

        all_loss = 0.0
        losses_key = losses.keys()
        # print(f"losses_key: {losses_key}")

        for key in losses_key:
            all_loss = all_loss + losses[key]

        if all_loss.isnan().any():
            print("Loss is NaN issue")
            all_loss = 0.0

        # losses["all_loss"] = all_loss
        optimizer.zero_grad()

        all_loss.backward()
        for nm, param in model.named_parameters():
            if param.grad is None:
                pass
            elif param.grad.sum().isnan():
                param.grad = torch.zeros_like(param.grad)

        optimizer.step()

        # Log batch metrics to wandb or tensorboard
        batch_metrics = {
            "train_step/total_loss": all_loss.item(),
            "train_step/direction_loss": losses["direction_loss"].item(),
            "train_step/magnitude_loss": losses["magnitude_loss"].item(),
            # "train_step/part_label_loss": losses["part_label_loss"].item(),
            # "train_step/confidence_loss": losses["confidence_loss"].item(),
            "train_step/learning_rate": optimizer.param_groups[0]["lr"],
            # "batch/step": num_batches,
        }
        for k, v in batch_metrics.items():
            writer.add_scalar(k, v, global_step)
        global_step += 1
        # wandb.log(batch_metrics)
        # losses_key.append("total_loss")
        losses["total_loss"] = all_loss
        # print(f"losses_key: {losses_key}")
        desc = "Batch losses: "
        for key in losses_key:
            desc += f"{key}: {losses[key].item():.05f}, "
        pbar.set_description(desc)

        num_batches += 1
        for key in losses_key:
            epoch_losses[key] += losses[key].item()

    for key in epoch_losses.keys():
        epoch_losses[key] /= num_batches

    return epoch_losses


def main():
    parser = argparse.ArgumentParser(description="PyTorch Template")

    parser.add_argument("--dynamic_start_epoch", type=int, default=9999)
    parser.add_argument("--aug", type=bool, default=False)
    parser.add_argument(
        "--activated_ids_path",
        type=str,
        # default="datafolder/useful_data_gen-data/train_ids.pkl",
        # default="datafolder/useful_data_cape/train_ids.pkl",
        default="datafolder/useful_data_4d-dress/train_ids.pkl",
        help="activated ids",
    )  # datafolder/useful_data_cape/train_ids.pkl #
    parser.add_argument(
        "--scale_magnitude", type=int, default=10, help="scale for magnitude"
    )
    parser.add_argument(
        "--markerset_path",
        # default="datafolder/useful_data_gen-data/superset_smpl.json",
        # default="datafolder/useful_data_cape/superset_smpl.json",
        default="datafolder/useful_data_4d-dress/superset_smpl.json",
        type=str,
    )
    parser.add_argument(
        "--vis_loss",
        type=bool,
        default=False,
        help="whether visualize loss when training or not",
    )
    parser.add_argument(
        "--infopoints_dir",
        type=str,
        # default="datafolder/gt_gen_data_centered/npz",
        # default="datafolder/gt_CAPE_data/npz",
        default="datafolder/gt_4D-Dress_data/npz",
        help="data dir path of npy files",
    )  #  # datafolder/gt_CAPE_data/npz
    parser.add_argument(
        "--scan_dir",
        type=str,
        # default="datafolder/Generative_data/gen_data_reorganized/model",
        # default="datafolder/CAPE_reorganized/cape_release/model_reorganized",
        default="datafolder/4D-DRESS/data_processed/model",
        help="data dir path of obj files",
    )  # datafolder/CAPE_reorganized/cape_release/model_reorganized #
    parser.add_argument(
        "--smpl_dir",
        type=str,
        # default="datafolder/Generative_data/gen_data_reorganized/smplh",
        # default="datafolder/CAPE_reorganized/cape_release/smpl_reorganized",
        default="datafolder/4D-DRESS/data_processed/smplh",
        help="data dir path of smpl files",
    )  # datafolder/CAPE_reorganized/cape_release/smpl_reorganized #

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    # parser.add_argument(
    #     "--latent_num", type=int, default=128, metavar="N", help="input batch size for training (default: 128)"
    # )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    # lr 绝对不要调大除非你知道你在做什么，1e-3试过，loss会很波动，结果也不好。
    parser.add_argument(
        "--lr", type=float, default=1e-4, metavar="N", help="learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    parser.add_argument(
        "--num_point",
        type=int,
        default=5000,
        metavar="N",
        help="point num sampled from mesh surface",
    )
    parser.add_argument(
        "--aug_type", type=str, default="so3", metavar="N", help="so3, zrot, no"
    )
    # parser.add_argument("--gt_part_seg", type=str, default="auto", metavar="N", help="")
    parser.add_argument(
        "--EPN_input_radius",
        type=float,
        default=0.4,
        help="train from pretrained model",
    )
    parser.add_argument(
        "--EPN_layer_num",
        type=int,
        default=2,
        metavar="N",
        help="point num sampled from mesh surface",
    )
    # parser.add_argument(
    #     "--kinematic_cond", type=str, default="yes", metavar="N", help="point num sampled from mesh surface"
    # )

    parser.add_argument("--direction_w", type=float, default=1.0, help="")  # 2. # 2.5
    parser.add_argument("--magnitude_w", type=float, default=1.0, help="")  # 30. # 20
    parser.add_argument("--part_label_w", type=float, default=1.0, help="")  # 1.
    parser.add_argument("--confidence_w", type=float, default=1.0, help="")  # 50.
    parser.add_argument("--i", type=str, default=None, help="")
    # parser.add_argument("--part_w", type=float, default=5, help="")
    # parser.add_argument("--angle_w", type=float, default=5, help="")
    # parser.add_argument("--jpos_w", type=float, default=1e2, help="")
    # parser.add_argument("--vertex_w", type=float, default=1e2, help="")
    # parser.add_argument("--normal_w", type=float, default=1e0, help="")
    # parser.add_argument('--skip_handsfeet', default=True, type=bool)
    # parser.add_argument('--aug', default=False, type=bool, help='whether to use augmentation or not')
    # parser.add_argument("--rep_type", type=str, default="6d", metavar="N", help="aa, 6d")
    # parser.add_argument("--part_num", type=int, default=22, metavar="N", help="part num of the SMPL body")

    # Add datetime import for automatic naming
    # import datetime

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.cuda
    args.cuda_visible_devices = torch.cuda.device_count()
    args_dict = dict(vars(args))

    default_stuff(args.seed)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    # initialize output folder
    exps_folder = "EPN_layer_{}_radius_{}_aug_{}".format(
        args.EPN_layer_num,
        args.EPN_input_radius,
        args.aug_type,
    )
    exps_folder = exps_folder + f"_num_point_{args.num_point}"

    # if args.i is not None:
    #     exps_folder = exps_folder + f"_{args.i}"
    output_folder = os.path.sep.join(
        ["./all_experiments/experiments", exps_folder, f"{args.i}"]
    )
    # wandb.save(os.path.join(output_folder, "config.json"))
    args.output_folder = output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, "checkpoints")):
        os.makedirs(os.path.join(output_folder, "checkpoints"))

    # Initialize tensorboard writer with timestamp-based name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_name = f"{timestamp}_{args.i}_train" if args.i else timestamp
    writer = SummaryWriter(
        log_dir=os.path.join(
            os.path.sep.join(["./all_experiments/experiments", exps_folder]),
            "tensorboard_logs",
            tensorboard_name,
        )
    )

    # save training args
    args_save_path = os.path.join(args.output_folder, "training_args.json")
    with open(args_save_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    print(f"Training args saved to {args_save_path}")

    # load markerset
    with open(args.markerset_path, "r") as f:
        args.markerset = json.load(f)

    model = GT_network_equiv(
        option=args,
    )
    if args.cuda_visible_devices > 1:
        print("======== Running on multiple GPUs ========")
        model = torch.nn.DataParallel(model)
    else:
        print("======== Running on single GPU ========")

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # val_args = args.copy()
    # val_args = parser.parse_args()
    # val_args.activated_ids_path = "datafolder/useful_data_gen-data/val_ids.pkl"
    # val_args.markerset = args.markerset
    # val_args.output_folder = args.output_folder
    # val_args.device = args.device

    # indices = torch.randperm(4000)[:100]
    train_dataset_4d_dress = GTDataset(args)
    # 随机选取1/6的数据
    # num_total = len(full_train_dataset_4d_dress)
    # num_subset = max(1, num_total // 6)
    # subset_indices = torch.randperm(num_total)[:num_subset]
    # train_dataset_4d_dress = torch.utils.data.Subset(
    #     full_train_dataset_4d_dress, subset_indices
    # )

    args_gen = parser.parse_args()

    args_gen.scan_dir = "datafolder/Generative_data/gen_data_reorganized/model"
    args_gen.smpl_dir = "datafolder/Generative_data/gen_data_reorganized/smplh"
    args_gen.infopoints_dir = "datafolder/gt_Gen_data/npz"
    args_gen.markerset_path = "datafolder/useful_data_gen-data/superset_smpl.json"
    args_gen.activated_ids_path = "datafolder/useful_data_gen-data/train_ids.pkl"
    with open(args_gen.markerset_path, "r") as f:
        args_gen.markerset = json.load(f)
    train_dataset_gen = GTDataset(args_gen)

    args_cape = parser.parse_args()
    args_cape.scan_dir = "datafolder/CAPE_reorganized/cape_release/model_reorganized"
    args_cape.smpl_dir = "datafolder/CAPE_reorganized/cape_release/smpl_reorganized"
    args_cape.infopoints_dir = "datafolder/gt_CAPE_data/npz"
    args_cape.markerset_path = "datafolder/useful_data_cape/superset_smpl.json"
    args_cape.activated_ids_path = "datafolder/useful_data_cape/train_ids.pkl"
    with open(args_cape.markerset_path, "r") as f:
        args_cape.markerset = json.load(f)
    train_dataset_cape = GTDataset(args_cape)

    train_dataset_mixed = torch.utils.data.ConcatDataset(
        [train_dataset_4d_dress, train_dataset_cape, train_dataset_gen]
    )
    # train_dataset = torch.utils.data.Subset(train_dataset, indices)

    print(f"train_dataset_4d_dress length: {len(train_dataset_4d_dress)}")
    print(f"train_dataset_gen length: {len(train_dataset_gen)}")
    print(f"train_dataset_cape length: {len(train_dataset_cape)}")
    print(f"train_dataset_mixed length: {len(train_dataset_mixed)}")
    train_loader = DataLoader(
        # train_dataset_gen,
        train_dataset_mixed,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=16,
        prefetch_factor=2,
    )
    # exit()

    if args.vis_loss:
        log_dir = os.path.join(output_folder, "log_all")
        os.makedirs(log_dir, exist_ok=True)
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(output_folder, "checkpoints", "model_epochs_00000039.pth")
    #     )
    # )
    all_epochs_losses = defaultdict(list)
    global_step = 0
    # global_step = 40 * len(train_loader)
    for epoch in range(0, args.epochs):
        print(f"====== Epoch {epoch} start! ======")
        args.use_dynamic_label_confidence = (
            True if epoch > args.dynamic_start_epoch - 1 else False
        )
        # val_args.use_dynamic_label_confidence = args.use_dynamic_label_confidence
        if args.use_dynamic_label_confidence:
            print("== Using dynamic label and confidence ==")
        average_epoch_losses = train(
            args, model, optimizer, train_loader, writer, global_step
        )
        global_step += len(train_loader)

        # print losses per epoch
        desc = f"===== Average losses of Epoch {epoch} are: "
        for key in average_epoch_losses.keys():
            desc += f"{key}: {average_epoch_losses[key]:.05f}, "
        desc += "======"
        print(desc)

        # Log epoch metrics to tensorboard
        writer.add_scalar(
            "train_epoch/direction_loss", average_epoch_losses["direction_loss"], epoch
        )
        writer.add_scalar(
            "train_epoch/magnitude_loss", average_epoch_losses["magnitude_loss"], epoch
        )
        writer.add_scalar(
            "train_epoch/total_loss", average_epoch_losses["total_loss"], epoch
        )

        # visualize losses
        if args.vis_loss:
            for key in average_epoch_losses.keys():
                all_epochs_losses[key].append(average_epoch_losses[key])
            vis_loss(all_epochs_losses, os.path.join(log_dir, "train"))

        # save model
        if epoch % 1 == 0 or epoch == args.epochs - 1:
            torch.save(
                (
                    model.module.state_dict()
                    if (args.cuda_visible_devices > 1)
                    else model.state_dict()
                ),
                os.path.join(
                    output_folder, "checkpoints", f"model_epochs_{epoch:08d}.pth"
                ),
            )

    writer.close()


if __name__ == "__main__":
    main()
