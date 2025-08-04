import argparse
import os

import numpy as np
import torch
import tqdm
import trimesh
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_utils.GT_dataloader import GTDataset, convert_geodesic_distances_to_confidence
from models.models_pointcloud import GT_network_equiv

import matplotlib.pyplot as plt
from collections import defaultdict
import pickle as pkl
import json
from scipy.spatial import cKDTree


def default_stuff(seed:int):
    torch.backends.cudnn.allow_tf32 = False
    torch.set_printoptions(precision=8, linewidth=10000, edgeitems=10, threshold=5000)
    torch.manual_seed(seed+1)
    np.random.seed(seed+10)

def vis_loss(all_epochs_losses, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    for name in all_epochs_losses.keys():
        plt.figure()
        plt.plot(all_epochs_losses[name], label=f'{name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} Loss')
        plt.legend()

        # Dynamically set y-axis range
        min_loss = 0
        max_n_loss = sorted(all_epochs_losses[name], reverse=True)[min(0, len(all_epochs_losses[name])-1)]
        margin = (max_n_loss - min_loss) * 0.1  # Add 10% margin
        plt.ylim(min_loss, max_n_loss + margin)
        plt.xlim(0, len(all_epochs_losses[name]))

        # Get coordinates of the last point
        last_epoch = len(all_epochs_losses[name]) - 1
        last_loss = all_epochs_losses[name][-1]

        # Annotate the loss value of the last point
        plt.annotate(f'{last_loss:.6f}', 
                     xy=(last_epoch, last_loss), 
                     xytext=(last_epoch, last_loss + margin * 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=8,
                     ha='center')

        plt.savefig(os.path.join(log_dir, f'{name}.png'))
        plt.close()  

def train(args, model, optimizer, train_loader):
    model.train()

    pbar = tqdm.tqdm(train_loader)
    epoch_losses = defaultdict(float)
    num_batches = 0

    for batch_data in pbar:
        hitpts = batch_data["hitpts"].to(args.device) #shape(B, N, 3)
        vectors = batch_data["vectors"].to(args.device) #shape(B, N, 3)
        confidences = batch_data["confidences"].to(args.device) #shape(B, N, 1)
        labels = batch_data["labels"].to(args.device) #shape(B, N)
        B = hitpts.shape[0]

        losses = {}

        PRED_ITEMS = ["confidence", "direction", "magnitude"] # ["confidence", "direction", "magnitude"]
        results, selected_indexs = model(hitpts, pred_items=PRED_ITEMS, direction_mode="standard_vector")


        # Compute Losses
        if "direction" in PRED_ITEMS:
            pred_directions = results["direction"] # shape(B, K, 3)
            vectors_k = torch.gather(vectors, 1, selected_indexs) # shape(B, N, 3) -> shape(B, K, 3)
            cosine_loss = 1 - F.cosine_similarity(vectors_k, pred_directions, dim=-1)
            losses['direction_loss'] = cosine_loss.mean() * args.direction_w

        if "magnitude" in PRED_ITEMS:
            pred_magnitudes = results["magnitude"] # shape(B, K, 1)
            vectors_k = torch.gather(vectors, 1, selected_indexs) # shape(B, N, 3) -> shape(B, K, 3)
            vector_norms = torch.norm(vectors_k, dim=-1, keepdim=True) # shape(B, K, 1)
            losses['magnitude_loss'] = F.mse_loss(vector_norms * args.scale_magnitude, pred_magnitudes) * args.magnitude_w

        if "confidence" in PRED_ITEMS:
            pred_part_labels = results["part_labels"] # shape(B, K, num_parts)
            pred_confidences = results["confidences"] # shape (B, K, 1) 

            labels_k = torch.gather(labels, 1, selected_indexs[:, :, 0]) # shape(B, N) -> shape(B, K)
            confidences_k = torch.gather(confidences, 1, selected_indexs[:, :, 0].unsqueeze(-1)) # shape(B, N, 1) -> shape(B, K, 1)

            losses["confidence_loss"] = F.mse_loss(pred_confidences, confidences_k) * args.confidence_w
            losses["part_label_loss"] = F.cross_entropy(pred_part_labels.permute(0, 2, 1).contiguous(), labels_k) * args.part_label_w

        # losses["vector_loss"] = vector_loss(vectors, pred_vectors)

        all_loss = 0.0
        losses_key = losses.keys()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--activated_ids_path", type=str, default="datafolder/useful_data_4d-dress/train_ids.pkl", help="activated ids")
    parser.add_argument("--scale_magnitude", type=int, default=10, help="scaling ratio for magnitude")
    parser.add_argument('--markerset_path', default="datafolder/useful_data_4d-dress/superset_smpl.json", type=str)
    parser.add_argument("--vis_loss", type=bool, default=True, help="whether visualize loss when training or not")
    parser.add_argument("--infopoints_dir", type=str, default="datafolder/gt_4D-Dress_data/npz", help="data dir path of npy files") #  # datafolder/gt_CAPE_data/npz
    parser.add_argument("--scan_dir", type=str, default="datafolder/4D-DRESS/data_processed/model", help="data dir path of obj files")  # datafolder/CAPE_reorganized/cape_release/model_reorganized # 
    parser.add_argument("--smpl_dir", type=str, default="datafolder/4D-DRESS/data_processed/smplh", help="data dir path of smpl files")  # datafolder/CAPE_reorganized/cape_release/smpl_reorganized # 


    parser.add_argument(
        "--batch_size", type=int, default=1, metavar="N", help="input batch size for training"
    )

    parser.add_argument("--epochs", type=int, default=30, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="N", help="learning rate")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")

    parser.add_argument("--num_point", type=int, default=5000, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--EPN_input_radius", type=float, default=0.4)
    parser.add_argument("--EPN_layer_num", type=int, default=2, metavar="N")


    parser.add_argument("--direction_w", type=float, default=1., help="") # 2. # 2.5
    parser.add_argument("--magnitude_w", type=float, default=1., help="") # 30. # 20
    parser.add_argument("--part_label_w", type=float, default=1., help="") # 1.
    parser.add_argument("--confidence_w", type=float, default=1., help="") #50.
    parser.add_argument("--i", type=str, default=None, help="")


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.cuda
    args.cuda_visible_devices = torch.cuda.device_count()
    args_dict = dict(vars(args))

    default_stuff(args.seed)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    # initialize output folder
    exps_folder = "EPN_layer_{}_radius_{}".format(
        args.EPN_layer_num,
        args.EPN_input_radius,
    )
    exps_folder = exps_folder + f"_num_point_{args.num_point}"
    if args.i is not None:
        exps_folder = exps_folder + f"_{args.i}"
    output_folder = os.path.sep.join(["./all_experiments/experiments", exps_folder])
    args.output_folder = output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # save training args
    args_save_path = os.path.join(args.output_folder, 'training_args.json')
    with open(args_save_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f'Training args saved to {args_save_path}')


    # load markerset
    with open(args.markerset_path, 'r') as f:
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

    dataset = GTDataset(args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=16, prefetch_factor=2)

    if args.vis_loss:
        log_dir = os.path.join(output_folder, "log_all")
        os.makedirs(log_dir, exist_ok=True)
    
    all_epochs_losses = defaultdict(list)
    for epoch in range(args.epochs):
        print(f"====== Epoch {epoch} start! ====== \n")
     
        average_epoch_losses = train(args, model, optimizer, train_loader)

        # print losses per epoch
        desc = f"===== Average losses of Epoch {epoch} are: "
        for key in average_epoch_losses.keys():
            desc += f"{key}: {average_epoch_losses[key]:.05f}, "
        desc += "======\n"
        print(desc)
        
        # visualize losses
        if args.vis_loss:
            for key in average_epoch_losses.keys():
                all_epochs_losses[key].append(average_epoch_losses[key])
            vis_loss(all_epochs_losses, os.path.join(log_dir, 'train'))

        # save model
        if epoch % 1 == 0 or epoch == args.epochs - 1:
            torch.save(
                model.module.state_dict() if (args.cuda_visible_devices > 1) else model.state_dict(),
                os.path.join(output_folder, f"model_epochs_{epoch:08d}.pth"),
            )
