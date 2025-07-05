import os
import numpy as np


# pred_dir = "/home/boqian/code/Generative_Tightness_workspace/all_experiments/experiments/eval_outputs_4d-dress_exppt_epoch21_1.939"
# gt_dir = "/home/boqian/code/Generative_Tightness_workspace/datafolder/4D-DRESS/data_processed/smplh"

pred_dir = "/home/boqian/code/Generative_Tightness_workspace/all_experiments/experiments/eval_outputs_cape_exppt_epoch39_1.647"
gt_dir = "/home/boqian/code/Generative_Tightness_workspace/datafolder/CAPE_reorganized/cape_release/smpl_reorganized"

MPJPE_error_all = 0.0
sample_num = 0

for file in os.listdir(pred_dir):
    if os.path.isdir(os.path.join(pred_dir, file)):
        id_ = file
        gt_info = np.load(os.path.join(gt_dir, id_, f"info_{id_}.npz"))
        pred_info = np.load(os.path.join(pred_dir, id_, f"output_smpl_info_{id_}.npz"))

        gt_joints = gt_info["joints"]
        pred_joints = pred_info["joints"]

        joints_num_considered = 22
        MPJPE_error = np.linalg.norm(pred_joints[:joints_num_considered, :] - gt_joints[:joints_num_considered, :], axis=-1).mean()
        print(MPJPE_error)

        MPJPE_error_all += MPJPE_error
        sample_num += 1


print("mean MPJPE: ", MPJPE_error_all / sample_num) #  0.011160592885033193 4D-Dress # 0.009222339536359828 CAPE
print("count: ", sample_num)



