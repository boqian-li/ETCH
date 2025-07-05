import os
import random
import pickle

folder_model = "datafolder/CAPE_reorganized/cape_release/model_reorganized"
folder_smpl = "datafolder/CAPE_reorganized/cape_release/smpl_reorganized"
save_dir = "datafolder/useful_data_cape/"

# id_list_model = [id for id in os.listdir(folder_model) if os.path.isdir(os.path.join(folder_model, id))]
# id_list_smpl = [id for id in os.listdir(folder_smpl) if os.path.isdir(os.path.join(folder_smpl, id))]
# id_list_final = list(set(id_list_model) & set(id_list_smpl))
# print(len(id_list_final))

# # random split
# random.seed(420)
# random.shuffle(id_list_final)

# train_ratio = 0.8
# train_ids = id_list_final[:int(len(id_list_final) * train_ratio)]
# val_ids = id_list_final[int(len(id_list_final) * train_ratio):]
# print("train_ids: ", sorted(train_ids))
# print("val_ids: ", sorted(val_ids))

# ['00032', '00096', '00127', '00134', '00145', '02474', '03223', '03284', '03331', '03375', '03383', '03394'] + ['00122', '00159', '00215']:

train_subjects = ['00032', '00096', '00127', '00134', '00145', '02474', '03223', '03284', '03331', '03375', '03383', '03394']
val_subjects = ['00122', '00159', '00215']

train_ids = []
val_ids = []

for fn in os.listdir(folder_model):
    if os.path.isdir(os.path.join(folder_model, fn)) and os.path.isdir(os.path.join(folder_smpl, fn)):
        if fn.split('_')[0] in train_subjects:
            assert fn not in train_ids, f"Warning: {fn} is duplicated in train_ids"
            train_ids.append(fn)
        elif fn.split('_')[0] in val_subjects:
            assert fn not in val_ids, f"Warning: {fn} is duplicated in val_ids"
            val_ids.append(fn)
        else:
            assert False, f"Warning: {fn} is not in train or val subjects"

print(f"train_ids: {len(train_ids)}")
print(f"val_ids: {len(val_ids)}")

with open(os.path.join(save_dir, "train_ids.pkl"), "wb") as f:
    pickle.dump(train_ids, f)
with open(os.path.join(save_dir, "val_ids.pkl"), "wb") as f:
    pickle.dump(val_ids, f)
