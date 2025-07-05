import os
import random
import pickle
from tqdm import tqdm

folder_model = "datafolder/4D-DRESS/data_processed/model"
folder_smpl = "datafolder/4D-DRESS/data_processed/smplh"
save_dir = "datafolder/useful_data_4d-dress/"

SUBJ_OUTFIT_SEQ_HUMANCLOTHRECON_HUMANPARSING = {

    '00122': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take8'],
        'Outer': ['Take11', 'Take16'],
    },

    '00123': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take5'],
        'Outer': ['Take10', 'Take11'],
    },

    '00127': {
        'Gender': 'male',
        'Inner': ['Take8', 'Take9'],
        'Outer': ['Take16', 'Take18'],
    },

    '00129': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take5'],
        'Outer': ['Take11', 'Take13'],
    },

    '00134': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take6'],
        'Outer': ['Take12', 'Take19'],
    },

    '00135': {
        'Gender': 'male',
        'Inner': ['Take7', 'Take10'],
        'Outer': ['Take21', 'Take24'],
    },

    '00136': {
        'Gender': 'female',
        'Inner': ['Take8', 'Take12'],
        'Outer': ['Take19', 'Take28'],
    },

    '00137': {
        'Gender': 'female',
        'Inner': ['Take5', 'Take7'],
        'Outer': ['Take16', 'Take19'],
    },

    '00140': {
        'Gender': 'female',
        'Inner': ['Take6', 'Take8'],
        'Outer': ['Take19', 'Take21'],
    },

    '00147': {
        'Gender': 'female',
        'Inner': ['Take11', 'Take12'],
        'Outer': ['Take16', 'Take19'],
    },

    '00148': {
        'Gender': 'female',
        'Inner': ['Take6', 'Take7'],
        'Outer': ['Take16', 'Take19'],
    },

    '00149': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take12'],
        'Outer': ['Take14', 'Take24'],
    },

    '00151': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take9'],
        'Outer': ['Take15', 'Take20'],
    },

    '00152': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take17', 'Take18'],
    },

    '00154': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take9'],
        'Outer': ['Take20', 'Take21'],
    },

    '00156': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take14', 'Take19'],
    },

    '00160': {
        'Gender': 'male',
        'Inner': ['Take6', 'Take7'],
        'Outer': ['Take17', 'Take18'],
    },

    '00163': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take10'],
        'Outer': ['Take13', 'Take15'],
    },

    '00167': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take9'],
        'Outer': ['Take12', 'Take14'],
    },

    '00168': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take7'],
        'Outer': ['Take11', 'Take16'],
    },

    '00169': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take10'],
        'Outer': ['Take17', 'Take19'],
    },

    '00170': {
        'Gender': 'female',
        'Inner': ['Take9', 'Take11'],
        'Outer': ['Take15', 'Take24'],
    },

    '00174': {
        'Gender': 'male',
        'Inner': ['Take6', 'Take9'],
        'Outer': ['Take13', 'Take15'],
    },

    '00175': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take9'],
        'Outer': ['Take13', 'Take20'],
    },

    '00176': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take6'],
        'Outer': ['Take11', 'Take14'],
    },

    '00179': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take13', 'Take15'],
    },

    '00180': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take7'],
        'Outer': ['Take14', 'Take17'],
    },

    '00185': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take8'],
        'Outer': ['Take17', 'Take18'],
    },

    '00187': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take6'],
        'Outer': ['Take10', 'Take15'],
    },

    '00188': {
        'Gender': 'male',
        'Inner': ['Take7', 'Take8'],
        'Outer': ['Take12', 'Take18'],
    },

    '00190': {
        'Gender': 'female',
        'Inner': ['Take2', 'Take7'],
        'Outer': ['Take14', 'Take17'],
    },

    '00191': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take6'],
        'Outer': ['Take13', 'Take19'],
    },

}

print(len(SUBJ_OUTFIT_SEQ_HUMANCLOTHRECON_HUMANPARSING.keys()))

train_ids = []
val_ids_all = []

for fn in tqdm(os.listdir(folder_model)):
    if os.path.isdir(os.path.join(folder_model, fn)) and os.path.isdir(os.path.join(folder_smpl, fn)):
        subject = fn.split('_')[0]
        flag_i_o = fn.split('_')[1]
        take = fn.split('_')[2]
        assert subject in SUBJ_OUTFIT_SEQ_HUMANCLOTHRECON_HUMANPARSING.keys()
        if take in SUBJ_OUTFIT_SEQ_HUMANCLOTHRECON_HUMANPARSING[subject][flag_i_o]:
            assert fn not in val_ids_all, f"Warning: {fn} is duplicated in val_ids_all"
            val_ids_all.append(fn)
        else:
            assert fn not in train_ids, f"Warning: {fn} is duplicated in train_ids"
            train_ids.append(fn)


print(f"train_ids: {len(train_ids)}")
print(f"val_ids_all: {len(val_ids_all)}")

val_ids_sampled = []
sample_ratio = 10

for subject in tqdm(SUBJ_OUTFIT_SEQ_HUMANCLOTHRECON_HUMANPARSING.keys()):
    for flag_i_o in ['Inner', 'Outer']:
        for take in SUBJ_OUTFIT_SEQ_HUMANCLOTHRECON_HUMANPARSING[subject][flag_i_o]:
            this_take_ids = sorted([fn for fn in val_ids_all if (fn.split('_')[0] == subject) and (fn.split('_')[1] == flag_i_o) and (fn.split('_')[2] == take)])
            for i in range(0, len(this_take_ids), sample_ratio):
                val_ids_sampled.append(this_take_ids[i])

print(f"val_ids_sampled: {len(val_ids_sampled)}")

with open(os.path.join(save_dir, "train_ids.pkl"), "wb") as f:
    pickle.dump(train_ids, f)
with open(os.path.join(save_dir, "val_ids_all.pkl"), "wb") as f:
    pickle.dump(val_ids_all, f)

with open(os.path.join(save_dir, "val_ids_sampled_ratio10.pkl"), "wb") as f:
    pickle.dump(val_ids_sampled, f)
