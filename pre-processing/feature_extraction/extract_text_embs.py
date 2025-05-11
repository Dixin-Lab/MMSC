from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import os
import math
import time
import numpy as np
import os.path as osp
import shutil
import json
from nltk.tokenize import sent_tokenize
import nltk
import warnings
warnings.filterwarnings('ignore')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# =================================== label extraction ==================================
# Open metadata file
metadata_info_path = 'Movie labels path'

with open(metadata_info_path, 'r', encoding='utf-8') as f:
    metadata_info = json.load(f)

# Create label list of each movie
for movieid, value in metadata_info.items():
    print("Meta data of movie {} is processing.".format(movieid))

    # label_list = value["label"]
    label_one_sentence = ', '.join(value["label"])
    label_list = [label_one_sentence]
    print(len(label_list))

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(label_list, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    meta_label_emb = embeddings[ModalityType.TEXT]
    print(meta_label_emb.size())
    np_meta_label = meta_label_emb.detach().cpu().numpy()

    v_save_path = './movie_label_embs/{}.npy'.format(movieid)
    np.save(v_save_path, np_meta_label)

print("All movie label features are extracted.")

# ====================================== plot extraction ===================================
# Open metadata file
metadata_info_path = 'Movie plot keywords path'
with open(metadata_info_path, 'r', encoding='utf-8') as f:
    metadata_info = json.load(f)

# Create label list of each movie
for movieid, value in metadata_info.items():
    print("Meta data of movie {} is processing.".format(movieid))

    # label_list = sent_tokenize(value[0])
    label_list = value
    print(len(label_list))

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(label_list, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    meta_label_emb = embeddings[ModalityType.TEXT]
    print(meta_label_emb.size())
    np_meta_label = meta_label_emb.detach().cpu().numpy()

    v_save_path = './movie_keywords_embs/{}.npy'.format(movieid)
    np.save(v_save_path, np_meta_label)

print("All movie plot keywords features are extracted.")