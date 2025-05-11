 [IJCAI 2025] Weakly-Supervised Movie Trailer Generation Driven by Multi-Modal Semantic Consistency

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://github.com/Dixin-Lab/MMSC" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href='[https://www.bilibili.com/video/BV15sWMeAE8R/?spm_id_from=333.999.0.0&vd_source=4526cf207f29ce6d50810b04d3105cfd](https://space.bilibili.com/487967491/lists/4641072?type=series)' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-bilibili-pink.svg' alt='demo'>
  </a>
<!--   <a href="https://github.com/Zheng-Chong/CatVTON/LICENCE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a> -->
</div>

**TL;DR**: Given a raw video, a piece of music, video metadata (i.e., video plot keywords and category labels), and video subtitles, we can generate an appealing video trailer/montage with narration. 

![scheme](fig/framework.png)

## ⏳ Project Structure
```
.
├── dataset
|   ├── training_dataset
|   |   ├── train_audio_shot_embs (npy format, segmented audio shots)
|   |   ├── train_movie_shot_embs (npy format, segmented movie shots)
|   |   ├── train_trailer_shot_embs (npy format, segmented trailer shots)
|   |   ├── train_labels_embs (npy format, movie category labels)
|   |   ├── train_keywords_embs (npy format, movie plot keywords)
|   |   ├── train_trailerness_score (npy format, processed trailerness score of each movie shot)
|   |   └── train_emotion_score (npy format, processed emotion score of each movie shot)
|   └── test_dataset
|       ├── test_audio_shot_embs (npy format, segmented audio shots)
|       ├── test_movie_shot_embs (npy format, segmented movie shots)
|       ├── test_trailer_shot_embs (npy format, segmented trailer shots)
|       ├── test_labels_embs (npy format, movie category labels)
|       ├── test_keywords_embs (npy format, movie plot keywords)
|       ├── test_trailerness_score (npy format, processed trailerness score of each movie shot)
|       └── test_emotion_score (npy format, processed emotion score of each movie shot)
|—— checkpoint
|   └── network_1500.net
├── pesudo_score_calculation
|   ├── trailerness_pesudo_score.py
|   └── emotion_pesudo_score.py
|—— model.py
|—— inference.py
|—— post-processing
|   ├── movie_shot_duration_adjustment.py
|   ├── deepseek_narration_selection.py
|   └── dp_narration_insertion.py
├── feature_extratction
├── shot_segmentation
└── utils
```
## ⚙️ Main Dependencies
- python=3.8.19
- pytorch=2.3.0+cu121
- numpy=1.24.1
- matplotlib=3.7.5
- scikit-learn=1.3.2
- scipy=1.10.1
- sk-video=1.1.10
- ffmpeg=1.4

Or create the environment by:
```commandline 
pip install -r requirements.txt
```

## 🎞 Dataset
###  Dataset structure
We expand CMTD dataset from 200 movies to 500 movies for movie trailer generation and future video understanding tasks. We train and evaluate various trailer generators on this dataset. Please download the new dataset from these links: [MMSC_DATASET](https://drive.google.com/drive/folders/1Iw6OXMi6_nyFyvyK5hXb_aYwRTcg7oHj?usp=drive_link). Compared with CMTD dataset, MMSC dataset contains extrated movie category labels embeddings, movie plot keywords embeddings, processed movie trailerness scores, and processed movie emotion scores.
It is worth noting that due to movie copyright issues, we cannot provide the original movies. The dataset only provides the visual and acoustic features extracted by ImageBind after we segmented the movie shot and audio shot using TransNet V2.

### Model ckpt
We provide the trained model ```network_1500.net``` under the checkpoint folder.

### Movie Shot Segmentation 
We use [TransNet V2]([https://github.com/kakaobrain/bassl](https://github.com/soCzech/TransNetV2)), a shot transition detection model, to split each movie into movie shots, the codes can be found in ```./segmentation/scene_segmentation_transnetv2.py```. 
If you want to perform shot segmentation on your local video, please be aware of modifying the path for reading the video and the path for saving the segmentation results in the code.

```commandline
movie_dataset_base = '' # video data directory
movies = os.listdir(movie_dataset_base)

save_scene_dir_base = '' # save directory of scene json files 
finished_files = os.listdir(save_scene_dir_base)
```

### Segment audio based on movie shots
During the training phase, in order to obtain aligned movie shots and audio shots from each official trailer, we segment the official trailer audio according to the duration of the movie shots.
The codes can be found in ```./segmentation/seg_audio_based_on_shots.py```. 
If you want to perform audio segmentation based on your movies shot segmentation, please be aware of modifying the path for reading the audio and the path for saving the segmentation results in the code.

```commandline
seg_json = dict()  # save the segmentation info of audio 
base = ''
save_seg_json_name = 'xxx.json'
save_bar_base = ""
scene_trailer_base = ""
audio_base = ""
```

### Music Shot Segmentation 
We use [Ruptures](https://github.com/deepcharles/ruptures) to split music into movie shots and scenes, the codes can be found in ```./segmentation/scene_segmentation_ruptures.py```. 
If you want to perform shot segmentation on your local audio, please be aware of modifying the path for reading the audio and the path for saving the segmentation results in the code.

```commandline
audio_file_path = ''  # music data path
save_result_base = ''  # save segmentation result
```
During testing phase, given a movie and a piece of music, we use BaSSL to segment the movie shots and Ruptures to segment the music shots.


### Feature Extraction
We use [ImageBind](https://github.com/facebookresearch/ImageBind) to extract visual features of movie shots and acoustic features of audio shots, the codes can be found in ```./feature_extraction/```. 
