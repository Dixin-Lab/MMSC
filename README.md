# [IJCAI 2025] Weakly-Supervised Movie Trailer Generation Driven by Multi-Modal Semantic Consistency

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
