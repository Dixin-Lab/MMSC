import os
import os.path as osp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


trailer_path_base = 'music segments path'
trailer_paths = os.listdir(trailer_path_base)
trailer_paths_sorted = sorted(trailer_paths, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))


for trailer in trailer_paths_sorted:
    print('trailer {}'.format(trailer))
    audio_path_base = osp.join(trailer_path_base, '{}'.format(trailer))
    audio_paths = os.listdir(audio_path_base)
    audios = sorted(audio_paths, key=lambda x: (int(x[:-4])))
    arrays = []

    for audio in audios:
        audio_path = osp.join(audio_path_base, '{}'.format(audio))
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        arrays.append(np.mean(mfccs[0]))

    arrays_np = np.array(arrays)
    print('{}'.format(arrays_np.shape))

    arr_min = min(arrays_np)
    arr_max = max(arrays_np)

    # min-max normalization
    arr_normalized = (arrays_np - arr_min) / (arr_max - arr_min)
    print('arr_normalized: {}'.format(arr_normalized))
    print('{}'.format(arr_normalized.shape))
    save_path = './emotion_score_{}.npy'.format(trailer)
    np.save(save_path, arr_normalized)