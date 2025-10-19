import torch
import numpy as np
import os
import json
import cv2
import wave
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skvideo.io
import ffmpeg
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import os.path as osp
import time
import subprocess
import argparse
import os.path as osp
from torch.nn.functional import normalize
from model import MMSC_model
from dp_narration_insertion import insert_narration

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument('--input_dim', type=int, default=1024, help='dimension of input feature')
    parser.add_argument('--output_dim', type=int, default=512, help='dimension of output feature')
    parser.add_argument('--num_layers', type=int, default=2, help='the number of Transformer Encoder layer')
    parser.add_argument('--heads', type=int, default=4, help='the number of attention head')
    parser.add_argument('--dropout', type=float, default=0.2)

    # test movie
    parser.add_argument("--test_movie_shot_embs", type=str, default="./dataset/test_dataset/test_movie_shot_embs/")
    parser.add_argument("--test_audio_shot_embs", type=str, default="./dataset/test_dataset/test_audio_shot_embs/")
    parser.add_argument("--test_labels_embs", type=str, default="./dataset/test_dataset/test_labels_embs/")
    parser.add_argument("--test_keywords_embs", type=str, default="./dataset/test_dataset/test_keywords_embs/")
    parser.add_argument("--music_mfcc_score", type=str, default="ouput path of music_mfcc_score.py")

    # control text input
    parser.add_argument("--label_exist", type=bool, default=True)
    parser.add_argument("--plot_exist", type=bool, default=True)

    args = parser.parse_args()
    return args


def inference(video_name):
    args = get_args()

    # GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = args.device

    # load model
    model = MMSC_model(output_dim=args.output_dim, num_layers=args.num_layers, heads=args.heads, dropout=args.dropout).to(device)
    model_net_path = 'saved model path'
    model.load_state_dict(torch.load(model_net_path))
    model.eval()

    with torch.no_grad():

        # test-8
        v_I = torch.Tensor(np.load(osp.join(args.test_movie_shot_embs, '{}.npy'.format(video_name)))).to(device)
        a_A = torch.Tensor(np.load(osp.join(args.test_audio_shot_embs, '{}.npy'.format(video_name)))).to(device)
        dl_I = torch.Tensor(np.load(osp.join(args.test_labels_embs, '{}.npy'.format(video_name)))).to(device)
        kw_I = torch.Tensor(np.load(osp.join(args.test_keywords_embs, '{}.npy'.format(video_name)))).to(device)
        energy_coeffient = torch.Tensor(np.load(osp.join(args.music_mfcc_score, '{}.npy'.format(video_name)))).to(device)

        # normalize embeddings
        v_I = normalize(v_I, p=2.0, dim=1)
        a_A = normalize(a_A, p=2.0, dim=1)
        label_I = normalize(dl_I, p=2.0, dim=1)
        plot_I = normalize(kw_I, p=2.0, dim=1)

        # predicted trailerness score and emotion score
        trailerness_pre, emotional_pre = model(v_I, a_A, label_I, plot_I, args.label_exist, args.plot_exist)

        # shot selection
        shot_l = 0
        shot_r = int(v_I.size(0))
        movie_shots_index = torch.arange(shot_l, shot_r).to(device)
        new_trailerness_pre = trailerness_pre[movie_shots_index].T[0]
        _, topk_idx = torch.topk(new_trailerness_pre, k=a_A.size(0))
        choose_idx = [movie_shots_index[i].item() for i in topk_idx]

        # shot sorting
        top_emotional_pre = emotional_pre[choose_idx].T[0]
        _, sorted_idx = torch.sort(top_emotional_pre, dim=0)
        _, gt_sorted_idx = torch.sort(energy_coeffient, dim=0)

        generated_sequence = [1] * a_A.size(0)
        for i in range(len(gt_sorted_idx)):
            generated_sequence[gt_sorted_idx[i]] = choose_idx[sorted_idx[i]]
        print('generated_sequence: {}'.format(generated_sequence))
        
    # return generated_sequence
    return generated_sequence

def time_to_seconds(time_string):
    time_components = time_string.split(':')

    hours = int(time_components[0])
    minutes = int(time_components[1])
    seconds = int(time_components[2].split('.')[0])
    milliseconds = int(time_components[2].split('.')[1])

    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds


def idx_to_seconds(fps, index): 
    index = int(index)
    return index / fps


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open this file, please check the file path.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def seconds_to_time(seconds):
    # input: float:  seconds
    # return: time_string hh:mm:ss.xxx
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = round((seconds - int(seconds)) * 1000)
    return "{:02d}:{:02d}:{:02d}.{}".format(int(hours), int(minutes), int(seconds), int(milliseconds))


def cal_movie_length(movie_info, fps):
    n = movie_info["shot_num"]
    ls = []
    for i in range(n): 
        n_frame = int(movie_info["shot_meta_list"][i]["frame"][1]) - int(movie_info["shot_meta_list"][i]["frame"][0]) 
        ls.append(idx_to_seconds(fps, n_frame))
    return ls

def cal_audio_length_rup(audio_info):
    # input is a list of second timestamp 
    ls = []
    last = 0.
    for item in audio_info:
        ls.append(item - last)
        last = item
    return ls


def get_duration_from_ffmpeg(filename):
    probe = ffmpeg.probe(filename)
    format = probe['format']
    duration = format['duration']
    size = int(format['size']) / 1024 / 1024
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found!')
        return
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])
    fps = int(video_stream['r_frame_rate'].split('/')[0]) / int(video_stream['r_frame_rate'].split('/')[1])
    duration = float(video_stream['duration'])
    return float(duration), int(num_frames)


def seg_video_by_timestamp(src_path, j, l_t, duration_t, save_seg_shots_base2):
    # movie_base = "./test_movies_8_320p/"
    # name = movidx#.split('-')[0]
    # src_path = osp.join(movie_base, name + '.mp4')

    tgt_path = osp.join(save_seg_shots_base2, str(j) + '.mp4')
    #cmd = 'ffmpeg -ss {} -i {} -t {} -c copy {} -y'.format(l_t, src_path, duration_t, tgt_path)
    cmd = 'ffmpeg -y -ss {} -i {}  -t {} -c:a aac -c:v libx264 {}'.format(l_t, src_path, duration_t, tgt_path)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # if result.returncode != 0:
    #     cmd = 'ffmpeg -y -ss {} -i {}  -t {} -c:a aac -c:v libx264 {}'.format(l_t, src_path, duration_t, tgt_path)
    #     os.system(cmd)
    print('movie {}: {}-th shot segmented done.'.format(movidx, str(j)))


def add_official_audio(trailer_path, input_video_path, output_video_path): 
    # 1. extract audio(music) from trailer, save it to temp_audio path 
    temp_audio = "temp_audio.aac"
    extract_audio_cmd = f"ffmpeg -i {trailer_path} -q:a 0 -map a {temp_audio} -y"
    subprocess.run(extract_audio_cmd, shell=True, check=True)

    # 2. add audio file to input_video 
    add_audio_cmd = f"ffmpeg -i {input_video_path} -i {temp_audio} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {output_video_path} -y"
    subprocess.run(add_audio_cmd, shell=True, check=True)

    print('Insert trailer\'s audio into silent generated trailer successfully.')
    subprocess.run(f"rm -f {temp_audio}", shell=True)


def video_seg_concat_by_shot_list(movidx, select_shot_list):
    # 1. Depend on the one-to-one alignment, find the right corresponding timestamps for each audio shot,
    # 2. segment raw movie based on timestamps, derive multiple video shots 
    # 3. add fade in and fade out for each shot 
    # 4. concate all video shots. 
    
    print('start trailer generation!')
    s_time = time.time()
    movidx = str(movidx)
     # save the segmented shots based on audio
    save_seg_shots_base = "./output-seg-shots/"
    # save the segmented shots based on audio (fade in and fade out)
    save_seg_shots_fade_base = "./output-seg-shots-fade/"
    output_video_base = "./output-videos/"
    save_seg_shots_base2 = osp.join(save_seg_shots_base, movidx)
    os.makedirs(save_seg_shots_base2, exist_ok=True)

    scene_movie_base = "./test_movies_8_320_transnetv2_info_json"
    scene_movie_info_path = osp.join(scene_movie_base, movidx + '.json')
    with open(scene_movie_info_path, 'r') as f:
        scene_movie_info = json.load(f)

    bar_seg_info_path = f"./ruptures_audio_segmentation_test_MT_2s.json"
    with open(bar_seg_info_path, 'r') as f:
        bar_seg_info = json.load(f)[movidx]
    
    #movie_base = "./test_movies_8_320p_intra/"
    movie_base = "./test_movies_intra"
    movie_path = osp.join(movie_base, movidx + '.mp4')
    m_fps = get_video_fps(movie_path)

    print(f'movie fps:{m_fps}')

    # trailer_base = "./test_trailers_8_320p/"
    # trailer_path = osp.join(trailer_base, movidx + '.mp4')
    # t_fps = get_video_fps(trailer_path)

    # print(f'trailer fps:{t_fps}')

    # calculate each shot's duration, each bar's duration 
    shot_length = cal_movie_length(scene_movie_info, m_fps)
    audio_length = cal_audio_length_rup(bar_seg_info)

    trailerness_score_base = './save_trailerness_predict' 
    trailerness_score_path = os.path.join(trailerness_score_base, f'{movidx}.npy')
    sig_score = np.load(trailerness_score_path)

    # 1. Depend on the one-to-one alignment, find the right corresponding timestamps for each audio shot,
    # 2. segment raw movie based on timestamps, derive multiple video shots 
    J = len(audio_length)
    I = len(shot_length)

    for j in range(J):
        i = select_shot_list[j]
        if shot_length[i] >= audio_length[j]:
            # expand from the mid time of the shot 
            t0 = scene_movie_info["shot_meta_list"][i]["frame"][0]
            t1 = scene_movie_info["shot_meta_list"][i]["frame"][1]
            t0_seconds = idx_to_seconds(m_fps, t0)
            t1_seconds = idx_to_seconds(m_fps, t1)
            mid_seconds = (t0_seconds + t1_seconds) / 2.0
            l_seconds = mid_seconds - audio_length[j] / 2.0
            # r_seconds = mid_seconds + audio_length[j]/2.0
            l_t = seconds_to_time(l_seconds)
            duration = audio_length[j]
            duration_t = seconds_to_time(duration)
            #print(f'{j}\t\t type1 , {l_t}, frames:{duration_t}')
            seg_video_by_timestamp(movie_path, j, l_t, duration_t, save_seg_shots_base2)
        else:
            difference = audio_length[j] - shot_length[i]
            l = i
            r = i
            t0 = scene_movie_info["shot_meta_list"][i]["frame"][0]
            t1 = scene_movie_info["shot_meta_list"][i]["frame"][1]
            t0_seconds = idx_to_seconds(m_fps, t0)
            t1_seconds = idx_to_seconds(m_fps, t1)
            left_seconds = t0_seconds
            while difference:
                if l > 0 and r < I - 1:
                    if sig_score[l - 1] >= sig_score[r + 1]:
                        if shot_length[l - 1] >= difference:
                            t_r = scene_movie_info["shot_meta_list"][l - 1]["frame"][1]
                            t_r_seconds = idx_to_seconds(m_fps, t_r)
                            left_seconds = t_r_seconds - difference
                            difference = 0
                        else:
                            difference -= shot_length[l - 1]
                        l = l - 1
                    else:
                        if shot_length[r + 1] >= difference:
                            difference = 0
                        else:
                            difference -= shot_length[r + 1]
                        r = r + 1
                elif r < I - 1:
                    if shot_length[r + 1] >= difference:
                        difference = 0
                    else:
                        difference -= shot_length[r + 1]
                    r = r + 1
                elif l > 0:
                    t_r = scene_movie_info["shot_meta_list"][l - 1]["frame"][1]
                    t_r_seconds = idx_to_seconds(m_fps, t_r)
                    left_seconds = t_r_seconds - difference
                    difference = 0
                    l = l - 1
            # now: seg from left_seconds, duration = audio_length[j]
            l_t = seconds_to_time(left_seconds)
            duration = audio_length[j]
            duration_t = seconds_to_time(duration)
            #print(f'{j}\t\t type2, {l_t}, frames:{duration_t}')
            seg_video_by_timestamp(movie_path, j, l_t, duration_t, save_seg_shots_base2)

    print('begin deal with fade in and fade out.')
    
    
    # 3. add fade in and fade out for each shot 
    output_file = 'output-test-' + str(movidx)  + '-adjust_duration.mp4'
    tmp_video_file = 'output-test-' + str(movidx) + '-adjust_duration-silent.mp4'
    shot_base = os.path.join(save_seg_shots_base, str(movidx))
    shot_idx_list = range(J)
    tmp_txt_file_path = './tmp{}.txt'.format(movidx)

    # # make each shot in shot_idx_list in style of fade in and out
    shot_tmp_base = os.path.join(save_seg_shots_fade_base, str(movidx))
    os.makedirs(shot_tmp_base, exist_ok=True)
    for shot_idx in shot_idx_list:
        shot_file_name = "{}.mp4".format(shot_idx)
        path_in = os.path.join(shot_base, shot_file_name)
        path_out = os.path.join(shot_tmp_base, shot_file_name)
        duration, nframe = get_duration_from_ffmpeg(path_in)
        if nframe > 60:  # 
            cmd = 'ffmpeg -i "{}" -vf "fade=in:0:10,fade=out:{}:8" "{}" -y'.format(path_in, int(nframe - 8), path_out)
        else:
            cmd = 'cp {} {}'.format(path_in, path_out)
        os.system(cmd)

    # 4. concate all video shots. 
    # write the required txt file 

    print('begin to concate all video shots.')

    with open(tmp_txt_file_path, 'w') as f:
        for shot_idx in shot_idx_list:
            shot_file = "{}.mp4".format(shot_idx)
            shot_file_path = os.path.join(shot_tmp_base, shot_file)
            write_line = "file '{}'\n".format(shot_file_path)
            f.writelines(write_line)

    output_file_path = os.path.join(output_video_base, output_file)
    tmp_video_file_path = os.path.join(output_video_base, tmp_video_file)

    # generate initial trailer based on the movie shots concatenation
    cmd1 = 'ffmpeg -f concat -safe 0 -i {} -c copy {} -y'.format(tmp_txt_file_path, output_file_path)
    os.system(cmd1)

    # remove the audio in the generated trailer (optional)
    cmd2 = 'ffmpeg -i {} -c:v copy -an {} -y'.format(output_file_path, tmp_video_file_path)
    os.system(cmd2)

    mt_trailer_base = './test_trailer_MT'
    mt_trailer_path = os.path.join(mt_trailer_base, f'{movidx}.mp4')
    output_file_audio = 'output-test-' + str(movidx) + '-adjust_duration-withaudio-1080.mp4'
    output_file_audio_path = os.path.join(output_video_base, output_file_audio)
    add_official_audio(mt_trailer_path, tmp_video_file_path, output_file_audio_path)
    
    e_time = time.time()
    print(f'all cost {e_time-s_time}s.')


def video_seg_concat_directly_by_shot_list(movidx, select_shot_list):
    # 1. Depend on the one-to-one alignment, find the right corresponding timestamps for each audio shot,
    # 2. segment raw movie based on timestamps, derive multiple video shots 
    # 3. add fade in and fade out for each shot 
    # 4. concate all video shots. 
    
    print('start trailer generation!')
    s_time = time.time()
    movidx = str(movidx)
     # save the segmented shots based on audio
    save_seg_shots_base = "./output-seg-shots/"
    # save the segmented shots based on audio (fade in and fade out)
    save_seg_shots_fade_base = "./output-seg-shots-fade/"
    output_video_base = "./output-videos/"
    save_seg_shots_base2 = osp.join(save_seg_shots_base, movidx)
    os.makedirs(save_seg_shots_base2, exist_ok=True)

    scene_movie_base = "./test_movies_8_320_transnetv2_info_json"
    scene_movie_info_path = osp.join(scene_movie_base, movidx + '.json')
    with open(scene_movie_info_path, 'r') as f:
        scene_movie_info = json.load(f)

    bar_seg_info_path = f"./ruptures_audio_segmentation_test_MT_2s.json"
    with open(bar_seg_info_path, 'r') as f:
        bar_seg_info = json.load(f)[movidx]

    movie_base = "./test_movies_8_320p_intra/"
    #movie_base = "./test_movies_intra"
    movie_path = osp.join(movie_base, movidx + '.mp4')
    m_fps = get_video_fps(movie_path)

    print(f'movie fps:{m_fps}')

    trailer_base = "./test_trailers_8_320p/"
    trailer_path = osp.join(trailer_base, movidx + '.mp4')
    t_fps = get_video_fps(trailer_path)

    print(f'trailer fps:{t_fps}')

    # calculate each shot's duration, each bar's duration 
    shot_length = cal_movie_length(scene_movie_info, m_fps)
    audio_length = cal_audio_length_rup(bar_seg_info)

    # 1. Depend on the one-to-one alignment, find the right corresponding timestamps for each audio shot,
    # 2. segment raw movie based on timestamps, derive multiple video shots 
    J = len(audio_length)
    I = len(shot_length)

    for j in range(J): 
        i = select_shot_list[j]
        t0 = scene_movie_info["shot_meta_list"][i]["frame"][0]
        t0_seconds = idx_to_seconds(m_fps, t0)
        t1 = scene_movie_info["shot_meta_list"][i]["frame"][1]
        t1_seconds = idx_to_seconds(m_fps, t1)
        t0_t = seconds_to_time(t0_seconds)
        duration = idx_to_seconds(m_fps, int(t1) - int(t0))
        duration_t = seconds_to_time(duration)
        seg_video_by_timestamp(movie_path, j, t0_t, duration_t, save_seg_shots_base2)

    print('begin deal with fade in and fade out.')
    
    # 3. add fade in and fade out for each shot 
    output_file = 'output-test-' + str(movidx) + '-woadjust.mp4'
    tmp_video_file = 'output-test-' + str(movidx) + '-woadjust-silent.mp4'
    shot_base = os.path.join(save_seg_shots_base, str(movidx))
    shot_idx_list = range(J)
    tmp_txt_file_path = './tmp{}.txt'.format(movidx)

    # # make each shot in shot_idx_list in style of fade in and out
    shot_tmp_base = os.path.join(save_seg_shots_fade_base, str(movidx))
    os.makedirs(shot_tmp_base, exist_ok=True)
    for shot_idx in shot_idx_list:
        shot_file_name = "{}.mp4".format(shot_idx)
        path_in = os.path.join(shot_base, shot_file_name)
        path_out = os.path.join(shot_tmp_base, shot_file_name)
        duration, nframe = get_duration_from_ffmpeg(path_in)
        if nframe > 60:  # 
            cmd = 'ffmpeg -i "{}" -vf "fade=in:0:10,fade=out:{}:8" "{}" -y'.format(path_in, int(nframe - 8), path_out)
        else:
            cmd = 'cp {} {}'.format(path_in, path_out)
        os.system(cmd)

    # 4. concate all video shots. 
    # write the required txt file 

    print('begin to concate all video shots.')

    with open(tmp_txt_file_path, 'w') as f:
        for shot_idx in shot_idx_list:
            shot_file = "{}.mp4".format(shot_idx)
            shot_file_path = os.path.join(shot_tmp_base, shot_file)
            write_line = "file '{}'\n".format(shot_file_path)
            f.writelines(write_line)

    output_file_path = os.path.join(output_video_base, output_file)
    tmp_video_file_path = os.path.join(output_video_base, tmp_video_file)

    # generate initial trailer based on the movie shots concatenation
    cmd1 = 'ffmpeg -f concat -safe 0 -i {} -c copy {} -y'.format(tmp_txt_file_path, output_file_path)
    os.system(cmd1)

    # remove the audio in the generated trailer (optional)
    cmd2 = 'ffmpeg -i {} -c:v copy -an {} -y'.format(output_file_path, tmp_video_file_path)
    os.system(cmd2)
    
    # cmd3 = f'rm {output_file_path}'
    # os.system(cmd3)

    e_time = time.time()
    print(f'all cost {e_time-s_time}s.')


if __name__ == "__main__":
    print("Inference start!")
    video_name = 'movie name under "./dataset/test_dataset/test_movie_shot_embs/" folder'
    generated_sequence = inference(video_name)
    video_seg_concat_directly_by_shot_list(video_name, generated_sequence)
    insert_narration(generated_sequence)
