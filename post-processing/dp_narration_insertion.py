import argparse
import numpy as np
from moviepy.editor import VideoFileClip
import os
import scipy.io.wavfile as wavfile


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_selected_narration_embs", type=str, default="ouput selected narration of deepseek_narration_selection.py, using ImageBind to extract selected narration embeddings, npy format")
    parser.add_argument("--trailer_shot_mini_caption_embs", type=str, default="ouput trailer shot caption of mini_caption_generation.py, using ImageBind to extract selected narration embeddings, npy format")
    parser.add_argument("--narration_shot", type=str, default="output selected narration of deepseek_narration_selection.py")

    args = parser.parse_args()
    return args


def get_wav_duration_scipy(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path)
    num_samples = len(audio_data)
    duration = num_samples / sample_rate
    return duration


def minimize_placement_cost_no_overlap(M, N, volumes, capacities, costs):
    """
    Args:
        M (int): number of selected narration
        N (int): number of music shots
        volumes (list[int]): duration of each selected narration
        capacities (list[int]): duration of each music shot
        similarity (list[list[int]]): sim[i][j] the similarity between trailer shot caption and selected narration

    Returns:
        tuple: (max similarity, start timestamp of each selected narration)
    """
    # initialize DP table
    dp = [[float('inf')] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0

    position_choice = [[-1] * (N + 1) for _ in range(M + 1)]

    prefix_capacity = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_capacity[i] = prefix_capacity[i - 1] + capacities[i - 1]

    # DP iterations
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            for k in range(j):
                volume_sum = prefix_capacity[j] - prefix_capacity[k]
                if volume_sum >= volumes[i - 1]:
                    cost = dp[i - 1][k] + costs[i - 1][k]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        position_choice[i][j] = k

    positions = [-1] * M
    current_j = N
    for i in range(M, 0, -1):
        start = position_choice[i][current_j]
        positions[i - 1] = start
        current_j = start

    return dp[M][N], positions

def insert_narration(generated_trailer):
    args = get_args()

    # test-1
    narration_embs = np.load(args.llm_selected_narration_embs)
    caption_embs = np.load(args.trailer_shot_mini_caption_embs)

    # narration duration
    narration_list = os.listdir(args.narration_shot)
    sorted_narration_list = sorted(narration_list, key=lambda x: int(str(x)[:-4]))
    narration_duration = []
    for narration_idx in sorted_narration_list:
        shot_path = os.path.join(args.narration_shot, narration_idx)
        narration_duration.append(get_wav_duration_scipy(shot_path))

    # music duration
    music_list = os.listdir(args.music_shot)
    sorted_music_list = sorted(music_list, key=lambda x: int(str(x)[:-4]))
    music_duration = []
    for music_idx in sorted_music_list:
        shot_path = os.path.join(args.music_shot, music_idx)
        music_duration.append(get_wav_duration_scipy(shot_path))
    
    # distance
    distances = []
    narration_rows = narration_embs.shape[0]
    for i in range(narration_rows):
        narration_row = narration_embs[i]
        narration_row = np.expand_dims(narration_row, 0)
        distance = np.sqrt(np.sum((caption_embs - narration_row) ** 2, axis=1)).tolist()
        distances.append(distance)

    # DP
    M = narration_embs.shape[0]
    N = caption_embs.shape[0]
    total_cost, positions = minimize_placement_cost_no_overlap(M, N, narration_duration, music_duration, distances)
    insert_index = [generated_trailer[i] for i in positions]
    print(insert_index)

    # insert timestamp
    insert_timestamp = []
    for i in positions:
        insert_timestamp.append(sum(music_duration[:i+1]))
    print('insert timestamp: {}'.format(insert_timestamp))