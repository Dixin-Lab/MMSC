import argparse
import os
import os.path as osp
import torch
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default='4')

    # movie-trailer dataset
    parser.add_argument("--movie_shot_embs", type=str, default="...")
    parser.add_argument("--trailer_shot_embs", type=str, default="...")
    parser.add_argument("--movie_trailer_similarity", type=str, default="...")
    parser.add_argument("--music_energy_coefficient", type=str, default="...")
    
    args = parser.parse_args()
    return args


def emotion():
    args = get_args()

    # GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = args.device

    # dataset & dataloader
    movie_videos = os.listdir(args.trailer_shot_embs)

    save_path = 'emotion pseudo-score save path'
    os.makedirs(save_path, exist_ok=True)


    # train movie-200
    for video in movie_videos:
        trailer_idx = video[:-4]

        similarity_matrix = torch.Tensor(np.load(osp.join(args.movie_trailer_similarity, '{}.npy'.format(trailer_idx)))).to(device)
        coefficient_matrix = torch.Tensor(np.load(osp.join(args.music_energy_coefficient, '{}.npy'.format(trailer_idx)))).to(device)

        # filter similarity matrix ()
        filter_matrix = torch.zeros_like(similarity_matrix)
        for i in range(similarity_matrix.size(0)):
            max_val, _ = torch.max(similarity_matrix[i], dim=0)
            max_indices = (similarity_matrix[i] == max_val).nonzero(as_tuple=True)[0]
            for index in max_indices:
                filter_matrix[i, index] = 1 / len(max_indices)

        # save emotion pseudo-score
        emotion_score = torch.matmul(filter_matrix, coefficient_matrix)
        save_file_path = osp.join(save_path, '{}.npy'.format(trailer_idx))
        emotion_score_np = emotion_score.detach().cpu().numpy()
        np.save(save_file_path, emotion_score_np)


if __name__ == "__main__":
    print("Emotion pseudo-score calculation start!")
    emotion()