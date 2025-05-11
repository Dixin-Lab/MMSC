import argparse
import os
import os.path as osp
import torch
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default='0')

    # movie-trailer dataset
    parser.add_argument("--movie_shot_embs", type=str, default="movie shot embedding path, npy format")
    parser.add_argument("--trailer_shot_embs", type=str, default="trailer shot embedding path, npy format")

    args = parser.parse_args()
    return args



def similarity():
    args = get_args()

    # GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = args.device

    # dataset & dataloader
    movie_videos = os.listdir(args.trailer_shot_embs)

    save_path = 'trailerness pseudo-score save path'
    os.makedirs(save_path, exist_ok=True)

    save_path_all = 'movie trailer similarity save path'
    os.makedirs(save_path_all, exist_ok=True)


    # train movie-200
    for video in movie_videos:
        trailer_idx = video[:-4]
        movie_idx = trailer_idx.split('-')[0]

        movie_shot = torch.Tensor(np.load(osp.join(args.movie_shot_embs, '{}.npy'.format(movie_idx)))).to(device)
        trailer_shot = torch.Tensor(np.load(osp.join(args.trailer_shot_embs, '{}.npy'.format(movie_idx)))).to(device)

        dot_product = torch.matmul(movie_shot, trailer_shot.T)
        norm_A = torch.norm(movie_shot, p=2, dim=1, keepdim=True)
        norm_B = torch.norm(trailer_shot, p=2, dim=1, keepdim=True)
        norm_B = norm_B.T
        cosine_similarity = dot_product / (norm_A * norm_B)
        max_values, max_indices = torch.max(cosine_similarity, dim=1)
        max_ = max_values.max()
        min_ = max_values.min()
        max_values = (max_values - min_) / (max_ - min_)

        # calculate movie-trailer similarity
        cosine_similarity_exp = torch.exp(cosine_similarity)
        for i in range(cosine_similarity_exp.size(0)):
            row = cosine_similarity_exp[i]
            max_val = torch.max(row)
            min_val = torch.min(row)
            normalized_row = (row - min_val) / (max_val - min_val)
            cosine_similarity_exp[i] = normalized_row

        # save trailerness pseudo-score
        save_file_path = osp.join(save_path, '{}.npy'.format(trailer_idx))
        max_values_np = max_values.detach().cpu().numpy()
        np.save(save_file_path, max_values_np)

        # save movie-trailer similarity
        save_file_path_all = osp.join(save_path_all, '{}.npy'.format(trailer_idx))
        cosine_similarity_exp_np = cosine_similarity_exp.detach().cpu().numpy()
        np.save(save_file_path_all, cosine_similarity_exp_np)


if __name__ == "__main__":
    print('*' * 100)
    print("Training start!")
    print('*' * 100)
    similarity()
