import argparse
import os
import os.path as osp
from torch.nn.functional import normalize
import torch
import numpy as np
from model import MMSC_model


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


# LD metric
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# AA metric
def video_pairwise_agreement_accuracy(predicted_ranking, true_ranking):
    max_length = max(len(predicted_ranking), len(true_ranking))
    
    predicted_ranking += [0] * (max_length - len(predicted_ranking))
    true_ranking += [0] * (max_length - len(true_ranking))
    
    correct_predictions = 0
    total_pairs = 0
    
    for i in range(max_length):
        for j in range(i + 1, max_length):
            total_pairs += 1
            if (predicted_ranking[i] < predicted_ranking[j] and true_ranking[i] < true_ranking[j]) or \
               (predicted_ranking[i] == predicted_ranking[j] and true_ranking[i] == true_ranking[j]) or \
               (predicted_ranking[i] > predicted_ranking[j] and true_ranking[i] > true_ranking[j]):
                correct_predictions += 1
                
    accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0
    return accuracy



def inference(video_num):
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
        v_I = torch.Tensor(np.load(osp.join(args.test_movie_shot_embs, '{}.npy'.format(video_num)))).to(device)
        a_A = torch.Tensor(np.load(osp.join(args.test_audio_shot_embs, '{}.npy'.format(video_num)))).to(device)
        dl_I = torch.Tensor(np.load(osp.join(args.test_labels_embs, '{}.npy'.format(video_num)))).to(device)
        kw_I = torch.Tensor(np.load(osp.join(args.test_keywords_embs, '{}.npy'.format(video_num)))).to(device)
        energy_coeffient = torch.Tensor(np.load(osp.join(args.music_mfcc_score, '{}.npy'.format(video_num)))).to(device)

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



if __name__ == "__main__":
    print("Inference start!")
    inference()
