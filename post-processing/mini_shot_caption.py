import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
import os
import json

model = AutoModel.from_pretrained('MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('MiniCPM-V-2_6', trust_remote_code=True)

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames


movie = 'Movie name'
selected_trailer_shots = 'The path where all trailer shots selected by the model are saved'
output_path = './mini_caption/'

shots = os.listdir(selected_trailer_shots)
sorted_shots = sorted(shots, key=lambda x: int(x[:-4]))

output_json = dict()

for shot in sorted_shots:

    shot_path= os.path.join(selected_trailer_shots, shot)
    frames = encode_video(shot_path)
    question = "Please provide a single-sentence description that captures both the objects present and the events occurring within the video"
    msgs = [
        {'role': 'user', 'content': frames + [question]},
    ]

    # Set decode params for video
    params = {}
    params["use_image_id"] = False
    params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution > 448*448
    
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
       **params
    )
    print(answer)

    output_json[shot[:-4]] = answer

with open(os.path.join(output_path, '{}.json'.format(movie)), 'w') as f:
    json.dump(output_json, f, indent=4)