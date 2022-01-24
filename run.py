# %%
import torch
from model import MattingNetwork
from inference import convert_video

model = MattingNetwork('resnet50').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_resnet50.pth'))

vid_name = 'anomaly'
convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    input_source='data/{vid_name}.mp4',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='data/segmented/{vid_name}_seg.mp4',    # File path if video; directory path if png sequence.
    output_alpha="data/segmented/{vid_name}_pha.mp4",          # [Optional] Output the raw alpha prediction.
    output_foreground="data/segmented/{vid_name}_fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
)
