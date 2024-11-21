import cv2
import os
import numpy as np
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from TennisV3.downscale import get_original_frame_index
from TennisV3.test import predict_location
from TennisV3.dataset import Video_IterableDataset
from TennisV3.utils.general import *
from TennisV3.bounce_detector import BounceDetector
from TennisV3.court_detection_net import CourtDetectorNet
from TennisV3.draw_court import draw_ball_trace

def extract_frames(video_path, center_index, total_frames=60):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_frames = total_frames // 2
    start_index = max(0, center_index - half_frames)
    end_index = min(frame_count, center_index + half_frames)
    frames = []

    for i in range(start_index, end_index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    print(f"提取了 {len(frames)} 帧，从 {start_index} 到 {end_index - 1}.")
    return frames

def frames_to_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"视频已保存到 {output_path}")

def predict_ball_trajectory(indices, y_pred, img_scaler):
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Bounce': []}
    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices
    y_pred = (y_pred > 0.5).detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    y_pred = to_img_format(y_pred)  # (N, L, H, W)
    ball_track = []
    prev_f_i = -1

    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                y_p = y_pred[n][f]
                bbox_pred = predict_location(to_img(y_p))
                cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)
                cx_pred, cy_pred = int(cx_pred * img_scaler[0]), int(cy_pred * img_scaler[1])
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                ball_track.append((cx_pred, cy_pred))

                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                pred_dict['Bounce'].append(0)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict, ball_track

def save_bounce_frames(bounces, cap, video_name, ball_track, output_dir, court_detector):
    bounce_frames_dir = os.path.join(output_dir, 'bounce_frames')
    os.makedirs(bounce_frames_dir, exist_ok=True)

    for frame_num in bounces:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(bounce_frames_dir, f'{video_name}_bounce_frame_{frame_num}.png')
            cv2.imwrite(frame_path, frame)
            draw_ball_trace(frame_path, ball_track, frame_num, trace_len=7, output_dir='court_detect', court_detector=court_detector)
    print("反弹帧已保存。")

def run_inference(
    ori_video_file, tracknet_file, batch_size, max_sample_num, video_range,
    save_dir, seg_video_index, ori_video_fps, segment_duration, segment_video_frame_index,
    path_bounce_model, path_court_model):

    os.makedirs(save_dir, exist_ok=True)

    downsample_rate = ori_video_fps // 15
    frame_index = get_original_frame_index(seg_video_index, segment_video_frame_index,
                                           ori_video_fps, segment_duration, downsample_rate)
    frame_list = extract_frames(ori_video_file, frame_index)
    output_video_path = os.path.join(save_dir, f'centered_segment_{seg_video_index}_frame_{segment_video_frame_index}.mp4')
    frames_to_video(frame_list, output_video_path, fps=ori_video_fps)

    video_file = output_video_path
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_csv_file = os.path.join(save_dir, f'{video_name}.csv')

    # 加载 TrackNet 模型
    tracknet_ckpt = torch.load(tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])
    tracknet.eval()

    cap = cv2.VideoCapture(video_file)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_scaler = (w / WIDTH, h / HEIGHT)

    dataset = Video_IterableDataset(video_file, seq_len=tracknet_seq_len, sliding_step=3,
                                    bg_mode=bg_mode, max_sample_num=max_sample_num, video_range=video_range)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    bounce_detector = BounceDetector(path_bounce_model)
    tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Bounce': []}
    ball_track = []

    for i, x in tqdm(data_loader):
        x = x.float().cuda()
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()
        tmp_pred, tmp_ball_track = predict_ball_trajectory(i, y_pred, img_scaler)
        ball_track.extend(tmp_ball_track)
        for key in tmp_pred.keys():
            tracknet_pred_dict[key].extend(tmp_pred[key])

    x_ball, y_ball = zip(*ball_track)
    bounces = bounce_detector.predict(list(x_ball), list(y_ball))
    for idx, frame in enumerate(tracknet_pred_dict['Frame']):
        if frame in bounces:
            tracknet_pred_dict['Bounce'][idx] = 1
    write_pred_csv(tracknet_pred_dict, save_file=out_csv_file)

    court_detector = CourtDetectorNet(path_court_model, 'cuda' if torch.cuda.is_available() else 'cpu')
    save_bounce_frames(bounces, cap, video_name, ball_track, save_dir, court_detector)

    print('推断完成。')



if __name__ == "__main__":

    # 直接调用函数并传递参数
    run_inference(
        ori_video_file='ori_video/result100fps_3min.avi',
        tracknet_file='ckpts/TrackNet_best.pt',
        batch_size=2,
        max_sample_num=1800,
        video_range=None,
        save_dir='prediction/detail',
        seg_video_index=0,
        ori_video_fps=100,
        segment_duration=10,
        segment_video_frame_index=53,
        path_bounce_model='ckpts/bounce_detection_pretrained.cbm',
        path_court_model='ckpts/court_detection_pretrained.pt'
    )
