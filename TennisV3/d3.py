import os
import cv2
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from TennisV3.test import predict_location
from TennisV3.dataset import Video_IterableDataset
from TennisV3.utils.general import *
from TennisV3.downscale import get_original_frame_index
from TennisV3.draw_court import draw_ball_trace #绘制轨迹以及落点映射的代码
import base64

# Folder to save video segments
output_folder = 'TennisV3/se_input/'
os.makedirs(output_folder, exist_ok=True)


# Specify the computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(indices, y_pred=None, img_scaler=(1, 1)):
    """Predict coordinates from heatmap."""
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Bounce': []}
    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices

    ball_track = []
    y_pred = y_pred > 0.5
    y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    y_pred = to_img_format(y_pred)

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

def save_segment_downscale(frames, target_fps, segment_number, downsample_rate):
    """
    Save the video segment and return the path.
    """
    #encoded_file = f'segment_{segment_number}*'
    segment_path = os.path.join(output_folder, f'segment_{segment_number}.mp4')
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(segment_path, fourcc, target_fps, (width, height))

    for index, frame in enumerate(frames):
        if index % downsample_rate == 0:
            out.write(frame)
            # # 将帧转换为Base64格式
            # _, buffer = cv2.imencode('.jpg', frame)  # 将帧压缩为JPEG格式
            # frame_base64 = base64.b64encode(buffer).decode('utf-8')  # 转换为Base64字符串
            # encoded_file += frame_base64

    out.release()
    print(f'Saved segment {segment_number} to {segment_path}')
    return segment_path#返回该切片文件路径


#取原视频Clip0 的第200帧的前后共60帧，也就是170-229帧 
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



#仲裁的详细检测阶段保存bounce帧
def save_bounce_frames(bounces, cap, video_name, ball_track, output_dir, court_detector):
    bounce_frames_dir = os.path.join(output_dir, 'bounce_frames/')
    os.makedirs(bounce_frames_dir, exist_ok=True)
    image_path_list = []
    for frame_num in bounces:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(bounce_frames_dir, f'{video_name}_bounce_frame_{frame_num}.png')
            cv2.imwrite(frame_path, frame)
            inv_mat,bounce_frame_path,minimap_path,minimap_cropped_path=draw_ball_trace(frame_path, ball_track, frame_num, trace_len=7, output_dir='TennisV3/court_detect/', court_detector=court_detector)
            print(f"反弹帧已保存: {frame_path}")
            if inv_mat is not None:
                image_path_list.append(bounce_frame_path)
                image_path_list.append(minimap_path)
                image_path_list.append(minimap_cropped_path)
    return image_path_list

def read_video(source,fps,segment_number, tracknet, param_dict, bounce_detector):
    """
    从原始视频中读取帧，生成视频段并进行预测。
    """
    target_fps=16
    downsample_rate = max(1, fps // target_fps)
    print(f"接收到视频片段，并开始降采集：{segment_number}")
    segment_path = save_segment_downscale(source, target_fps, segment_number, downsample_rate)
    video_file,bounces=predict_segment(segment_path, tracknet, param_dict, bounce_detector)
    return video_file,bounces

def predict_segment(video_file, tracknet, param_dict, bounce_detector):
    """
    Predict using the model on a video segment.
    """
    cap = cv2.VideoCapture(video_file)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_scaler = (w / WIDTH, h / HEIGHT)
    seq_len = param_dict['seq_len']
    bg_mode = param_dict['bg_mode']

    dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=8, bg_mode=bg_mode)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)

    tracknet.eval()
    tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Bounce': []}
    ball_track = []

    for step, (i, x) in enumerate(tqdm(data_loader)):
        x = x.float().to(device)
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()

        tmp_pred, tmp_ball_track = predict(i, y_pred=y_pred, img_scaler=img_scaler)
        ball_track.extend(tmp_ball_track)
        for key in tmp_pred.keys():
            tracknet_pred_dict[key].extend(tmp_pred[key])

    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)
    print("Bounces detected at frames:", bounces)
    return video_file,bounces



#仲裁
def run_inference(
    ori_video_file, tracknet, param_dict, bounce_detector, court_detector,
    batch_size, max_sample_num, video_range,
    save_dir, seg_video_index, ori_video_fps, segment_duration, segment_video_frame_index,down_scale_rate):

    os.makedirs(save_dir, exist_ok=True)

    downsample_rate = ori_video_fps // down_scale_rate #down_scale_rate表示降采样片段帧率

    # #假设：通过映射计算得知，Clip0 se0的第80帧，对应的是原视频Clip0 的第200帧
    # frame_index = get_original_frame_index(seg_video_index, segment_video_frame_index,
    #                                        ori_video_fps, segment_duration, downsample_rate)
    #
    print(f"对应原视频的第{segment_video_frame_index}帧")
    
    #取原视频Clip0 的第200帧的前后共60帧，也就是170-229帧 进行predict，找出更准确的落地帧
    frame_list = extract_frames(ori_video_file, segment_video_frame_index)
    output_video_path = os.path.join(save_dir, f'centered_segment_{seg_video_index}_frame_{segment_video_frame_index}.mp4')
    frames_to_video(frame_list, output_video_path, fps=ori_video_fps)

    video_file = output_video_path
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_csv_file = os.path.join(save_dir, f'{video_name}.csv')

    cap = cv2.VideoCapture(video_file)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_scaler = (w / WIDTH, h / HEIGHT)

    dataset = Video_IterableDataset(video_file, seq_len=param_dict['seq_len'], sliding_step=8,
                                    bg_mode=param_dict['bg_mode'], max_sample_num=max_sample_num, video_range=video_range)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Bounce': []}
    ball_track = []

    for step, (i, x) in enumerate(tqdm(data_loader)):
        x = x.float().to(device)
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()

        tmp_pred, tmp_ball_track = predict(i, y_pred=y_pred, img_scaler=img_scaler)
        ball_track.extend(tmp_ball_track)
        for key in tmp_pred.keys():
            tracknet_pred_dict[key].extend(tmp_pred[key])

    print(ball_track,len(ball_track))

    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)
    print(bounces)
    print('********************')
    for i in bounces:
        print(ball_track[i])

    for idx, frame in enumerate(tracknet_pred_dict['Frame']):
        if frame in bounces:
            tracknet_pred_dict['Bounce'][idx] = 1
    write_pred_csv(tracknet_pred_dict, save_file=out_csv_file)

    image_path_list=save_bounce_frames(bounces, cap, video_name, ball_track, save_dir, court_detector)
    return output_video_path,image_path_list



# if __name__ == '__main__':
#
#     # 实例化 ModelLoader
#     model_loader = ModelLoader()
#
#     # 加载 TrackNet、BounceDetector 和 CourtDetector 模型
#     tracknet, param_dict = model_loader.load_tracknet_model('ckpts/TrackNet_best.pt')
#     bounce_detector = model_loader.load_bounce_detector('ckpts/bounce_detection_pretrained.cbm')
#     court_detector = model_loader.load_court_detector('ckpts/court_detection_pretrained.pt')
#
#     """
#     以上从实例化是在后端进行的，实例化后像下面的一样传参调用即可，类地址 from Model_Loader import ModelLoader
#
#     """
#
#
#
#
#     video_file = 'ori_video/result100fps_16s.avi'
#
#
#     # 分段处理视频并推断反弹帧
#     #read_video(video_file, segment_duration=8, target_fps=16,tracknet=tracknet, param_dict=param_dict, bounce_detector=bounce_detector)
#     #要处理的文件，降采样切片文件时长（s），降采样帧率，后面都是模型参数
#     for segment_path, result in read_video(video_file, segment_duration=8, target_fps=16,
#                                            tracknet=tracknet, param_dict=param_dict, bounce_detector=bounce_detector):
#
#         #将{segment_path}和result传出去（即切片和相应切片的预测结果
#         print(f"Processed segment saved at: {segment_path}")
#         print("Segment processing bounce result:", result)
#
#
#
#     # 使用 run_inference 进行详细推断
#     run_inference(
#         ori_video_file='ori_video/result100fps_16s.avi', #仲裁原视频Clip
#         tracknet=tracknet,
#         param_dict=param_dict,
#         bounce_detector=bounce_detector,
#         court_detector=court_detector,
#         batch_size=16,
#         max_sample_num=1800,
#         video_range=None,
#         save_dir='prediction/detail',
#         seg_video_index=0,#仲裁的降采样视频是原视频的第i个降采样片段
#         ori_video_fps=100,#仲裁原视频Clip帧率
#         segment_duration=8,#降采样的视频片段长度
#         segment_video_frame_index=53, #仲裁降采样视频的bounce帧
#         down_scale_rate = 16#降采样视频帧率
#     )
