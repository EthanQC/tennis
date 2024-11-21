import cv2
import os
import csv

# 创建保存视频段的路径
output_folder = 'se_input'
os.makedirs(output_folder, exist_ok=True)

def read_video(source, segment_duration=10, target_fps=15):
    """
    从原始视频中读取帧，并按指定帧率和段长生成视频段。
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Original Video FPS: {fps}, Total Frames: {frame_count}')

    segment_frames = []  # 存储当前段的视频帧
    segment_number = 0  # 命名视频段

    # 自动计算采样步长
    downsample_rate = max(1, fps // target_fps)  # 确保步长至少为1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        segment_frames.append(frame)

        # 如果当前段满 segment_duration 秒，保存并重置帧列表
        if len(segment_frames) >= fps * segment_duration:
            save_segment(segment_frames, target_fps, segment_number, downsample_rate)
            segment_frames = []  # 重置帧列表
            segment_number += 1  # 更新段编号

    # 如果有剩余帧，保存为最后一个视频段
    if segment_frames:
        save_segment(segment_frames, target_fps, segment_number, downsample_rate)

    cap.release()

def save_segment(frames, target_fps, segment_number, downsample_rate):
    """
    保存视频段，并将帧对应关系保存为 CSV 文件。
    """
    # 视频段文件路径
    segment_path = os.path.join(output_folder, f'segment_{segment_number}.mp4')

    # 获取帧尺寸 (height, width)
    height, width, _ = frames[0].shape

    # 使用 mp4v 编码器以确保与 MP4 格式的兼容性
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(segment_path, fourcc, target_fps, (width, height))

    original_indices = []  # 保存对应的原始帧索引

    # 遍历所有帧并降采样
    for index, frame in enumerate(frames):
        if index % downsample_rate == 0:
            out.write(frame)  # 写入降采样后的帧
            original_indices.append(index)

    out.release()  # 释放资源，确保视频可播放

    print(f'Saved segment {segment_number} to {segment_path}')

    # # 将帧索引信息保存为 CSV 文件（可选）
    # csv_path = os.path.join(output_folder, f'segment_{segment_number}.csv')
    # with open(csv_path, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(['Frame', 'Original_Frame_Index'])  # 写入表头
    #     for i, original_index in enumerate(original_indices):
    #         csv_writer.writerow([i, original_index])  # 写入帧对应关系

    # print(f'Saved frame data to {csv_path}')

def get_original_frame_index(segment_index, k, fps, segment_duration, downsample_rate):
    """
    计算 segment_i.mp4 的第 k 帧在原始视频中的帧索引。
    """
    total_frames_per_segment = fps * segment_duration
    original_frame_index = segment_index * total_frames_per_segment + k * downsample_rate
    return original_frame_index

if __name__ == '__main__':
    # 原始视频路径
    path_video = 'ori_video/result100fps_3min.avi'
    
    # 读取视频并生成段视频，目标帧率为 15 FPS
    read_video(path_video, segment_duration=10, target_fps=15)
