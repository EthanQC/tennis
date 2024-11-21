import os
import cv2
import math
import parse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset
from TennisV3.utils.general import get_rally_dirs, get_match_median, HEIGHT, WIDTH, SIGMA, IMG_FORMAT

data_dir = 'data'
global_median = None  # 全局变量用于缓存 median image

class Shuttlecock_Trajectory_Dataset(Dataset):
    def __init__(self,
        root_dir=data_dir,
        split='train',
        seq_len=8,
        sliding_step=1,
        data_mode='heatmap',
        bg_mode='',
        frame_alpha=-1,
        rally_dir=None,
        frame_arr=None,
        pred_dict=None,
        padding=False,
        debug=False,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        SIGMA=SIGMA,
        median=None
    ):
        global global_median  # 使用全局 median image
        assert split in ['train', 'test', 'val'], f'Invalid split: {split}, should be train, test or val'
        assert data_mode in ['heatmap', 'coordinate'], f'Invalid data_mode: {data_mode}, should be heatmap or coordinate'
        assert bg_mode in ['', 'subtract', 'subtract_concat', 'concat'], f'Invalid bg_mode: {bg_mode}, should be "", subtract, subtract_concat or concat'

        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.mag = 1
        self.sigma = SIGMA

        self.root_dir = root_dir
        self.split = split if rally_dir is None else self._get_split(rally_dir)
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha

        self.frame_arr = frame_arr
        self.pred_dict = pred_dict
        self.padding = padding and self.sliding_step == self.seq_len

        # 加载或生成 median image
        if self.frame_arr is not None:
            assert self.data_mode == 'heatmap', f'Invalid data_mode: {self.data_mode}, frame_arr only for heatmap mode' 
            self.data_dict, self.img_config = self._gen_input_from_frame_arr()
            if self.bg_mode:
                if median is None and global_median is None:
                    global_median = np.median(self.frame_arr, 0)
                self.median = global_median
                if self.bg_mode == 'concat':
                    self.median = Image.fromarray(self.median.astype('uint8'))
                    self.median = np.array(self.median.resize(size=(self.WIDTH, self.HEIGHT)))
                    self.median = np.moveaxis(self.median, -1, 0)
            else:
                self.median = global_median

        # 其他数据初始化
        # 省略部分初始化代码以节省空间

class Video_IterableDataset(IterableDataset):
    """ Dataset for inference especially for large video. """
    def __init__(self,
        video_file,
        seq_len=8,
        sliding_step=1,
        bg_mode='',
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        max_sample_num=1800,
        video_range=None,
        median=None
    ):
        global global_median  # 使用全局 median image
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.video_file = video_file
        self.cap = cv2.VideoCapture(self.video_file)
        self.video_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.w, self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w_scaler, self.h_scaler = self.w / self.WIDTH, self.h / self.HEIGHT

        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.bg_mode = bg_mode

        if self.bg_mode:
            if global_median is None:
                global_median = self.__gen_median__(max_sample_num, video_range)
            self.median = global_median

    def __gen_median__(self, max_sample_num, video_range):
        print('Generating median image...')
        if video_range is None:
            start_frame, end_frame = 0, self.video_len
        else:
            start_frame = max(0, video_range[0] * self.fps)
            end_frame = min(video_range[1] * self.fps, self.video_len)
        video_seg_len = end_frame - start_frame
        sample_step = max(1, video_seg_len // max_sample_num)

        frame_list = []
        for i in range(start_frame, end_frame, sample_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = self.cap.read()
            if not success:
                break
            frame_list.append(frame)
        
        median = np.median(frame_list, 0)[..., ::-1]  # BGR to RGB
        if self.bg_mode == 'concat':
            median = Image.fromarray(median.astype('uint8'))
            median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
            median = np.moveaxis(median, -1, 0)  # 转换为 (C, H, W) 格式
        print('Median image generated.')
        return median

    def __iter__(self):
        """ Return the data sequentially. """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success = True
        start_f_id, end_f_id = 0, 0
        frame_list = []
        while success:
            while len(frame_list) < self.seq_len:
                success, frame = self.cap.read()
                if not success:
                    break
                frame_list.append(frame)
                end_f_id += 1

            data_idx = [(0, i) for i in range(start_f_id, end_f_id)]

            if len(data_idx) < self.seq_len:
                data_idx.extend([(0, end_f_id-1)] * (self.seq_len - len(data_idx)))
                frame_list.extend([frame_list[-1]] * (self.seq_len - len(frame_list)))
                
            data_idx = np.array(data_idx)
            frames = self.__process__(np.array(frame_list)[..., ::-1])
            yield data_idx, frames

            frame_list = frame_list[self.sliding_step:]
            start_f_id += self.sliding_step

        self.cap.release()

    def __process__(self, imgs):
        frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
        median_img = self.median if self.bg_mode else None

        for i in range(self.seq_len):
            img = Image.fromarray(imgs[i])
            if self.bg_mode == 'subtract':
                img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = img.reshape(1, self.HEIGHT, self.WIDTH)
            elif self.bg_mode == 'subtract_concat':
                diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0)
                img = np.concatenate((img, diff_img), axis=0)
            else:
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0)

            frames = np.concatenate((frames, img), axis=0)

        if self.bg_mode == 'concat':
            frames = np.concatenate((median_img, frames), axis=0)

        frames /= 255.
        return frames
