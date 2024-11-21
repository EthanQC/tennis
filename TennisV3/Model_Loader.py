# import torch  # 用于加载模型文件和分配设备
# from bounce_detector import BounceDetector  # 用于加载 BounceDetector 模型
# from utils.general import get_model  # 用于加载 TrackNet 模型

# # 指定设备 (CPU 或 GPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class ModelLoader:
#     """Singleton-style class for loading and storing the TrackNet and BounceDetector models."""
#     _tracknet = None
#     _bounce_detector = None
#     _param_dict = None

#     @staticmethod
#     def load_tracknet_model(tracknet_file):
#         if ModelLoader._tracknet is None:
#             # 加载 TrackNet 模型权重
#             tracknet_ckpt = torch.load(tracknet_file, map_location=device)
#             ModelLoader._param_dict = tracknet_ckpt['param_dict']
#             tracknet_seq_len = ModelLoader._param_dict['seq_len']
#             bg_mode = ModelLoader._param_dict['bg_mode']
#             tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
#             tracknet.load_state_dict(tracknet_ckpt['model'])
#             ModelLoader._tracknet = tracknet
#             print("TrackNet model loaded on device:", device)
#         return ModelLoader._tracknet, ModelLoader._param_dict

#     @staticmethod
#     def load_bounce_detector(bounce_model_path):
#         if ModelLoader._bounce_detector is None:
#             # 加载 BounceDetector 模型
#             ModelLoader._bounce_detector = BounceDetector(bounce_model_path)
#             print("BounceDetector model loaded.")
#         return ModelLoader._bounce_detector

import torch  # 用于加载模型文件和分配设备
from TennisV3.bounce_detector import BounceDetector  # 用于加载 BounceDetector 模型
from TennisV3.court_detection_net import CourtDetectorNet  # 用于加载 CourtDetectorNet 模型
from TennisV3.utils.general import get_model  # 用于加载 TrackNet 模型

# 指定设备 (CPU 或 GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelLoader:
    """Class for loading and storing the TrackNet, BounceDetector, and CourtDetector models."""
    
    def __init__(self):
        self._tracknet = None
        self._bounce_detector = None
        self._court_detector = None
        self._param_dict = None

    def load_tracknet_model(self, tracknet_file):
        if self._tracknet is None:
            # 加载 TrackNet 模型权重
            tracknet_ckpt = torch.load(tracknet_file, map_location=device)
            self._param_dict = tracknet_ckpt['param_dict']
            tracknet_seq_len = self._param_dict['seq_len']
            bg_mode = self._param_dict['bg_mode']
            tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
            tracknet.load_state_dict(tracknet_ckpt['model'])
            self._tracknet = tracknet
            print("TrackNet model loaded on device:", device)
        return self._tracknet, self._param_dict

    def load_bounce_detector(self, bounce_model_path):
        if self._bounce_detector is None:
            # 加载 BounceDetector 模型
            self._bounce_detector = BounceDetector(bounce_model_path)
            print("BounceDetector model loaded.")
        return self._bounce_detector

    def load_court_detector(self, court_model_path):
        if self._court_detector is None:
            # 加载 CourtDetector 模型
            self._court_detector = CourtDetectorNet(court_model_path, device)
            print("CourtDetector model loaded on device:", device)
        return self._court_detector

