from datetime import datetime
from pathlib import Path
"""
存放常量类
"""

#存放异常信息的常量类
class CommonError:
    #统一异常处理捕获的位置错误
    UNKNOWN_ERROR="发生错误，请稍后再试"

    # 摄像头相关错误
    CAMERA_NOT_AVAILABLE = "摄像头不可用"
    CAMERA_READ_ERROR = "读取录像失败"

    # video相关错误
    VIDEO_SEND_ERROR = "发送视频失败"
    VIDEO_WRITE_ERROR = "视频写入失败"
    VIDEO_READ_ERROR = "视频读取失败"

    # 指令执行相关错误
    PROCESS_EXE_ERROR = "指令执行异常"


#存放常量的类
class PreDefine:
    #与安卓端约定的录制模式——实时录制并处理
    REAL_TIME_MODE = "real_time_mode"

    #与安卓端约定的录制模式——使用已经录好的测试视频进行处理
    TEST_MODE = "test_mode"

    #BASE_DIR = Path('F:/pythonCode/')#Path("/home/jetson/POJ/TennisV3/")
    BASE_DIR = Path("/home/jetson/tennis/TennisV3/")#'E:/ceshi/'

    #GStreamer的管道配置
    @staticmethod
    def PIPELINE_CONFIG(width, height, fps) :
        return f"v4l2src device=/dev/video0 io-mode=2 " \
                  f"! image/jpeg, width={width}, height={height}, framerate={fps}/1, format=MJPG " \
                  f"! nvv4l2decoder mjpeg=1 " \
                  f"! nvvidconv " \
                  f"! video/x-raw, format=BGRx " \
                  f"! videoconvert " \
                  f"! video/x-raw, format=BGR " \
                  f"! appsink drop=1"


    #视频的文件名
    @staticmethod
    def VIDEO_FILENAME(videoFileName) :
        return f"{PreDefine.BASE_DIR}ori_video/{videoFileName}.avi"

    #接收到的视频名
    @staticmethod
    def ORI_VIDEO_NAME(oriVideoName) :
        return f"{PreDefine.BASE_DIR}ori_video/{oriVideoName}"


    #发送100帧视频的文件名
    @staticmethod
    def FILENAME(oriVideoNameWithoutSuffix,segmentIndex,frameIndex):
        return f"prediction/detail/{oriVideoNameWithoutSuffix}/centered_segment_{segmentIndex}_frame_{frameIndex}.mp4"

    @staticmethod
    def DatetimeUtil():
        now = datetime.now()
        return now.strftime("%Y_%m_%d_%H_%M_%S")