#coding=utf-8
import logging
import platform
from threading import Lock
from app import mvsdk  # 假设你在使用的是海康威视的SDK
import numpy as np

# 全局变量和锁
camera_lock = Lock()  # 用于防止多次初始化摄像头的锁
logger = logging.getLogger()
hCamera = 0
pFrameBuffer = None

def open_camera(DevInfo):
    global  camera_lock,hCamera

    with camera_lock:
        # 检查是否已经初始化过相机
        if hCamera > 0:
            logger.info("Camera is already initialized.")
            return True

        # 尝试初始化相机
        try:
            hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            logger.debug("Camera initialized successfully.")
            return True
        except mvsdk.CameraException as e:
            logger.debug("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return False

def init_camera():
    global hCamera,pFrameBuffer

    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return 0

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    logger.debug(DevInfo)

    mvsdk.CameraUnInit(1)
    hCamera = 0
    # 释放摄像头资源
    mvsdk.CameraAlignFree(pFrameBuffer)
    pFrameBuffer = None
    logger.info("摄像头资源已释放")

    # 打开相机
    ret = open_camera(DevInfo)  # 使用 open_camera 函数来打开摄像头
    if not ret:
        return 0

    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)

    sRoiResolution = mvsdk.tSdkImageResolution()
    sRoiResolution.iIndex = 0xff
    sRoiResolution.iWidth = 1280
    sRoiResolution.iWidthFOV = 1280
    sRoiResolution.iHeight = 720
    sRoiResolution.iHeightFOV = 720

    mvsdk.CameraSetImageResolution(hCamera, sRoiResolution)

    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # 手动曝光，曝光时间30ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)


    # 让SDK内部取图线程开始工作
    mvsdk.CameraPlay(hCamera)

    # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

    # 分配RGB buffer，用来存放ISP输出的图像
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    return hCamera


def grap():
    global hCamera,pFrameBuffer

    # 从相机取一帧图片
    try:
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
        # linux下直接输出正的，不需要上下翻转
        if platform.system() == "Windows":
            mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

        # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
        # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                               1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
        return frame
    except mvsdk.CameraException as e:
        if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
            logger.debug("获取视频帧失败({}): {}".format(e.error_code, e.message))
        return None



# 释放摄像头资源的函数
def close_camera():
    global hCamera,pFrameBuffer
    if hCamera > 0:
        try:
            mvsdk.CameraUnInit(hCamera)
            hCamera = 0
            # 释放摄像头资源
            mvsdk.CameraAlignFree(pFrameBuffer)
            pFrameBuffer = None
            logger.info("摄像头资源已释放")
        except mvsdk.CameraException as e:
            logger.error(f"释放摄像头资源失败: {e.message}")

def get_info():
    return hCamera, pFrameBuffer