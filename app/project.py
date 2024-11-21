import atexit
import threading
from flask import Flask, session
from app import mvsdk, CameraUtil
from app.common import PreDefine, CommonError
from app.exceptions import TennisError
from app.models import Result
from flask_socketio import SocketIO
import logging
import cv2 as cv
import base64
import os
import sys
import numpy as np
from TennisV3.d3 import read_video, run_inference
from multiprocessing import Queue
from TennisV3.Model_Loader import ModelLoader
from app.mvsdk import CameraGetImageBuffer
from app.routes import http_bp, hCamera
from app.CameraUtil import get_info

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
# 初始化 SocketIO 对象，并将 Flask 应用实例传递给它
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")  # 允许跨域

# 算法模型导入
model_loader = None
tracknet, param_dict = None, None
bounce_detector = None
court_detector = None


def create_app():
    global model_loader
    global tracknet
    global param_dict
    global bounce_detector
    global court_detector
    global socketio
    app.config.from_object('config.Config')

    # app上下文
    app_context = app.app_context()
    app_context.push()

    # 注册蓝图
    app.register_blueprint(http_bp)

    # app.logger.debug(app.url_map)

    # 配置日志记录
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    # 加载模型
    model_loader = ModelLoader()
    # 加载 TrackNet、BounceDetector 和 CourtDetector 模型
    tracknet, param_dict = model_loader.load_tracknet_model('TennisV3/ckpts/TrackNet_best.pt')
    bounce_detector = model_loader.load_bounce_detector('TennisV3/ckpts/bounce_detection_pretrained.cbm')
    court_detector = model_loader.load_court_detector('TennisV3/ckpts/court_detection_pretrained.pt')

    # 全局异常处理器
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.info("[系统异常]%s", e)
        return Result.error(str(e))

    @app.errorhandler(TennisError)
    def handle_exception(e):
        app.logger.info(e.message)
        return Result.error(e.message)

    def release_camera():
        app.logger.debug('Camera released')
        CameraUtil.close_camera()

    atexit.register(release_camera)

    return app, socketio


# websocket接口

# 此活动的session
active_session = []


# 启动时，执行的操作
@socketio.on('connect', namespace='/mapping')
def handle_connect():
    app.logger.info('Mapping is connected')
    active_session.append(session)


# 使用websocket实现仲裁结果的推送
@socketio.on('to_mapping', namespace='/mapping')
def handle_message(message):
    global tracknet
    global param_dict
    global bounce_detector
    global court_detector
    global socketio
    split = message.split("_")
    file_name = f'TennisV3/ori_video/{split[0]}_{split[1]}.avi'
    index = split[3]
    print(f"filename: {file_name} : index: {index}")
    video_path, image_path_list = run_inference(
        ori_video_file=file_name,  # 仲裁原视频Clip
        tracknet=tracknet,
        param_dict=param_dict,
        bounce_detector=bounce_detector,
        court_detector=court_detector,
        batch_size=16,
        max_sample_num=1800,
        video_range=None,
        save_dir='TennisV3/prediction/detail/',
        seg_video_index=int(split[1]),  # 仲裁的降采样视频是原视频的第i个降采样片段
        ori_video_fps=100,  # 仲裁原视频Clip帧率
        segment_duration=8,  # 降采样的视频片段长度
        segment_video_frame_index=int(index),  # 仲裁降采样视频的bounce帧
        down_scale_rate=16)  # 降采样视频帧率
    send_map_result(video_path, image_path_list)


def send_map_result(video_path, image_path_list):
    logging.info("开始发送视频和图片文件...")
    encoded_file = None
    # 获取文件名（不带路径和扩展名）
    video_name = os.path.basename(video_path)  # 获取带后缀的文件名
    video_name = video_name.rsplit('.', 1)[0]  # 去除扩展名
    print(video_name)

    if os.path.exists(video_path):  # 检查视频文件是否存在
        try:
            with open(video_path, 'rb') as file:  # 以二进制读取文件
                file_bytes = file.read()  # 读取文件内容
                encoded_video = base64.b64encode(file_bytes).decode('utf-8')  # 将视频内容进行Base64编码
                # 格式化编码后的文件，包含路径和文件名
                encoded_file = f"{video_name}*{encoded_video}"
        except IOError as e:
            print(f"读取视频文件时出错: {e}")
    else:
        print("文件未找到:", video_path)

    # 处理图片文件
    for image_path in image_path_list:
        with open(image_path, "rb") as image_file:
            # 读取图片内容并编码为Base64
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_name = os.path.basename(image_path)  # 获取带后缀的文件名
            # 将编码后的图像数据添加到文件字符串中
            encoded_file += f"*{image_name}*{encoded_image}"

    if encoded_file:
        socketio.emit('map_result', encoded_file, namespace='/mapping')  # 通过Socket.IO发送数据
        print("仲裁结果发送成功!")


# 关闭时
@socketio.on('disconnect', namespace='/mapping')
def handle_disconnect():
    active_session.remove(session)


# 异常断联
@socketio.on('transport closed', namespace='/mapping')
def handle_transport():
    pass


# 降采集以及分析关键帧相关接口

# 创建一个阻塞队列
q = Queue()  # 使用上下文初始化队列

# 获取视频的相关信息
width = 1280
height = 720
fps = 100
segment_duration = 1
time = 10
segment_number = 0
hCamera = 0


def record_process():
    global q
    global fps
    global width
    global height
    global segment_duration
    global segment_number
    global hCamera

    hCamera, pFrameBuffer = get_info()

    frame_size = (int(width), int(height))
    while hCamera > 0:
        output_file = os.path.join('TennisV3/ori_video/', f'segment_{segment_number}.avi')
        # writer = cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, frame_size, True)
        mvsdk.CameraInitRecord(hCamera, 0, output_file, 0, 60, 100)
        segment_frames = []
        for i in range(int(fps * segment_duration)):
            frame_head = None
            try:
                phy_buffer, frame_head = CameraGetImageBuffer(hCamera, 200)
                mvsdk.CameraImageProcess(hCamera, phy_buffer, pFrameBuffer, frame_head)
                mvsdk.CameraReleaseImageBuffer(hCamera, phy_buffer)

                # 获取帧
                frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((frame_head.iHeight, frame_head.iWidth,
                                       1 if frame_head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

                segment_frames.append(frame)
            except mvsdk.CameraException as e:
                app.logger.debug("视频录制失败({}):{}".format(e.error_code, e.message))

            mvsdk.CameraPushFrame(hCamera, pFrameBuffer, frame_head)

        mvsdk.CameraStopRecord(hCamera)
        q.put(segment_frames)
        print(f"录制成功，文件保存在：{output_file}")
        segment_number += 1


# 降采集并分析关键帧的线程
def handle_process():
    global q
    global segment_number
    global tracknet
    global param_dict
    global bounce_detector
    global hCamera
    index = 0
    while True:
        if not q.empty():
            source = q.get()
            video_file, bounces = read_video(source, fps, index, tracknet, param_dict, bounce_detector)
            send_video(video_file, bounces)
            index += 1
        elif hCamera <= 0:
            break


def send_video(path, bounces):
    global socketio
    print("start to send video")
    encoded_file = None
    # 获取文件名（不带路径和扩展名）
    video_name = os.path.basename(path)  # 获取带后缀的文件名
    video_name = video_name.rsplit('.', 1)[0]  # 去除扩展名
    print(video_name)
    if os.path.exists(path):  # 检查视频文件是否存在
        try:
            with open(path, 'rb') as file:  # 以二进制读取文件
                file_bytes = file.read()  # 读取文件内容
                encoded_file = base64.b64encode(file_bytes).decode('utf-8')  # 将文件内容进行Base64编码
                # 格式化编码后的文件，包含路径和文件名
                encoded_file = f"{video_name}*{encoded_file}"
        except IOError as e:
            print(f"读取视频文件时出错: {e}")
    else:
        print("文件未找到:", path)

    # 打开视频文件
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # 获取视频的总帧数
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    for index in bounces:  # 遍历目录中的文件
        # 检查 frame_index 是否在有效范围内
        if index >= total_frames or index < 0:
            print(f"Error: Invalid frame index. The video has {total_frames} frames.")
            return
        # 设置视频读取位置到指定帧
        cap.set(cv.CAP_PROP_POS_FRAMES, index)
        # 读取指定帧
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read the frame.")
            cap.release()
            return
        _, buffer = cv.imencode('.jpg', frame)  # 将帧编码为JPEG格式
        encoded_frame = base64.b64encode(buffer).decode('utf-8')  # 转为Base64字符串
        encoded_file += f"*frame_{index}*{encoded_frame}"
        # 释放资源
    cap.release()

    if encoded_file:
        print(f'---{encoded_file}')
        socketio.emit('video_transfer', encoded_file, namespace='/predict')  # 通过Socket.IO发送数据
        print("视频发送成功!")


@socketio.on('connect', namespace='/predict')
def handle_connect():
    app.logger.info('Predict is connected')


# 录制视频以及降采集请求
@socketio.on('video_record', namespace='/predict')
def video_record(mode):
    print(f"mode: {mode}")

    #测试模式
    if mode == PreDefine.TEST_MODE:
        path = os.path.join('TennisV3/ori_video/result100fps_16s.avi')
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return
        #获取视频总帧数
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        segment_frames = []

        #将每一帧存入数组中
        for index in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                raise TennisError(CommonError.VIDEO_READ_ERROR)
            segment_frames.append(frame)

        #将帧数组存入队列中
        q.put(segment_frames)
        handle_process()


    #实时模式
    elif mode == PreDefine.REAL_TIME_MODE:
        # 创建生产者线程
        t1 = threading.Thread(target=record_process)
        # 创建消费者线程
        t2 = threading.Thread(target=handle_process)
        t1.start()
        t2.start()
        t1.join()
        t2.join()


