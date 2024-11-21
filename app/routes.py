import logging
import platform
from flask import Blueprint,Response,after_this_request
import cv2 as cv
from app import mvsdk, CameraUtil
from app.exceptions import TennisError
from app.common import PreDefine, CommonError
import numpy as np

#使用HTTP协议请求
http_bp = Blueprint('http_bp', __name__)

logger = logging.getLogger()

hCamera = 0

def send_video():
    if hCamera < 0:
        raise TennisError(CommonError.CAMERA_NOT_AVAILABLE)
    while True:
        frame = CameraUtil.grap()
        if frame is None:
            break
        else:
            # 将帧编码为 JPEG 格式
            ret, buffer = cv.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            # 构建响应体
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@http_bp.route('/video',methods=['GET'])
def stream_video():
    global hCamera
    logger.debug('==========stream_video=========')
    # 初始化摄像头
    hCamera,pFrameBuffer=CameraUtil.init_camera()
    
    return Response(send_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@http_bp.route('/favicon.ico',methods=['GET'])
def favicon():
    # 返回 HTTP 204 无内容响应
    return '', 204


#在app关闭时释放摄像头资源
@http_bp.route('/cleanUp',methods=['GET'])
def cleanup_camera():
    CameraUtil.close_camera()



