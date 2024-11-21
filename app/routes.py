import logging
import threading
from flask import Blueprint,Response
import cv2 as cv
from app import mvsdk, CameraUtil
from app.exceptions import TennisError
from app.common import PreDefine, CommonError

#使用HTTP协议请求
http_bp = Blueprint('http_bp', __name__)

logger = logging.getLogger()

Lock = threading.Lock()  # 在threading模块中获得锁类

def send_video(hCamera):
    if hCamera < 0:
        raise TennisError(CommonError.CAMERA_NOT_AVAILABLE)
    while True:
        Lock.acquire()  # 设置锁
        frame = CameraUtil.grap()
        if frame is None:
            Lock.release()  # 释放锁
            break
        else:
            # 将帧编码为 JPEG 格式
            ret, buffer = cv.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            # 构建响应体
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        Lock.release()  # 释放锁

@http_bp.route('/video',methods=['GET'])
def stream_video():
    logger.debug('==========stream_video=========')
    # 初始化摄像头
    hCamera=CameraUtil.init_camera()

    video = threading.Thread(target=send_video,args=(hCamera,))
    video.start()
    
    return Response(send_video(hCamera), mimetype='multipart/x-mixed-replace; boundary=frame')

@http_bp.route('/favicon.ico',methods=['GET'])
def favicon():
    # 返回 HTTP 204 无内容响应
    return '', 204


#在app关闭时释放摄像头资源
@http_bp.route('/cleanUp',methods=['GET'])
def cleanup_camera():
    CameraUtil.close_camera()



