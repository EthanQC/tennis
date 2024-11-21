import atexit
from flask import Flask, session
from app import mvsdk, CameraUtil
from app.exceptions import TennisError
from app.models import Result
import logging
import os
import sys
from TennisV3.Model_Loader import ModelLoader
from app.routes import http_bp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

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

    return app




