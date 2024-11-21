import cv2
import numpy as np
from TennisV3.court_reference import CourtReference
import argparse
import os


def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
    return court_img

def save_minimap(court_img, frame_num, output_dir, file_name, ball_point,inv_mat):
    """
    保存带有弹跳点的minimap图
    :param court_img: 带有弹跳点的minimap图片
    :param frame_num: 弹跳发生时的帧编号
    :param output_dir: 保存路径
    :param file_name: 文件名
    """
    width_minimap = 664#166
    height_minimap = 1400#350

    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
    
    court_img = cv2.circle(court_img, 
                        (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                        radius=6, color=(0, 255, 255), thickness=-1)
    court_img = cv2.circle(court_img, 
                        (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                        radius=1, color=(0, 0, 255), thickness=-1)  
    os.makedirs(output_dir, exist_ok=True)
    minimap_path = os.path.join(output_dir, f"{file_name}_minimap_{frame_num}.png")
    cv2.imwrite(minimap_path, court_img)
    print(f'minimap_path: {minimap_path}')
    crop_width = 400  # 裁剪的宽度
    crop_height = 300  # 裁剪的高度

    ball_x, ball_y = int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])

    # 计算裁剪区域的左上角和右下角坐标
    x1 = max(0, ball_x - crop_width // 2)
    y1 = max(0, ball_y - crop_height // 2)
    x2 = min(court_img.shape[1], ball_x + crop_width // 2)
    y2 = min(court_img.shape[0], ball_y + crop_height // 2)
    cropped_img = court_img[y1:y2, x1:x2]
    minimap_cropped_path = os.path.join(output_dir, f"{file_name}_minimap_cropped_{frame_num}.png")
    cv2.imwrite(minimap_cropped_path, cropped_img)
    print(f'minimap_cropped_path: {minimap_cropped_path}')
    return minimap_path,minimap_cropped_path

def draw_ball_trace(image_path, ball_track, ball_point_bounce_num, trace_len, output_dir, court_detector):
    imgs_res = []
    ball_point = ball_track[ball_point_bounce_num]
    print(ball_point)

    width_minimap = 400  #166
    height_minimap = 300  #350

    # Load the frame from the given path
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法加载图片：{image_path}")
        return

    # Draw the ball trajectory on the frame
    print(f"绘制轨迹")
    for j in range(len(ball_track)):
        x, y = ball_track[j]
        if x is not None and y is not None and (x, y) != (0, 0):
            frame = cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=1)

    file_name = os.path.basename(image_path)
    split1 = file_name.split('_')
    split2 = split1[-1].split('.')
    frame_part = split1[-2] + '_' + split2[0]

    bounce_frame_path = os.path.join(output_dir, f"{frame_part}_bounce_frame.png")
    cv2.imwrite(bounce_frame_path, frame)

    width_minimap = 400  #166
    height_minimap = 300  #350
    # Original aspect ratio 4:3

    homography_matrices, kps_court = court_detector.infer_single_image(frame)
    inv_mat = homography_matrices
    court_img = get_court_img()

    height, width, _ = frame.shape
    minimap_path, minimap_cropped_path=None,None
    if inv_mat is not None:
        ball_point_bounce_num = ball_track[ball_point_bounce_num]
        minimap_path,minimap_cropped_path = save_minimap(court_img, ball_point_bounce_num, output_dir, frame_part, ball_point, inv_mat)
    return inv_mat, bounce_frame_path, minimap_path, minimap_cropped_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_court_model', type=str, default='TennisV3/ckpts/court_detection_pretrained.pt',
                        help='Path to pretrained model for court detection')

    #
    args = parser.parse_args()
    # # Example usage
    # image_path = '/home/jetson/POJ/TennisV3/prediction/bounce_frames/centered_segment_0_frame_53_bounce_frame_25.png'
    # ball_track = [
    #     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (710, 172), (0, 0),
    #     (0, 0), (0, 0), (730, 170), (730, 170), (732, 172), (740, 172), (0, 0), (0, 0), (0, 0),
    #     (0, 0), (0, 0), (0, 0), (0, 0), (765, 182), (765, 182), (765, 182), (772, 185), (772, 185),
    #     (775, 185), (780, 185), (780, 187), (780, 185), (785, 177), (785, 172), (785, 170), (792, 165),
    #     (792, 162), (795, 160), (797, 152), (797, 155), (797, 150), (802, 145), (802, 140), (805, 137),
    #     (805, 135), (805, 135), (807, 135), (810, 130), (810, 130), (812, 127), (815, 125), (817, 122),
    #     (817, 120), (820, 117), (820, 117), (0, 0), (0, 0), (0, 0), (0, 0)
    # ]
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # court_detector = CourtDetectorNet(args.path_court_model, device)
    #
    #
    #
    # ball_point_bounce_num = 25
    #
    # draw_ball_trace(image_path, ball_track, ball_point_bounce_num, trace_len=7, output_dir='court_detect',court_detector=court_detector)
