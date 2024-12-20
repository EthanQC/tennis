
import cv2
import numpy as np
import torch
from TennisV3.tracknet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from TennisV3.postprocess import refine_kps
from TennisV3.homography import get_trans_matrix, refer_kps

class CourtDetectorNet():
    def __init__(self, path_model=None,  device='cuda'):
        self.model = BallTrackerNet(out_channels=15)
        self.device = device
        if path_model:
            model_weights = torch.load(path_model, map_location=device)
            self.model.load_state_dict(model_weights)
            self.model = self.model.to(device)
            self.model.eval()
            
    def infer_model(self, frames):
        output_width = 640
        output_height = 360
        scale = 2 
        
        kps_res = []
        matrixes_res = []
        for num_frame, image in enumerate(tqdm(frames)):
            img = cv2.resize(image, (output_width, output_height))
            inp = (img.astype(np.float32) / 255.)
            inp = torch.tensor(np.rollaxis(inp, 2, 0))
            inp = inp.unsqueeze(0)
            inp = inp.float()  


            out = self.model(inp.float().to(self.device))[0]
            #pred = F.sigmoid(out).detach().cpu().numpy()
            pred = torch.sigmoid(out).detach().cpu().numpy()

            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num]*255).astype(np.uint8)
                ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                           minRadius=10, maxRadius=25)
                if circles is not None:
                    x_pred = circles[0][0][0]*scale
                    y_pred = circles[0][0][1]*scale
                    if kps_num not in [8, 12, 9]:
                        x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
                    points.append((x_pred, y_pred))                
                else:
                    points.append(None)
            
            # (debug)Ensure points are 1-D before passing to get_trans_matrix
            #points = [np.array(point).flatten() if point is not None else None for point in points]


            matrix_trans = get_trans_matrix(points) 
            points = None
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                matrix_trans = cv2.invert(matrix_trans)[1]
            kps_res.append(points)
            matrixes_res.append(matrix_trans)
            
        return matrixes_res, kps_res    
    
    def infer_single_image(self, image):
        output_width = 640
        output_height = 360
        scale = 2

        # Resize and prepare the image
        img = cv2.resize(image, (output_width, output_height))
        inp = (img.astype(np.float32) / 255.0)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)
        inp = inp.float().to(self.device)

        # Perform inference
        out = self.model(inp)[0]
        pred = torch.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                       minRadius=10, maxRadius=25)
            if circles is not None:
                x_pred = circles[0][0][0] * scale
                y_pred = circles[0][0][1] * scale
                if kps_num not in [8, 12, 9]:
                    x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
                points.append((x_pred, y_pred))
            else:
                points.append(None)

        # Compute homography matrix
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            matrix_trans = cv2.invert(matrix_trans)[1]

        return matrix_trans, points