from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
import torch
import cv2
import copy
from config import config


class ImgProcessor:
    def __init__(self, show_img=True):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetectionYolo()
        self.object_tracker = ObjectTracker()
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.img_black = []
        self.show_img = show_img

    def init_sort(self):
        self.object_tracker.init_tracker()

    def __process_kp(self, kps, idx):
        new_kp = []
        for bdp in range(len(kps)):
            for coord in range(2):
                new_kp.append(kps[bdp][coord])
        return {idx: new_kp}

    def process_img(self, frame):
        # frame = cv2.resize(frame, config.frame_size)
        img = copy.deepcopy(frame)
        img_black = cv2.imread('video/black.jpg')
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(frame)

            if boxes is not None:
                key_points, img, black_img = self.pose_estimator.process_img(inps, orig_img, boxes, scores, pt1, pt2)
                if config.plot_bbox:
                    img = self.BBV.visualize(boxes, frame)

                if key_points is not []:
                    id2ske, id2bbox = self.object_tracker.track(boxes, key_points)
                    if config.plot_id:
                        img = self.IDV.plot_bbox_id(id2bbox, copy.deepcopy(img))
                        # img = self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(img))

                    if config.track_idx != "all":
                        try:
                            kps = self.__process_kp(id2ske[config.track_idx], config.track_idx)
                        except KeyError:
                            kps = {}
                    else:
                        kps = id2ske
                    #
                    # if config.plot_kps:
                    #     vis_kps = self.KPV.dict2ls(kps)
                    #     img = self.KPV.vis_ske(orig_img, vis_kps, kp_score)
                    #     img_black = self.KPV.vis_ske_black(orig_img, vis_kps, kp_score)

                    return kps, img, img_black
                else:
                    return {}, img, img_black
            else:
                return {}, frame, frame

