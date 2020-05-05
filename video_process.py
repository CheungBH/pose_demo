from src.human_detection import ImgProcessorForbbox
from src.human_track_detection import ImgProcessor
import cv2
import os
import numpy as np
from utils.utils import Utils

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}

IP = ImgProcessor()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
write = True


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if write:
            self.out = cv2.VideoWriter('video/demo_frames.mp4', fourcc, 20, (720, 540))

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (720, 540))
            cnt += 1
            if ret:
                kps, img, black_img = IP.process_img(frame)
                if kps:
                    cv2.putText(img, "cnt{}".format(cnt), (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)

                    cv2.imshow("res", img)
                    cv2.waitKey(2)
                else:
                    img = frame
                    cv2.putText(img, "cnt{}".format(cnt), (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)

                    cv2.imshow("res", img)
                    cv2.waitKey(2)
                if write:
                    self.out.write(img)
            else:
                self.cap.release()
                if write:
                    self.out.release()
                break

    def locate(self, kps):
        return kps


if __name__ == '__main__':
    # src_folder = "video/push_up"
    # dest_fodler = src_folder + "kps"
    # sub_folder = [os.path.join(src_folder, folder) for folder in os.listdir(src_folder)]
    # sub_dest_folder = [os.path.join(dest_fodler, folder) for folder in os.listdir(src_folder)]

    video = "video/VID-20200428-WA0003.mp4"
    VideoProcessor(video).process_video()
