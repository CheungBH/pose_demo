import os
import cv2
body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}

fourcc = cv2.VideoWriter_fourcc(*'XVID')
write = True
video_dest = 'video/swim/0.avi'


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if write:
            if os.path.exists(video_dest):
                raise ValueError("Target video Exists!")
            else:
                self.out = cv2.VideoWriter(video_dest, fourcc, 20, (1080, 720))

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (1080, 720))
            cnt += 1
            if ret:
                cv2.imshow("res", frame)
                cv2.waitKey(1)
                if write:
                    self.out.write(frame)
            else:
                self.cap.release()
                if write:
                    self.out.release()
                break


if __name__ == '__main__':
    # src_folder = "video/push_up"
    # dest_fodler = src_folder + "kps"
    # sub_folder = [os.path.join(src_folder, folder) for folder in os.listdir(src_folder)]
    # sub_dest_folder = [os.path.join(dest_fodler, folder) for folder in os.listdir(src_folder)]

    video = 0
    VideoProcessor(video).process_video()
