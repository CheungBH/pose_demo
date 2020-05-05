import torch


device = "cuda:0"
print("Using {}".format(device))

confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64

fast_inference = True
pose_batch = 80


frame_size = (540, 360)

pose_backbone = "seresnet"
pose_model = "models/sppe/duc_se.pth"
seresnet_config = None
mobilenet_setting = [
                # t, c, n, s
                [1, 14, 1, 1],
                [6, 24, 2, 2],
                [6, 28, 3, 2],
                [6, 48, 4, 2],
                [6, 72, 3, 1],
                [6, 120, 3, 2],
                [6, 318, 1, 1], ]
DUCs = [480, 240]


if "duc" in pose_model:
    pose_classes = 17
    pose_config = None
    model_class = 33
else:
    pose_classes = 13
    model_class = pose_classes


yolo_cfg = "config/yolo_cfg/yolov3.cfg"
yolo_model = 'models/yolo/yolov3.weights'

track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

plot_bbox = True
plot_kps = True
plot_id = True
