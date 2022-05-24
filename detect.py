import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

# The full COCO dataset categories
# However, we only want to look at cars
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

AREAS = ["woodlands", "sle", "tpe", "kje", "bke", "cte",
         "pie", "kpe", "aye", "mce", "ecp", "stg"]


def get_prediction(img_path, confidence, model_quality):
    """
    Uses pre-trained torchvision models to detect objects in images.
    :param img_path: string, the path to an image.
    :param confidence: float, the confidence level to sift predictions with high certainties.
    :param model_quality: string, low, medium or high. A higher quality model takes longer time to detect.
    :return: pred_boxes (predicted bounding boxes) and pred_classes (predicted object classes).
    """

    # Load the pre-trained Faster R-CNN model from torchvision
    if model_quality == "low":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    elif model_quality == "medium":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    elif model_quality == "high":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Set the model to evaluation mode
    model.eval()

    img = Image.open(img_path)
    transform = T.ToTensor()
    try:
        img = transform(img)
    except OSError:
        print("Truncated image!")
        return [], []
    pred = model([img])
    # Pred is composed of "boxes", "labels", and "scores"
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]["boxes"].detach().numpy())]
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]["labels"].detach().numpy())]
    pred_scores = pred[0]["scores"].detach().numpy()
    if len(pred_boxes) == 0 or pred_scores.max() < confidence:
        return [], []
    thres = np.argwhere(pred_scores >= confidence)[-1][0]
    # pred_t = [pred_scores.index(x) for x in pred_scores if x > confidence][-1]
    return pred_boxes[:thres + 1], pred_classes[:thres + 1]


def detect_object(img_dir, tar_class="car", confidence=0.3, main_dir="/home/xinyuan/images/",
                  model_quality="medium", display_res=False, verbose=True):
    """
    Detects objects in all the images located in a given directory
    and counts the number of detected objects.
    :param img_dir: string, folder name.
    :param tar_class: string, target object class for detection.
    :param confidence: float, the confidence level to sift predictions with high certainties.
    :param main_dir: string, the main folder that contains sub-folders of images.
    :param model_quality: string, low, medium or high. A higher quality model takes longer time to detect.
    :param display_res: boolean, specifies whether to display the resulting detections.
    :param verbose: boolean, specifies whether to display detailed logging.
    :return: count_sum (the total number of detected objects in all images located in img_dir)
    """
    count_sum = 0
    # for filename in os.listdir("./" + img_dir):
    for filename in os.listdir(main_dir + img_dir):
        if verbose:
            print("Processing", filename, "...")
        img_path = main_dir + img_dir + "/" + filename
        pred_boxes, pred_classes = get_prediction(img_path, confidence, model_quality)
        if tar_class and tar_class in COCO_INSTANCE_CATEGORY_NAMES:
            pred_idx = [i for i in range(len(pred_classes)) if pred_classes[i] == tar_class]
            pred_boxes = [pred_boxes[j] for j in pred_idx]
            pred_classes = [tar_class] * len(pred_boxes)
            if verbose:
                print(len(pred_boxes), tar_class, "(s) found in the image!")
        elif tar_class:
            if verbose:
                print("Invalid class name. Exit.")
            break
        else:
            if verbose:
                print(len(pred_boxes), "object(s) found in the image!")
        count_sum += len(pred_boxes)

        if display_res:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for i in range(len(pred_boxes)):
                cv2.rectangle(img,
                              (int(pred_boxes[i][0][0]), int(pred_boxes[i][0][1])),
                              (int(pred_boxes[i][1][0]), int(pred_boxes[i][1][1])),
                              color=(200, 80, 80), thickness=2)
                cv2.putText(img, pred_classes[i],  # contents
                            (int(pred_boxes[i][0][0]), int(pred_boxes[i][0][1])),  # org
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(200, 80, 80), thickness=2)  # text settings
            plt.figure(figsize=[20, 30])
            plt.imshow(img)
            plt.axis("off")
            plt.show()
    return count_sum


if __name__ == "__main__":
    print("Detecting...")
    count_all = 0
    for area in AREAS:
        count_area = detect_object(area, verbose=False)
        print(count_area, "car(s) in the area", area)
        count_all += count_area
    print(count_all, "car(s) in all areas.")