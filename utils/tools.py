import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from config import *

classes = ['cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant',
           'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']


def sort_class_extract(datasets):
    """Groups dataset images and annotations by class."""
    datasets_per_class = defaultdict(lambda: defaultdict(list))

    for dataset in datasets:
        for img, target in dataset:
            filename = target['annotation']['filename']
            objects = target['annotation']['object']

            org = {cls: [img] for cls in classes}  # Initialize each class with the image

            for obj in objects:
                class_name = obj["name"]
                org[class_name].append([obj["bndbox"], target['annotation']['size']])

            for class_name, data in org.items():
                if len(data) > 1:  # Ensure there is annotation data
                    datasets_per_class[class_name][filename].append(data)

    return datasets_per_class


def show_new_bdbox(image, labels, color='r', count=0):
    """Displays bounding boxes on an image."""
    xmin, xmax, ymin, ymax = labels
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))  # Adjusting PyTorch tensor format for visualization

    width, height = xmax - xmin, ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Iteration {count}")
    plt.savefig(f"{count}.png", dpi=100)


def extract(index, loader):
    """Extracts images and ground truth bounding boxes from a dataloader."""
    img, ground_truth_boxes = loader[index][0], []
    
    for ex in loader[index]:
        bndbox, size = ex[1]
        xmin = (float(bndbox['xmin']) / float(size['width'])) * 224
        xmax = (float(bndbox['xmax']) / float(size['width'])) * 224
        ymin = (float(bndbox['ymin']) / float(size['height'])) * 224
        ymax = (float(bndbox['ymax']) / float(size['height'])) * 224

        ground_truth_boxes.append([xmin, xmax, ymin, ymax])
    
    return img, ground_truth_boxes


def voc_ap(rec, prec, voc2007=False):
    """Computes Average Precision (AP) using VOC metric."""
    if voc2007:
        return sum(max(prec[rec >= t]) if np.any(rec >= t) else 0 for t in np.arange(0.0, 1.1, 0.1)) / 11.0
    else:
        mrec, mpre = np.concatenate(([0.0], rec, [1.0])), np.concatenate(([0.0], prec, [0.0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        return np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])


def prec_rec_compute(bounding_boxes, gt_boxes, ovthresh):
    """Computes precision and recall based on IoU threshold."""
    tp, fp, npos = np.zeros(len(bounding_boxes)), np.zeros(len(bounding_boxes)), len(bounding_boxes)

    for i, (box1, box2) in enumerate(zip(bounding_boxes, [gt[0] for gt in gt_boxes])):
        xi1, yi1 = max(box1[0], box2[0]), max(box1[2], box2[2])
        xi2, yi2 = min(box1[1], box2[1]), min(box1[3], box2[3])
        inter_area = max((xi2 - xi1) * (yi2 - yi1), 0)
        union_area = (box1[1] - box1[0]) * (box1[3] - box1[2]) + (box2[1] - box2[0]) * (box2[3] - box2[2]) - inter_area
        iou = inter_area / union_area

        tp[i] = 1.0 if iou > ovthresh else 0.0
        fp[i] = 1.0 - tp[i]

    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tp / float(npos), tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return prec, rec


def compute_ap_and_recall(all_bdbox, all_gt, ovthresh):
    """Calculates AP and recall."""
    prec, rec = prec_rec_compute(all_bdbox, all_gt, ovthresh)
    return voc_ap(rec, prec), rec[-1]


def eval_stats_at_threshold(all_bdbox, all_gt, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """Evaluates statistics across different IoU thresholds."""
    stats = {thresh: dict(zip(["ap", "recall"], compute_ap_and_recall(all_bdbox, all_gt, thresh))) for thresh in thresholds}
    return pd.DataFrame.from_records(stats) * 100


class ReplayMemory:
    """Replay Memory for RL algorithms."""
    
    def __init__(self, capacity):
        self.capacity, self.memory, self.position = capacity, [], 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
