

import numpy as np
from tqdm import tqdm
from pathlib import Path
import openslide
import pandas as pd
import cv2

import torch
import torchvision

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data import DataLoader, Dataset

from object_detection_fastai.helper.wsi_loader import *

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from object_detection_fastai.helper.BoundingBox import BoundingBox
from object_detection_fastai.helper.BoundingBoxes import BoundingBoxes
from object_detection_fastai.helper.Evaluator import *
from object_detection_fastai.helper.utils import *



class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SlideContainerWithY(SlideContainer):

    def get_patch_y(self,  x: int=0, y: int=0):

        bboxes, labels = self.y

        bboxes = np.array([box for box in bboxes]) if len(np.array(bboxes).shape) == 1 else  np.array(bboxes)
        labels = np.array(labels)

        if len(labels) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y

            bb_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2
            bb_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2

            ids = ((bboxes[:, 0] + bb_widths) > 0)                       & ((bboxes[:, 1] + bb_heights) > 0)                       & ((bboxes[:, 2] - bb_widths) < self.width)                       & ((bboxes[:, 3] - bb_heights) < self.height)

            bboxes = bboxes[ids]
            bboxes = np.clip(bboxes, 0, max(self.height,self.width))
            #bboxes = bboxes[:, [1, 0, 3, 2]]

            labels = labels[ids]
        
        if len(labels) == 0:
            labels = np.array([0])
            bboxes = np.array([[0, 0, 1, 1]])

        return [bboxes, labels]


class WSIDataset(Dataset):
    
    def __init__(self, dataframe: list, label_id_lookup:dict, transforms=None):
        super().__init__()

        self.image_ids = range(len(dataframe))
        self.items = dataframe
        self.transforms = transforms
        self.label_id_lookup = label_id_lookup

    def __getitem__(self, index: int):

        slide_container = self.items[index]

        xmin, ymin = slide_container.get_new_train_coordinates()
        image = slide_container.get_patch(x=xmin, y=ymin) / 255.0

        bboxes, labels = slide_container.get_patch_y(x=xmin, y=ymin)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.as_tensor([self.label_id_lookup[i] for i in labels], dtype=torch.int64)
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)


        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).float()

        return image, target, index


    def __len__(self) -> int:
        return len(self.items)

# Albumentations
def get_train_transform():
    return A.Compose([
        A.Flip(),
        #A.OneOf([
        #    A.Blur(blur_limit=(15, 15)),
        #    A.JpegCompression(quality_lower=59, quality_upper=60),
        #    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True)
        #], p=0.25)
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

slides_train = list(set(['BAL Promyk Spray 4.svs',
                         'BAL AIA Blickfang Luft.svs']))

slides_val = list(set(['BAL 1 Spray 2.svs', 
                         'BAL Booker Spray 3.svs',
                         'BAL Bubi Spray 1.svs', 
                         'BAL cent blue Luft 2.svs']))

experiment_name = "Asthma-L0"
labels = ['Lymohozyten']
label_id_lookup = {label:id+1 for id, label in enumerate(labels)}
id_label_lookup = {id+1:label for id, label in enumerate(labels)}

annotations_path = Path("Statistics/Asthma_Annotations.pkl")
annotations = pd.read_pickle(annotations_path)

annotations = annotations[annotations["class"].isin(labels)]
annotations_train = annotations[annotations["image_name"].isin(slides_train)]
annotations_val = annotations[annotations["image_name"].isin(slides_val)]
annotations_val.head()




slides_path = Path("Slides")
files = {slide.name: slide for slide in slides_path.rglob("*.svs") if slide.name in slides_train + slides_val}

size = 1024 
level = 0
bs = 8
train_images = 32
val_images = 32




train_files = []
val_files = []

for image_name in annotations_train["image_name"].unique():
    
    annotations = annotations_train[annotations_train["image_name"] == image_name]
    annotations = annotations[annotations["deleted"] == False]
    
    slide_path = files[image_name]
    labels =  list(annotations["class"])
    bboxes = [[vector["x1"], vector["y1"], vector["x2"], vector["y2"]] for vector in annotations["vector"]]
    
    for label in labels:
        if label not in set(labels):
            bboxes.append([0,0,0,0])
            labels.append(label)

    train_files.append(SlideContainerWithY(slide_path, y=[bboxes, labels],  level=level, width=size, height=size))
    
for image_name in annotations_val["image_name"].unique():
    
    annotations = annotations_val[annotations_val["image_name"] == image_name]
    annotations = annotations[annotations["deleted"] == False]
    
    slide_path = files[image_name]
    labels =  list(annotations["class"])
    bboxes = [[vector["x1"], vector["y1"], vector["x2"], vector["y2"]] for vector in annotations["vector"]]
    
    for label in labels:
        if label not in set(labels):
            bboxes.append([0,0,0,0])
            labels.append(label)

    val_files.append(SlideContainerWithY(slide_path, y=[bboxes, labels],  level=level, width=size, height=size))

train_files = list(np.random.choice(train_files, train_images))
valid_files = list(np.random.choice(val_files, val_images))


def collate_fn(batch):
    return tuple(zip(*batch))


train_dataset = WSIDataset(train_files, label_id_lookup, get_train_transform())
valid_dataset = WSIDataset(valid_files, label_id_lookup, get_valid_transform())

train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280


backbone



anchor_generator = AnchorGenerator(sizes=((40, 50, 60),), aspect_ratios=((1.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],output_size=7,sampling_ratio=2)


model = FasterRCNN(backbone,
                   num_classes=len(labels)+1,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)




model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None


loss_hist = Averager()
itr = 1
num_epochs = 5

metrics = []
for epoch in range(num_epochs):
    loss_hist.reset()
    
    model.train()
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device, dtype=torch.float) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()


    model.eval()

    evaluator = Evaluator()
    boundingBoxes = BoundingBoxes()
    cpu_device = torch.device("cpu")
    ap = 'AP'
    for images, targets, image_ids in valid_data_loader:
        
        images = list(image.to(device, dtype=torch.float) for image in images)
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs  = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for target, output in zip(targets, outputs):

            for box, cla in zip(np.array(target["boxes"]), np.array(target["labels"])):
                temp = BoundingBox(imageName=str(int(targets[0]["image_id"])), classId=id_label_lookup[cla], x=box[0], y=box[1],
                               w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute,
                               bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2, imgSize=(size,size))

                boundingBoxes.addBoundingBox(temp)


            for box, cla, scor in zip(np.array(output["boxes"].detach()), np.array(output["labels"].detach()), np.array(output["scores"].detach())):
                temp = BoundingBox(imageName=str(int(targets[0]["image_id"])), classId=id_label_lookup[cla], x=box[0], y=box[1],
                                   w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute, classConfidence=scor,
                                   bbType=BBType.Detected, format=BBFormat.XYX2Y2, imgSize=(size, size))

                boundingBoxes.addBoundingBox(temp)

    metricsPerClass = evaluator.GetPascalVOCMetrics(boundingBoxes, IOUThreshold=0.3)
    metric = {"sum": max(sum([mc[ap] for mc in metricsPerClass]) / len(metricsPerClass), 0), "epoch": epoch}

    for mc in metricsPerClass:
        metric['{}-{}'.format(ap, mc['class'])] = max(mc[ap], 0)

    metrics.append(metric)
    print(f"Epoch #{epoch} metrics: {metric}")   