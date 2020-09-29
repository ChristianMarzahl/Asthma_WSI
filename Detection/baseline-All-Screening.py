

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from tqdm import tqdm
from pathlib import Path
import openslide
import pandas as pd
import pickle




from fastai.callbacks.csv_logger import CSVLogger



from object_detection_fastai.helper.object_detection_helper import *
from object_detection_fastai.helper.wsi_loader import *
from object_detection_fastai.loss.RetinaNetFocalLoss import RetinaNetFocalLoss
from object_detection_fastai.models.RetinaNet import RetinaNet
from object_detection_fastai.callbacks.callbacks import BBLossMetrics, BBMetrics, PascalVOCMetric, PascalVOCMetricByDistance

slides_train = list(set([#'BAL Promyk Spray 4.svs',
                         'BAL cent blue Luft 2.svs'
                         ]))

slides_val = list(set(['BAL cent blue Luft 2.svs'
                        #'BAL 1 Spray 2.svs', 
                        # 'BAL Booker Spray 3.svs',
                        # 'BAL Bubi Spray 1.svs', 
                        # 'BAL cent blue Luft 2.svs'
                         ]))

#'Mastzellen', "Makrophagen", "Neutrophile", "Eosinophile", "Lymohozyten"
labels = ["Eosinophile"]



experiment_name = "Asthma-L0-Screening"


annotations_path = Path("Statistics/Asthma_Annotations.pkl")
annotations = pd.read_pickle(annotations_path)

annotations = annotations[annotations["class"].isin(labels)]
annotations_train = annotations[annotations["image_name"].isin(slides_train)]
annotations_val = annotations[annotations["image_name"].isin(slides_val)]




annotations["width"] = [vector["x2"] - vector["x1"] for vector in annotations["vector"]]
annotations["height"] = [vector["y2"] - vector["y1"] for vector in annotations["vector"]]
annotations["scale"] = annotations["width"] / annotations["height"]


annotations.plot.hexbin(x='width', y='height', gridsize=25) #C='scale', 

slides_path = Path("Slides")
files = {slide.name: slide for slide in slides_path.rglob("*.svs") if slide.name in slides_train + slides_val}
files



with open('Statistics/Screening.pickle', 'rb') as handle:
    screening_modes = pickle.load(handle)


def filterAnnotations(image_name, xmin, ymin, screening_modes, patch_x:int = 1024, patch_y:int = 1024):
    
    screening = screening_modes[image_name]
    tiles = screening["screening_tiles"]  
    
    xmin, ymin = max(1, int(xmin - patch_x / 2)), max(1, int(ymin - patch_y / 2))
    
    x_step = int(np.floor(xmin / screening["x_resolution"])) + 1
    y_step = int(np.floor(ymin / screening["y_resolution"])) + 1 
    
    return tiles[str((y_step * screening["x_steps"]) + x_step)]['Screened']

annotations_train["border"] = annotations_train.apply(lambda x: filterAnnotations(x["image_name"], x["vector"]["x1"], x["vector"]["y1"], screening_modes), axis=1)
annotations_val["border"] = annotations_val.apply(lambda x: filterAnnotations(x["image_name"], x["vector"]["x1"], x["vector"]["y1"], screening_modes), axis=1)
annotations_val.head()


class SlideContainerWithScreening(SlideContainer):

    def get_new_train_coordinates(self):
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.y, **{"classes": self.classes, "size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "level": self.level})

        # use default sampling method
        width, height = self.slide.level_dimensions[self.level]
        if len(self.y[0]) == 0:
            return randint(0, width - self.shape[0]), randint(0, height - self.shape[1])
        else:
            # use default sampling method
            class_id = np.random.choice( self.classes, 1)[0]
            ids = (np.array(self.y[1]) == class_id) & (np.array(self.y[2]) == False)
            #  if you can´t fond any ignore screened area
            if np.count_nonzero(ids) == 0:
                ids = (np.array(self.y[1]) == class_id)
            xmin, ymin, _, _ = np.array( self.y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]

            xmin, ymin = max(1, int(xmin - self.shape[0] / 2)), max(1, int(ymin - self.shape[1] / 2))
            xmin, ymin = min(xmin, width - self.shape[0]), min(ymin, height - self.shape[1])

            return xmin, ymin

size = 1024 
level = 0
bs = 16
train_images = bs
val_images = bs


train_files = []
val_files = []


for image_name in annotations_train["image_name"].unique():
    
    annotations = annotations_train[annotations_train["image_name"] == image_name]
    annotations = annotations[annotations["deleted"] == False]
    
    slide_path = files[image_name]
    labels =  list(annotations["class"])
    borders = list(annotations["border"])
    
    bboxes = [[vector["x1"], vector["y1"], vector["x2"], vector["y2"]] for vector in annotations["vector"]]
    
    for label in labels:
        if label not in set(labels):
            bboxes.append([0,0,0,0])
            labels.append(label)
            borders.append(True)

    train_files.append(SlideContainerWithScreening(slide_path, y=[bboxes, labels, borders],  level=level, width=size, height=size))
    
for image_name in annotations_val["image_name"].unique():
    
    annotations = annotations_val[annotations_val["image_name"] == image_name]
    annotations = annotations[annotations["deleted"] == False]
    
    slide_path = files[image_name]
    labels =  list(annotations["class"])
    borders = list(annotations["border"])
    
    bboxes = [[vector["x1"], vector["y1"], vector["x2"], vector["y2"]] for vector in annotations["vector"]]
    
    for label in labels:
        if label not in set(labels):
            bboxes.append([0,0,0,0])
            labels.append(label)
            borders.append(True)

    val_files.append(SlideContainerWithScreening(slide_path, y=[bboxes, labels, borders],  level=level, width=size, height=size))

train_files = list(np.random.choice(train_files, train_images))
valid_files = list(np.random.choice(val_files, val_images))


tfms = get_transforms(do_flip=True,
                      flip_vert=True,
                      #max_rotate=90,
                      #max_lighting=0.0,
                      #max_zoom=1.,
                      #max_warp=0.0,
                      #p_affine=0.5,
                      #p_lighting=0.0,
                      #xtra_tfms=xtra_tfms,
                     )


def get_y_func(x):
    return x.y

train =  ObjectItemListSlide(train_files, path=slides_path)
valid = ObjectItemListSlide(valid_files, path=slides_path)
item_list = ItemLists(slides_path, train, valid)
lls = item_list.label_from_func(get_y_func, label_cls=SlideObjectCategoryList) #
lls = lls.transform(tfms, tfm_y=True, size=size)
data = lls.databunch(bs=bs, collate_fn=bb_pad_collate, num_workers=0)#.normalize()

for temp in data:
    print("")


data.show_batch(rows=2, ds_type=DatasetType.Valid, figsize=(15,15))



scales = [.8, .9, 1, 1.1]
anchors = create_anchors(sizes=[(64, 64), (32,32)], ratios=[1], scales=scales)



fig,ax = plt.subplots(figsize=(15,15))
ax.imshow(image2np(data.valid_ds[0][0].data))

for i, bbox in enumerate(anchors[:len(scales)]):
    bb = bbox.numpy()
    x = (bb[0] + 1) * size / 2 
    y = (bb[1] + 1) * size / 2 
    w = bb[2] * size / 2
    h = bb[3] * size / 2
    
    rect = [x,y,w,h]
    draw_rect(ax,rect)

all_boxes, all_labels = show_anchors_on_images(data, anchors, figsize=(24, 24))


crit = RetinaNetFocalLoss(anchors)

encoder = create_body(models.resnet18, True, -2)
model = RetinaNet(encoder, n_classes=data.train_ds.c, n_anchors=len(scales), sizes=[64, 32], chs=128, final_bias=-4., n_conv=2)

data.train_ds.y.classes[1:]


voc = PascalVOCMetricByDistance(anchors, size, [str(i) for i in data.train_ds.y.classes[1:]], radius=40)
learn = Learner(data, model, loss_func=crit, callback_fns=[BBMetrics], #BBMetrics, ShowGraph
                metrics=[voc])


learn.split([model.encoder[6], model.c5top5])
learn.freeze_to(-2)


learn.fit_one_cycle(3, 1e-3)

