from object_detection_fastai.helper.wsi_loader import *



class ScreeningSlideContainer(SlideContainer):

    def __init__(self, file: Path, y, level: int=0, width: int=256, height: int=256, sample_func: callable=None, screenedTiles: dict=None):

        super().__init__(file, y, level, width, height, sample_func)
        self.screenedTiles = screenedTiles

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
            ids = np.array( self.y[1]) == class_id

            
            xmin, ymin, _, _ = np.array( self.y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]

            xmin, ymin = max(1, int(xmin - self.shape[0] / 2)), max(1, int(ymin - self.shape[1] / 2))
            xmin, ymin = min(xmin, width - self.shape[0]), min(ymin, height - self.shape[1])

            return xmin, ymin
    
