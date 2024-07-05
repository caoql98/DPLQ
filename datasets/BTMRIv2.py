import os
import pickle
from re import S

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "no brain tumor",
    "pituitary": "pituitary",
}

@DATASET_REGISTRY.register()
class BTMRIv2(DatasetBase):

    dataset_dir = "BTMRI"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "BTMRI")
        data = self.read_data(self.image_dir)

        super().__init__(train_x=data, test=data)

    def read_data(self, image_dir):
        categories  = listdir_nohidden(image_dir)
        categories.sort()
        items = []
        for label, category in enumerate(categories):
            class_dir = os.path.join(image_dir, category)
            imnames = listdir_nohidden(class_dir)
            if NEW_CNAMES  is not None and category in NEW_CNAMES :
                category = NEW_CNAMES[category]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=category)
                items.append(item)

        return items

