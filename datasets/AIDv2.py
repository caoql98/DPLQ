import os
import pickle
from re import S

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "Airport": "Airport",
    "BareLand": "BareLand",
    "BaseballField": "Baseball Field",
    "Beach": "Beach",
    "Bridge": "Bridge",
    "Center": "Center Area",
    "Church": "Church",
    "Commercial": "Commercial Area",
    "Dense Residential": "Dense Residential Area",
    "Desert": "Desert",
    "Farmland": "Farmland",
    "Forest": "Forest",   
    "Industrial": "Industrial Area",   
    "Meadow": "Meadow",   
    "MediumResidential": "Medium Residential Area",  
    "Mountain": "Mountain",   
    "Park": "Park",   
    "Parking": "Parking",   
    "Playground": "Playground",  
    "Pond": "Pond",
    "Port": "Port",
    "RailwayStation": "Railway Station",
    "Resort": "Resort",
    "River": "River",
    "School": "School",
    "SparseResidential": "Sparse Residential Area",
    "Square": "Square",
    "Stadium": "Stadium",
    "StorageTanks": "Storage Tanks",
    "Viaduct": "Viaduct",    
}

@DATASET_REGISTRY.register()
class AIDv2(DatasetBase):

    dataset_dir = "AID"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "AID")
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

