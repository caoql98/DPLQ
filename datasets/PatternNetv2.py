import os
import pickle
from re import S

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "airplane": "airplane",
    "baseball_field": "baseball field",
    "basketball_court": "basketball court",
    "beach": "beach",
    "bridge": "bridge",
    "cemetery": "cemetery",
    "chaparral": "chaparral",
    "christmas_tree_farm": "christmas tree farm",
    "closed_road": "closed road",
    "coastal_mansion": "coastal mansion",
    "crosswalk": "crosswalk",
    "dense_residential": "dense residential area",
    "ferry_terminal": "ferry terminal",
    "football_field": "football field",
    "forest": "forest",
    "freeway": "freeway",
    "golf_course": "golf course",
    "harbor": "harbor",
    "intersection": "intersection",
    "mobile_home_park": "mobile home park",
    "nursing_home": "nursing home",
    "oil_gas_field": "oil gas field",
    "oil_well": "oil well",
    "overpass": "overpass",
    "parking_lot": "parking lot",
    "parking_space": "parking space",
    "railway": "railway",
    "river": "river",
    "runway": "runway",
    "runway_marking": "runway marking",
    "shipping_yard": "shipping yard",
    "solar_panel": "solar panel",
    "sparse_residential": "sparse residential area",
    "storage_tank": "storage tank",
    "swimmimg_pool": "swimmig pool",
    "tennis_court": "tennis court",
    "transformer_station": "transformer station",
    "wastewater_treatment_plant": "wastewater treatment plant",
}

@DATASET_REGISTRY.register()
class PatternNetv2(DatasetBase):

    dataset_dir = "PatternNet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "PatternNet")
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

