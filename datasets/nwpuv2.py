import os
import pickle
from re import S

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "airplane": "airplane",
    "airport": "airport",
    "baseball_diamond": "baseball diamond",
    "basketball_court": "basketball court",
    "beach": "beach",
    "bridge": "bridge",
    "chaparral": "chaparral",
    "church": "church",
    "circular_farmland": "circular farmland",
    "cloud": "cloud",
    "commercial_area": "commercial area",
    "dense_residential": "dense residential area",
    "desert": "desert",
    "forest": "forest",
    "freeway": "freeway",
    "golf_course": "golf course",
    "ground_track_field": "ground track field",
    "harbor": "harbor",
    "industrial_area": "industrial area",
    "intersection": "intersection",
    "island": "island",
    "lake": "lake",   
    "meadow": "meadow",   
    "medium_residential": "medium residential area",          
    "mobile_home_park": "mobile home park",    
    "mountain": "mountain",    
    "overpass": "overpass",    
    "palace": "palace",    
    "parking_lot": "parking lot",    
    "railway": "railway",    
    "railway_station": "railway station",    
    "rectangular_farmland": "rectangular farmland", 
    "river": "river",   
    "roundabout": "roundabout",    
    "runway": "runway",           
    "sea_ice": "sea ice",   
    "ship": "ship",   
    "snowberg": "snowberg",   
    "sparse_residential": "sparse residential area",  
    "stadium": "stadium",  
    "storage_tank": "storage tank",  
    "tennis_court": "tennis court",  
    "terrace": "terrace",  
    "thermal_power_station": "thermal power station",   
    "wetland": "wetland",         
}

@DATASET_REGISTRY.register()
class nwpuv2(DatasetBase):

    dataset_dir = "nwpu"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "nwpu")
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

