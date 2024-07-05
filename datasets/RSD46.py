import os
import pickle
from re import S

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "Airplane": "Airplane",
    "Airport": "Airport",
    "Aritificial dense forest land": "Aritificial dense forest land",
    "Aritificial sparse forest land": "Aritificial sparse forest land",
    "Bare land": "Bare land",
    "Basketball court": "Basketball court",
    "Blue structured factory building": "Blue structured factory building",
    "Building": "Building",
    "Construction site": "Construction site",
    "Cross river bridge": "Cross river bridge",
    "Crossroads": "Crossroads",
    "Dense tall building": "Dense tall building",
    "Dock": "Dock",
    "Fish pond": "Fish pond",
    "Footbridge": "Footbridge",
    "Graff": "Graff",
    "Grassland": "Grassland",
    "Irregular farmland": "Irregular farmland",
    "Low scatterd building": "Low scatterd building",
    "Medium density scattered building": "Medium density scattered building",
    "Medium density structured building": "Medium density structured building",
    "Natural dense forest land": "Natural dense forest land",
    "Natural sparse forest land": "Natural sparse forest land",
    "Oiltank": "Oiltank",
    "Overpass": "Overpass",
    "Parking_lot": "Parking lot",
    "Plasticgreenhouse": "Plastic greenhouse",
    "Playground": "Playground",
    "Railway": "Railway",
    "Red structured factory building": "Red structured factory building",
    "Refinery": "Refinery",
    "Regular farmland": "Regular farmland",
    "Scattered blue root factory building": "Scattered blue root factory building",
    "Scattered red root factory building": "Scattered red root factory building",
    "Sewage plant-type-one": "Sewage plant type one",
    "Sewage plant-type-two": "Sewage plant type two",
    "Ship": "Ship",
    "Solar power station": "Solar power station",
    "Sparse residential area": "Sparse residential area",
    "Square": "Square",
    "Steelsmelter": "Steel smelter",
    "Storage land": "Storage land",
    "Tennis court": "Tennis court",
    "Thermal power plant": "Thermal power plant",
    "Vegetable plot": "Vegetable plot",
    "Water": "Water",
}

@DATASET_REGISTRY.register()
class RSD46(DatasetBase):

    dataset_dir = "RSD46"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "RSD46")
        self.split_path = os.path.join(self.dataset_dir, "split_cao_RSD46.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data2(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
