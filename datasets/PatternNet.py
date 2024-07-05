import os
import pickle
from re import S

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

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
    "intersection": "inter section",
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
class PatternNet(DatasetBase):

    dataset_dir = "PatternNet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "PatternNet")
        self.split_path = os.path.join(self.dataset_dir, "split_cao_PatternNet.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
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
