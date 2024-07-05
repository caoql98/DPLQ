import os
import pickle
from re import S

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "GliomaT1": "Glioma T1",
    "GliomaT1C+": "Glioma T1 C+",
    "GliomaT2": "Glioma T2",
    "MeningiomaT1": "Meningioma T1",
    "MeningiomaT1C+": "Meningioma T1 C+",
    "MeningiomaT2": "Meningioma T2",
    "NeurocitomaT1": "Neurocitoma T1",
    "NeurocitomaT1C+": "Neurocitoma T1 C+",
    "NeurocitomaT2": "Neurocitoma T2",
    "NORMALT1": "Normal T1",
    "NORMALT2": "Normal T2",
    "OutrosT1": "Outros T1",
    "OutrosT1C+": "Outros T1 C+",
    "OutrosT2": "Outros T2",
    "SchwannomaT1": "Schwannoma T1",
    "SchwannomaT1C+": "Schwannoma T1 C+",
    "SchwannomaT2": "Schwannoma T2",
}

@DATASET_REGISTRY.register()
class TMR17(DatasetBase):

    dataset_dir = "TMR17"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "TMR17")
        self.split_path = os.path.join(self.dataset_dir, "split_cao_TMR17.json")
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
