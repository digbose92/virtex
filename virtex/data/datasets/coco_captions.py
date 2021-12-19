from collections import defaultdict
import json
import os
from typing import Dict, List

import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

class CocoCaptionsDataset(Dataset):
    r"""
    A PyTorch dataset to read COCO Captions dataset and provide it completely
    unprocessed. This dataset is used by various task-specific datasets
    in :mod:`~virtex.data.datasets` module.

    Args:
        data_root: Path to the COCO dataset root directory.
        split: Name of COCO 2017 split to read. One of ``{"train", "val"}``.
    """

    def __init__(self, data_root: str, split: str):

        # Get paths to image directory and annotation file.
        image_dir = os.path.join(data_root, f"{split}2017")
        captions = json.load(
            open(os.path.join(data_root, "annotations", f"captions_{split}2017.json"))
        )
        # Collect list of captions for each image.
        captions_per_image: Dict[int, List[str]] = defaultdict(list)
        for ann in captions["annotations"]:
            captions_per_image[ann["image_id"]].append(ann["caption"])

        # Collect image file for each image (by its ID).
        image_filepaths: Dict[int, str] = {
            im["id"]: os.path.join(image_dir, im["file_name"])
            for im in captions["images"]
        }
        # Keep all annotations in memory. Make a list of tuples, each tuple
        # is ``(image_id, file_path, list[captions])``.
        self.instances = [
            (im_id, image_filepaths[im_id], captions_per_image[im_id])
            for im_id in captions_per_image.keys()
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_id, image_path, captions = self.instances[idx]

        # shape: (height, width, channels), dtype: uint8
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return {"image_id": image_id, "image": image, "captions": captions}

class ArtemisCaptionsDataset(Dataset):

    """Pytorch dataset to read Artemis captions and generate the data 
       Args:
            root_data_dir: Path to Artemis data 
            split: "train" or "val"
    """

    def __init__(self, root_data_dir: str, image_data_dir:str, split: str):

        self.root_data_dir=root_data_dir
        self.image_data_dir=image_data_dir
        self.split=split 

        #preprocessed data
        self.caption_file=os.path.join(self.root_data_dir,"artemis_preprocessed.csv")
        self.caption_data=pd.read_csv(self.caption_file)

        #split data
        self.caption_data_split=self.caption_data[self.caption_data['split']==split]

    def __len__(self):
        return len(self.caption_data_split)

    def __getitem__(self, idx: int):
        
        #artsyle and painting 
        artstyle=self.caption_data_split['art_style'].iloc[idx]
        painting=self.caption_data_split['painting'].iloc[idx]
        captions=self.caption_data_split['utterance'].iloc[idx]

        artstyle_subfolder=os.path.join(self.image_data_dir,artstyle)
        painting_file=os.path.join(artstyle_subfolder,painting+".jpg")
        #print(painting_file)
        image = cv2.imread(painting_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_id=artstyle+"_"+painting

        return {"image_id": image_id, "image": image, "captions": captions}


# root_data_dir="/home/dbose_usc_edu/data/artemis/preprocessed_data/"
# image_data_dir="/data/wikiart_dataset"
# artemis_ds=ArtemisCaptionsDataset(root_data_dir=root_data_dir,
#                                         image_data_dir=image_data_dir,
#                                         split='train')

# artemis_dl=DataLoader(artemis_ds,batch_size=1,shuffle=False)
# dict_artemis_data=next(iter(artemis_dl))
#print(dict_artemis_data)

