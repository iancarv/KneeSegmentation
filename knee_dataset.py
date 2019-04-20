"""
Mask R-CNN
Configurations and data loading code for MS Knee.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained Knee weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download Knee dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run Knee evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import glob
import h5py

# import zipfile
# import urllib.request
# import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class KneeConfig(Config):
    """Configuration for training on MS Knee.
    Derives from the base Config class and overrides values specific
    to the Knee dataset.
    """
    # Give the configuration a recognizable name
    NAME = "knee"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)

    NUM_CLASSES = 1 + 6


############################################################
#  Dataset
############################################################

class KneeDataset(utils.Dataset):
    def load_knee(self, dataset_dir, subset, class_ids=None,
                  auto_download=False):

        class_names = ['Femoral Cart.', 'Medial Tibial Cart.', 'Lateral Tibial Cart.', 'Patellar Cart.', 'Lateral Meniscus', 'Medial Meniscus']
        if class_ids is None:
            class_ids = list(range(len(class_names)))

        for class_id in class_ids:
            self.add_class("shapes", class_id, class_names[class_id])

        assert subset in ['train', 'minitrain', 'valid', 'minivalid']
        
        
        if 'mini' in subset:
            dataset_dir = os.path.join(dataset_dir, 'valid')
            img_paths = [f for f in glob.glob(dataset_dir + "/*.im")]
            if 'train' in subset:
                img_paths = img_paths[:2*10]
            else:
                img_paths = img_paths[2*10:]
        else:
            dataset_dir = os.path.join(dataset_dir, subset)
            img_paths = [f for f in glob.glob(path + "/*.im")]


        for img_path in img_paths:
            base_id =  os.path.splitext(os.path.basename(img_path))[0]

            with h5py.File(img_path,'r') as hf:
                img = np.array(hf['data'])
            n_imgs = img.shape[-1]
            w, h = img.shape[:2]
            for i in range(n_imgs):
                img_id = base_id + '_im' + str(i)
                self.add_image(
                    "knee",
                    image_id=img_id,
                    path=img_path,
                    width=w, height=h,
                    slice=i)
            
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        sl = info["slice"]
        img_path = info["path"]

        with h5py.File(img_path,'r') as hf:
                image = np.array(hf['data'])

        image = image[:,:,sl]
        return image


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        if info["source"] != "knee":
            return super(CocoDataset, self).load_mask(image_id)

        class_ids = np.unique([c["id"] for c in self.class_info])

        sl = info["slice"]
        img_path = info["path"]

        masks_path = img_path.replace('im', 'seg')

        with h5py.File(masks_path,'r') as hf:
                masks = np.array(hf['data'])


        # Pack instance masks into an array
        if class_ids is not None:
            masks = masks[:,:,sl,class_ids].astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return masks, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the Knee Website."""
        info = self.image_info[image_id]
        if info["source"] == "knee":
            return info["id"]
        else:
            super(CocoDataset, self).image_reference(image_id)
