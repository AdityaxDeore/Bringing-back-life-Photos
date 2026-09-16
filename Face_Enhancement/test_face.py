# Copyright (c) Developers.
# Licensed under the MIT License.

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
import torchvision.utils as vutils
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# - This initializes the options for the Pix2Pix generative AI model used specifically for Face Enhancement.
opt = TestOptions().parse()

# - We set up a 'dataloader' which automatically loads the cropped faces from the hard drive into memory in small batches.
dataloader = data.create_dataloader(opt)

# - We create the AI model (SPADE generator) and set it to 'eval' (evaluation mode) since we are not training it.
model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)


single_save_url = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir, "each_img")


if not os.path.exists(single_save_url):
    os.makedirs(single_save_url)


# - We loop through all the batches of cropped faces.
# - The model performs 'inference' to generate a super-high-resolution, sharp version of the old face.
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode="inference")

    img_path = data_i["path"]

    # - We take the newly generated sharp faces and save them back to the hard drive as image files.
    # - These high-quality faces will later be stitched back onto the original photo in the final blending stage.
    for b in range(generated.shape[0]):
        img_name = os.path.split(img_path[b])[-1]
        save_img_url = os.path.join(single_save_url, img_name)

        vutils.save_image((generated[b] + 1) / 2, save_img_url)

