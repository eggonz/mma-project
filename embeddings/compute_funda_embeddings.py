import sys
sys.path.append("./fromage")
import os
from fromage import models
from fromage import utils
import glob
import torch
from PIL import Image
import json
import pickle
import glob



def get_image_from_jpeg(path):
  img = Image.open(path)
  img = img.resize((224, 224))
  img = img.convert('RGB')
  return img



def _compute_embeddings_from_images(args, path_to_images):
    model_dir = '../fromage/fromage_model/'
    # Load the model
    model = models.load_fromage(model_dir)
    embeddings = {'embeddings': [], 'paths': []}
    # Open the json file
    image_paths = glob.glob("*.jpeg", root_dir = path_to_images)
    for image_path in image_paths:
          print(image_path)
          p = get_image_from_jpeg(path=image_path)
          pixel_values = utils.get_pixel_values_for_model(model.model.feature_extractor, p)
          pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
          pixel_values = pixel_values[None, ...]
          visual_embs = model.model.get_visual_embs(pixel_values, mode='retrieval')
          embeddings['embeddings'].append(visual_embs)
          # This means that we will run the model from the /src directory!!

_compute_embeddings_from_images(None)