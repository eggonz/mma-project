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
import matplotlib.pyplot as plt
import numpy as np

def get_image_from_jpeg(path):
  img = Image.open(path)
  img = img.resize((224, 224))
  img = img.convert('RGB')
  return img
     
def _compute_embeddings_from_images(path_to_images):
    model_dir = '/content/mma-project/fromage/fromage_model'
    # Load the model
    model = models.load_fromage_for_embeddings(model_dir)
    embeddings = {'embeddings': [], 'paths': []}
    # Open the json file
    image_paths = glob.glob(path_to_images + "/**/*.jpeg", recursive = True)
    for image_path in image_paths:
      print(image_path)
      p = get_image_from_jpeg(path=image_path)
      with torch.no_grad():
        pixel_values = utils.get_pixel_values_for_model(model.model.feature_extractor, p)
        pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
        pixel_values = pixel_values[None, ...]
        visual_embs = model.model.get_visual_embs(pixel_values, mode='retrieval')
      embeddings['embeddings'].append(visual_embs.float().cpu().numpy())
      embeddings['paths'].append(image_path)
    with open('/content/mma-project/fromage/fromage_model/embeddings/funda_sample.pkl', 'wb') as f:
          pickle.dump(embeddings, f)

if __name__ == "__main__":
  _compute_embeddings_from_images('/content/mma-project/funda')

