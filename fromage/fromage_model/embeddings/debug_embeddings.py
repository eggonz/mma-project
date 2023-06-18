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

def _display_interleaved_outputs(model_outputs, x,  one_img_per_ret=True):
    for output in model_outputs:
        if type(output) == str:
            print(output)
        elif type(output) == list:
            if one_img_per_ret:
                plt.figure(figsize=(3, 3))
                plt.imshow(np.array(output[0]))
                plt.savefig("Figure_{}.png".format(x))
            else:
                fig, ax = plt.subplots(1, len(output), figsize=(3 * len(output), 3))
                for i, image in enumerate(output):
                    image = np.array(image)
                    ax[i].imshow(image)
                    ax[i].set_title(f'Retrieval #{i+1}')
                plt.savefig("Figure_{}.png".format(x))
        elif type(output) == Image.Image:
            plt.figure(figsize=(3, 3))
            plt.imshow(np.array(output))
            plt.savefig("Figure_{}.png".format(x))

def debug_model(path_to_images): 
  model_dir = '/content/mma-project/fromage/fromage_model'
  # Load the model
  model = models.load_fromage(model_dir)
  image_paths = glob.glob(path_to_images + "/*.jpeg")
  x = 0
  for image_path in image_paths[:3]:
    for prompt in ['Similar image [RET]']:
      p = get_image_from_jpeg(path=image_path)
      fig = plt.figure()
      promp = [p, prompt]
      print('Prompt:')
      _display_interleaved_outputs(promp, x)
      x += 1
      print('=' * 30)
      model_outputs, image_idx = model.generate_for_images_and_texts(promp, max_img_per_ret=2)
      # Display outputs.
      print('Model generated outputs:')
      print(image_idx)
      _display_interleaved_outputs(model_outputs, x,  one_img_per_ret=False)   
      x += 1  
       

if __name__ == "__main__":
  debug_model("/content/mma-project/funda/42194016")