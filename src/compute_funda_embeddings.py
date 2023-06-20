import argparse
import glob
import os

import pandas as pd
import torch
from PIL import Image

from fromage import models
from fromage import utils


def get_image_from_jpeg(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img


def _compute_embeddings_from_images(images_dir, model_dir, output_file):
    # Load the model
    model = models.load_fromage_for_embeddings(model_dir)
    embeddings = {'embeddings': [], 'paths': []}
    # Open the json file
    full_image_paths = glob.glob(images_dir + "/**/*.jpeg", recursive=True)
    for full_image_path in full_image_paths:
        #print(full_image_path)
        p = get_image_from_jpeg(path=full_image_path)
        with torch.no_grad():
            pixel_values = utils.get_pixel_values_for_model(model.model.feature_extractor, p)
            pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
            pixel_values = pixel_values[None, ...]
            visual_embs = model.model.get_visual_embs(pixel_values, mode='retrieval')
        embeddings['embeddings'].append(visual_embs.float().cpu().numpy())
        image_path = os.path.join(*full_image_path.split('/')[-2:])
        embeddings['paths'].append(image_path)
    df = pd.DataFrame(embeddings)
    df.set_index('paths', inplace=True)
    df.to_pickle(output_file)


if __name__ == "__main__":
    """
    python compute_funda_embeddings.py --images_dir ../funda/ --model_dir ../fromage/fromage_model/ --output_file embeddings.pkl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    _compute_embeddings_from_images(args.images_dir, args.model_dir, args.output_file)
