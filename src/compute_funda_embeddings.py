"""
Script file intended to precompute embeddings for the whole funda dataset.
In principle, not a module nor part of the pipeline.
"""
import argparse
import glob
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

import clip


def get_image_from_jpeg(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img


def _compute_embeddings_from_images(images_dir, output_file):
    # Load the model
    clip_model = clip.load_clip_model()
    embeddings = {'embeddings': [], 'paths': []}
    # Open the json file
    full_image_paths = glob.glob(images_dir + "/**/*.jpeg", recursive=True)
    for full_image_path in tqdm(full_image_paths):
        img = get_image_from_jpeg(path=full_image_path)
        visual_embs = clip_model.get_visual_emb_for_img(img)
        embeddings['embeddings'].append(visual_embs)

        image_path = os.path.join(*full_image_path.split('/')[-2:])
        embeddings['paths'].append(image_path)

    df = pd.DataFrame(embeddings)
    df.set_index('paths', inplace=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_pickle(output_file)


if __name__ == "__main__":
    """
    python compute_funda_embeddings.py --images_dir ../data/Funda/funda_images_tiny/ --output_file embeddings.pkl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    _compute_embeddings_from_images(args.images_dir, args.output_file)
