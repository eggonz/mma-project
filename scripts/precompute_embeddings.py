"""
Script file intended to precompute embeddings for the whole funda dataset.
In principle, not a module nor part of the pipeline.
"""
import os, sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '../src'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

import argparse
import glob
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

import clip
from umap_utils import compute_umap


def get_image_from_jpeg(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img


def _compute_clip_embeddings(images_dir: str, use_gpu: bool = False, resume: int = 0, resume_file: str = None) -> pd.DataFrame:
    """
    Computes CLIP embeddings for the images inside a dataset folder
    """
    # Check if we have already computed the embeddings
    if resume_file is not None and os.path.exists(resume_file):
        df = pd.read_pickle(resume_file)
        embeddings = {'embeddings': df['embeddings'].tolist(), 'paths': df.index.tolist()}
    else:
        embeddings = {'embeddings': [], 'paths': []}

    # Load the model
    clip_model = clip.load_clip_model(gpu=use_gpu)

    # Open the json file
    full_image_paths = glob.glob(images_dir + "/**/*.jpeg", recursive=True)
    full_image_paths = sorted(full_image_paths)
    for full_image_path in tqdm(full_image_paths, desc='Computing CLIP embeddings'):
        id_ = int(full_image_path.split('/')[-2])
        if id_ < resume:
            continue
        img = get_image_from_jpeg(path=full_image_path)
        visual_embs = clip_model.get_visual_emb_for_img(img)
        embeddings['embeddings'].append(visual_embs)

        image_path = os.path.join(*full_image_path.split('/')[-2:])
        embeddings['paths'].append(image_path)

    df = pd.DataFrame(embeddings)
    df.drop_duplicates(subset=['paths'], inplace=True)

    df.set_index('paths', inplace=True)
    return df


def _compute_umap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add umap coords columns to DataFrame
    """
    print('Computing UMAP')
    umap_embeddings = compute_umap(df['embeddings'], n_components=2)
    df['umap_x'] = umap_embeddings[:, 0]
    df['umap_y'] = umap_embeddings[:, 1]
    return df


if __name__ == "__main__":
    """
    python precompute_embeddings.py --images_dir ../data/Funda/funda_images_tiny/ --output_file embeddings.pkl --gpu --start 4291067
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    df = _compute_clip_embeddings(args.images_dir, args.gpu, resume=args.resume, resume_file=args.output_file)
    if os.path.dirname(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_pickle(args.output_file)  # tmp save

    df = _compute_umap(df)
    df.to_pickle(args.output_file)
