import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


class FundaEmbeddings:
    def __init__(self, embeddings_path: str):
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)

        # TODO tmp: until we fix df format and relative paths
        def fn(x):
            return os.path.join(*x.split('/')[-2:])
        df['paths'] = df['paths'].apply(fn)
        df.set_index('paths', inplace=True)

        self.embeddings = df

    def __getitem__(self, image_path: str) -> np.ndarray:
        emb = self.embeddings.loc[image_path].values[0]
        return np.array(emb)


class FundaDataset:
    def __init__(self, jsonlines_path: str, images_dir: str):
        self.images_dir = images_dir
        self._load_df(jsonlines_path)
        self._format_df()

    def _load_df(self, jsonlines_path: str) -> None:
        """
        Loads dataframe from a jsonlines file. Sets the index to be `funda_identifier`
        """
        self.df = pd.read_json(jsonlines_path, lines=True)
        self.df.set_index('funda_identifier', inplace=True)

    def _format_df(self) -> None:
        """
        Performs formatting on the dataframe. This includes:
        - filtering the price
        - converting the geolocation to a tuple
        """

        # price
        def filter_price(df):
            """
            Filters the items to only keep the ones that contain the price. the price in € is converted to int
            """
            # filter rows with regex
            regex = re.compile(r'€\s*(?P<price>\d{1,3}([,\.]\d{3})*)\s*(k\.k\.|v\.o\.n\.)')
            df = df[df['price'].apply(lambda x: regex.match(x) is not None)].copy()
            # extract price
            df['price'] = df['price'].map(lambda x: regex.match(x).groupdict()['price'])
            # remove commas
            df['price'] = df['price'].map(lambda x: x.replace(',', '').replace('.', ''))
            # convert to int
            df['price'] = df['price'].astype(int)
            return df
        self.df = filter_price(self.df)

        # geolocation
        def geolocation_as_coord(x):
            """converts the geolocation to a tuple (lat, lon)"""
            return float(x['lat']), float(x['lon'])
        self.df['geolocation'] = self.df['geolocation'].apply(geolocation_as_coord)

    def get_images(self, funda_id: int) -> dict[str, Image.Image]:
        """
        Get the PIL.Images for a house with a given funda_id
        The images are returned as a dict of img_path->PIL.Image maps
        The dict can have a variable length from 1 to 5, depending on the number of images available for each house
        """
        paths = self.df.loc[funda_id, 'images_paths']
        images = {path: Image.open(os.path.join(self.images_dir, str(path))) for path in paths}
        return images

    def get_coords(self, funda_id: int) -> tuple[float, float]:
        """
        Get the coordinates for a house with a given funda_id
        The output is a tuple of two floats (lat, lon)
        """
        return self.df.loc[funda_id, 'geolocation']

    def get_data(self, funda_id: int):
        """
        Get the coordinates for a house with a given funda_id
        It returns all the relevant data for each house
        # TODO specify necessary data
        """
        return self.df.loc[funda_id]


if __name__ == '__main__':
    # TEST CODE
    fd = FundaDataset('../data/Funda/Funda/ads.jsonlines', 'funda')
    print(fd.df.head())
    print()
    print('VERIFY FORMATTING')
    print(fd.df['images_paths'].head())
    print(fd.df['price'].head())
    print(fd.df['geolocation'].head())
    print()
    print('VERIFY ATTRIBUTES')
    print(fd.get_images(42194092))
    print(fd.get_coords(42194092))
    print()
    print('VERIFY IMAGES')
    imgs = fd.get_images(42194092)
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for i, ax in enumerate(axs):
        if i >= len(imgs):
            break
        ax.imshow(list(imgs.values())[i])
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    print('VERIFY EMBEDDINGS')
    emb = FundaEmbeddings('../data/embeddings/funda_sample.pkl')
    print(emb.embeddings.head())
    print(emb['42194072/image1.jpeg'])
    print(type(emb['42194072/image1.jpeg']))
    print(emb['42194072/image1.jpeg'].shape)


