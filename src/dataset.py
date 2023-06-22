import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


class FundaEmbeddings:
    def __init__(self, embeddings_path: str):
        """
        This class is used to load the embeddings from a pickle file and to retrieve them
        The pickle file should contain a dataframe with the following columns:
        - embeddings: the embeddings
        - umap_x: the x coordinate of the umap projection (optional)
        - umap_y: the y coordinate of the umap projection (optional)
        """
        df = pd.read_pickle(embeddings_path)
        self.embeddings = df['embeddings']
        if 'umap_x' in df.columns:
            self.umap_x = df['umap_x']
            self.umap_y = df['umap_y']

    def get_embedding(self, image_path: str) -> np.ndarray:
        """
        Returns the embedding of the image at the given path
        """
        return self.embeddings.loc[image_path].values[0]

    def get_umap(self, image_path: str) -> tuple[float, float]:
        """
        Returns the umap coordinates of the image at the given path
        """
        if 'umap_x' not in self.__dict__:
            raise ValueError('UMap not computed')
        x = self.umap_x.loc[image_path].values[0]
        y = self.umap_y.loc[image_path].values[0]
        return x, y

    def __len__(self) -> int:
        return len(self.embeddings)


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

    def save_as_csv(self, path: str) -> None:
        """
        Saves the dataframe as a csv file
        """
        self.df.to_csv(path)

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
    fd = FundaDataset('../data/Funda/Funda/ads.jsonlines', '../data/Funda/images')
    fd.save_as_csv('../data/Funda/Funda/ads.csv')
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
    fe = FundaEmbeddings('../data/embeddings/funda_sample.pkl')
    print(len(fe))
    emb = fe.get_embedding('42194072/image1.jpeg')
    print(type(emb), emb.shape)
    ux, uy = fe.get_umap('42194072/image1.jpeg')
    print(type(ux), type(uy))


