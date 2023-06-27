from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import umap_utils


class FundaPrecomputedEmbeddings:
    def __init__(self, df: pd.DataFrame):
        """
        This class is used to load the embeddings from a pickle file and to retrieve them
        Use FundaEmbeddings.from_pickle to load the embeddings from a pickle file

        The pickle file should contain a dataframe with the following columns:
        - embeddings: the embeddings
        - umap_x: the x coordinate of the umap projection (optional)
        - umap_y: the y coordinate of the umap projection (optional)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df should be a pandas dataframe')
        self.df = df

    def clone(self) -> FundaPrecomputedEmbeddings:
        """
        Returns a clone of the current FundaEmbeddings object
        """
        return FundaPrecomputedEmbeddings(self.df.copy())

    @classmethod
    def from_pickle(cls, embeddings_path: str) -> FundaPrecomputedEmbeddings:
        """
        Loads a FundaEmbeddings from a pickle file
        """
        df = pd.read_pickle(embeddings_path)
        return cls(df)

    def get_embedding(self, image_path: str) -> np.ndarray:
        """
        Returns the embedding of the image at the given path
        """
        return self.df['embeddings'].loc[image_path]

    def get_umap(self, image_path: str) -> tuple[float, float]:
        """
        Returns the umap coordinates of the image at the given path
        """
        if 'umap_x' not in self.df.columns or 'umap_y' not in self.df.columns:
            raise ValueError('UMap not computed')
        x = self.df['umap_x'].loc[image_path]
        y = self.df['umap_y'].loc[image_path]
        return x, y

    def get_all_embeddings(self) -> np.ndarray:
        """
        Returns all the embeddings
        """
        return np.stack(self.df['embeddings'].values)

    def get_all_umaps(self) -> np.ndarray:
        """
        Returns all the umap coordinates
        """
        if 'umap_x' not in self.df.columns or 'umap_y' not in self.df.columns:
            raise ValueError('UMap not computed')
        return np.stack([self.df['umap_x'].values, self.df['umap_y'].values]).T

    def __len__(self) -> int:
        return len(self.df)

    def filter(self, selected_ids: list[int]) -> FundaPrecomputedEmbeddings:
        """
        Filters the embeddings using a list of selected ids
        Returns a new FundaEmbeddings object
        """
        def is_selected(x: str) -> bool:
            """extracts the id from the image path and checks if it is in the selected ids"""
            return int(x.split('/', maxsplit=1)[0]) in selected_ids
        idxs = self.df.index.filter(is_selected).values
        df = self.df.loc[idxs].copy()
        return FundaPrecomputedEmbeddings(df)

    def recompute_umap(self):
        umap_embeddings = umap_utils.compute_umap(self.df['embeddings'], n_components=2)
        self.df['umap_x'] = umap_embeddings[:, 0]
        self.df['umap_y'] = umap_embeddings[:, 1]


class FundaDataset:
    def __init__(self, df: pd.DataFrame, images_dir: str):
        """
        This class is used to load the dataset from a jsonlines file and to retrieve the data
        Use FundaDataset.from_jsonlines to load the dataset from a jsonlines file
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df should be a pandas dataframe')
        self.df = df
        self.images_dir = images_dir

    @classmethod
    def from_jsonlines(cls, jsonlines_path: str, images_dir: str) -> FundaDataset:
        """
        Loads a FundaDataset from a jsonlines file
        """
        df = cls._load_jsonlines(jsonlines_path)
        return cls(df, images_dir)

    def clone(self) -> FundaDataset:
        """
        Returns a clone of the current FundaDataset object
        """
        return FundaDataset(self.df.copy(), self.images_dir)

    @staticmethod
    def _load_jsonlines(jsonlines_path: str) -> pd.DataFrame:
        """
        Loads dataframe from a jsonlines file. Sets the index to be `funda_identifier`
        """
        df = pd.read_json(jsonlines_path, lines=True)
        df.set_index('funda_identifier', inplace=True)
        FundaDataset._format_df(df)
        return df

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
    
    def export_pickle(self, path='./data', remove=['crawl_time', 'location_part2', 'in_the_neighborhood']):
        """
        Export the DataFrame in a pickle file
        """
        export_df = self.df.copy()
        export_df = export_df.drop(columns=remove)

        mapper = {'location_part1': 'address', 'location_part3': 'neighborhood'}
        export_df = export_df.rename(columns=mapper)

        export_df.to_pickle(f'{path}/funda_processed.pkl')
        print(f'DataFrame successfully exported to {path}')

    def filter(self, filter_expression: str) -> FundaDataset:
        """
        Filters the dataset using a pandas query expression
        Returns a new FundaDataset object
        """
        return FundaDataset(self.df.filter(filter_expression, axis=0), self.images_dir)
    
    @staticmethod
    def _format_df(dataframe) -> None:
        """
        Performs formatting on the dataframe. This includes:
        - filtering the price
        - converting the geolocation to a tuple
        """
        def extract_features(df):
            df = df.drop(index=df.loc[42119366].name) # to be changed: replace all values with nan instead or translate
            for i, house in enumerate(df['features']):
                if 'asking price per m²' in house['transfer of ownership']:
                    price = house['transfer of ownership']['asking price per m²'].split(' ')[1]
                    regex = re.compile(r',')
                    price = re.sub(regex, '',price)
                    df.loc[df.index[i], 'price per m²'] = price
                if 'vve (owners association) contribution' in house['transfer of ownership']:
                    df.loc[df.index[i], 'vve contribution'] = house['transfer of ownership']['vve (owners association) contribution']
                if 'year of construction' in house['construction']:
                    df.loc[df.index[i], 'construction year'] = house['construction']['year of construction']
                if 'number of rooms' in house['layout']:
                    cell = house['layout']['number of rooms'].split(' ')
                    df.loc[df.index[i], 'no. rooms'] = cell[0]
                    if len(cell) > 2:
                        bedrooms = cell[2][1]
                        df.loc[df.index[i], 'no. bedrooms'] = bedrooms                
                if 'number of bath rooms' in house['layout']:
                    df.loc[df.index[i], 'no. bathrooms'] = house['layout']['number of bath rooms']
                if 'number of stories' in house['layout']:
                    df.loc[df.index[i], 'no. stories'] = house['layout']['number of stories']
                if 'energy' in house:
                    if 'energy label' in house['energy']:
                        df.loc[df.index[i], 'energy label'] = house['energy']['energy label']
                
                keys = ['kind of house', 'type apartment', 'type of property', 'type of apartment']
                type_key = list(set(keys) & set(house['construction'].keys()))
                if len(type_key) != 0:
                    df.loc[df.index[i], 'house type'] = house['construction'][type_key[0]]

            # drop features column
            df = df.drop(columns='features')
            return df
        dataframe = extract_features(dataframe)

        def extract_highlights(df):
            for i, house in enumerate(df['highlights']):
                if 'living area' in house:
                    area = house['living area'].split(' ')[0]
                    df.loc[df.index[i], 'living area (m²)'] = area
                if 'plot size' in house:
                    size = house['living area'].split(' ')[0]
                    df.loc[df.index[i], 'plot size (m²)'] = size
                if 'bedrooms' in house:
                    df.loc[df.index[i], 'no. bedrooms'] = house['bedrooms']
            # drop highlights column
            df = df.drop(columns='highlights')
            return df
        dataframe = extract_highlights(dataframe)

        def extract_cities(location_list):
            cities = []
            regex = re.compile(r'\(.+\)')
            for i in location_list:
                i = re.sub(regex, '', i)
                i_list = i.split(' ', maxsplit=2)
                cities.append(i_list[-1])
            return cities
        dataframe['city'] = extract_cities(dataframe['location_part2'])
        
        # price
        def filter_price(df):
            """
            Filters the items to only keep the ones that contain the price. the price in € is converted to int
            """
            # filter rows with regex
            regex = re.compile(r'€\s*(?P<price>\d{1,3}([,.]\d{3})*)\s*(k\.k\.|v\.o\.n\.)')
            df = df[df['price'].apply(lambda x: regex.match(x) is not None)].copy()
            # extract price
            df['price'] = df['price'].map(lambda x: regex.match(x).groupdict()['price'])
            # remove commas
            df['price'] = df['price'].map(lambda x: x.replace(',', '').replace('.', ''))
            # convert to int
            df['price'] = df['price'].astype(int)
            return df
        dataframe = filter_price(dataframe)

        # geolocation
        def geolocation_as_coord(x):
            """converts the geolocation to a tuple (lat, lon)"""
            return float(x['lat']), float(x['lon'])
        # dataframe['geolocation'] = dataframe['geolocation'].apply(geolocation_as_coord)
        
        def geolocation_as_cols(df):
            """
            Convert geolocation to two columns 'latitude' and 'longitude'
            TODO: add bool statement to choose between geolocation_as_coord or geolocation_as_cols
            """
            for i, coords in enumerate(df['geolocation']):
                df.loc[df.index[i], 'latitude'] = coords['lat']
                df.loc[df.index[i], 'longitude'] = coords['lon']
            df = df.drop(columns=['geolocation'])
            return df
        dataframe = geolocation_as_cols(dataframe)
        return dataframe
    



if __name__ == '__main__':
    # TEST CODE
    fd = FundaDataset.from_jsonlines('./data/ads.jsonlines', './data/funda_images_tiny')
    fd.export_pickle()
    # fd.save_as_csv('../data/Funda/Funda/ads.csv')
    # print(fd.df.head())
    # print()
    # print('VERIFY FORMATTING')
    # print(fd.df['images_paths'].head())
    # print(fd.df['price'].head())
    # print(fd.df['geolocation'].head())
    # print()
    # print('VERIFY ATTRIBUTES')
    # print(fd.get_images(42194092))
    # print(fd.get_coords(42194092))
    # print()
    # print('VERIFY IMAGES')
    # imgs = fd.get_images(42194092)
    # fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    # for i, ax in enumerate(axs):
    #     if i >= len(imgs):
    #         break
    #     ax.imshow(list(imgs.values())[i])
    #     ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    # print('ok')
    # print()
    # print('VERIFY EMBEDDINGS')
    # fe = FundaPrecomputedEmbeddings.from_pickle('../data/clip_embeddings/funda_images_tiny_umap.pkl')
    # print(len(fe))
    # emb = fe.get_embedding('42194072/image1.jpeg')
    # print(type(emb), emb.shape)
    # ux, uy = fe.get_umap('42194072/image1.jpeg')
    # print(type(ux), type(uy))
    # print()
    # print('VERIFY ALL EMBEDDINGS')
    # embs = fe.get_all_embeddings()
    # us = fe.get_all_umaps()
    # print(embs.shape, us.shape)
