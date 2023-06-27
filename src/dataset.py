import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image


class FundaDataset:
    def __init__(self, csv_path: str, images_path: str):
        self._load_df(csv_path)
        self._format_df(images_path)

    def _load_df(self, jsonlines_path: str) -> None:
        """
        Loads dataframe from a jsonlines file. Sets the index to be `funda_identifier`
        """
        self.df = pd.read_json(jsonlines_path, lines=True)
        self.df.set_index('funda_identifier', inplace=True)

    def _format_df(self, images_root: str) -> None:
        """
        Performs formatting on the dataframe. This includes:
        - appending the images root to the images paths
        - filtering the price
        - converting the geolocation to a tuple
        """
        def extract_features(df):
            df = df.drop(index=sdf.loc[42119366].name) # to be changed later: replace all values with nan instead or translate

            dict_funda = {  'price per square meters':[], 
                            'vve contribution':[],
                            'house type':[],
                            'construction year':[],
                            'energy label': [],
                            'no. rooms': [],
                            'no. bathrooms': [],
                            'no. stories': []
                        }

            for house in df['features']:
                if 'asking price per m²' in house['transfer of ownership']:
                    dict_funda['price per square meters'].append(house['transfer of ownership']['asking price per m²'])
                else:
                    dict_funda['price per square meters'].append(np.nan)

                if 'vve (owners association) contribution' in house['transfer of ownership']:
                    dict_funda['vve contribution'].append(house['transfer of ownership']['vve (owners association) contribution'])
                else:
                    dict_funda['vve contribution'].append(np.nan)

                if 'year of construction' in house['construction']:
                    dict_funda['construction year'].append(house['construction']['year of construction'])
                else:
                    dict_funda['construction year'].append(np.nan)

                if 'number of rooms' in house['layout']:
                    dict_funda['no. rooms'].append(house['layout']['number of rooms'])
                else:
                    dict_funda['no. rooms'].append(np.nan)

                if 'number of bath rooms' in house['layout']:
                    dict_funda['no. bathrooms'].append(house['layout']['number of bath rooms'])
                else:
                    dict_funda['no. bathrooms'].append(np.nan)

                if 'number of stories' in house['layout']:
                    dict_funda['no. stories'].append(house['layout']['number of stories'])
                else:
                    dict_funda['no. stories'].append(np.nan)

                if 'energy' in house:
                    if 'energy label' in house['energy']:
                        dict_funda['energy label'].append(house['energy']['energy label'])
                else:
                    dict_funda['energy label'].append(np.nan)

                if 'kind of house' in house['construction']:
                    dict_funda['house type'].append(house['construction']['kind of house'])
                elif 'type apartment' in house['construction']:
                    dict_funda['house type'].append(house['construction']['type apartment'])
                elif 'type of property' in house['construction']:
                    dict_funda['house type'].append(house['construction']['type of property'])
                elif 'type apartment' in house['construction']:
                    dict_funda['house type'].append(house['construction']['type apartment'])
                else:
                    dict_funda['house type'].append(np.nan)

            for key, value in dict_funda.items():
                df[key] = value
            return df
        self.df = extract_features(self.df)
        
        def extract_cities(location_list):
            cities = []
            regex = re.compile(r'\(.+\)')
            for i in location_list:
                i = re.sub(regex, '', i)
                i_list = i.split(' ', maxsplit=2)
                cities.append(i_list[-1])
            return cities
        self.df['city'] = extract_cities(self.df['location_part2'])
        
        # images paths
        def append_images_root(img_list):
            """appends the images root to the images paths"""
            #img_list = eval(x)
            return [os.path.join(images_root, str(img)) for img in img_list]
        self.df['images_paths'] = self.df['images_paths'].apply(append_images_root)

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

    def get_images(self, funda_id: int) -> list[Image.Image]:
        """
        Get the PIL.Images for a house with a given funda_id
        The images are returned as a list of PIL.Images
        The list can have a variable length from 1 to 5, depending on the number of images available for each house
        """
        images = self.df.loc[funda_id, 'images_paths']
        images = [Image.open(img) for img in images]
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
    fd = FundaDataset('data/Funda/Funda/ads.jsonlines', 'funda')
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
        ax.imshow(imgs[i])
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


