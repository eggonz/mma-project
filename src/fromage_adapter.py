import glob
import pickle
import pickle as pkl
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import umap_utils
from fromage import models
from fromage import utils


class FromageModel:
  '''
  This is an adapter class that takes initialized fromage model with path to embeddings and path to images.
  This class is used for initializing the embeddings of the fromage model for proximity search, or updating them
  as we go with filtering the data with chatGPT.
  '''
  def __init__(self, model, path_to_embeddings:str, path_to_images: str):
    '''
    Initialize the class with fromage model, path_to_embeddings and path_to_images. They need to be
    absolute paths
    '''
    self.model = model
    self.path_to_embeddings = path_to_embeddings
    self.path_to_images = path_to_images


  @staticmethod
  def _display_interleaved_outputs(model_outputs, one_img_per_ret=True):
    for output in model_outputs:
        if type(output) == str:
            print(output)
        elif type(output) == list:
            if one_img_per_ret:
                plt.figure(figsize=(3, 3))
                plt.imshow(np.array(output[0]))
            else:
                fig, ax = plt.subplots(1, len(output), figsize=(3 * len(output), 3))
                for i, image in enumerate(output):
                    image = np.array(image)
                    ax[i].imshow(image)
                    ax[i].set_title(f'Retrieval #{i+1}')
            plt.show()
        elif type(output) == Image.Image:
            plt.figure(figsize=(3, 3))
            plt.imshow(np.array(output))
            plt.show()


            
  def _get_embedding_from_image_path(self, image_path):
    index = self.model.path_array.index(image_path)
    return self.model.emb_matrix[index]

  def _load_embeddings(self):
    '''
    Utility function to load embeddings from the .pkl pre-computed funda embeddings.
    '''

    # Load embeddings.
    # Construct embedding matrix for nearest neighbor lookup.
    path_array = []
    emb_matrix = []
    with open(self.path_to_embeddings, 'rb') as wf:
          train_embs_data = pkl.load(wf)
          path_array.extend(train_embs_data['paths'])
          emb_matrix.append(train_embs_data['embeddings'])
    emb_matrix = np.concatenate(emb_matrix[0], axis=0)
    # Number of paths should be equal to number of embeddings.
    assert len(path_array) == emb_matrix.shape[0], (len(path_array), emb_matrix.shape[0])
    return path_array,emb_matrix

  def _filters_embeddings_from_id(self, ids):
    '''
    Utility function to update embeddings from the  ids extracted from the dataframe.
    This function initialized embeddings dictionary as fromage models expects and
    embeds all the images corresponding to the ids retrieved from the data.
    '''
    # This is used for passing the ids data to embeddings, then we can retrive the images for the id
    # and embed them.
    embeddings = {'embeddings': [], 'paths': []}
    # Open the json file
    for id_ in ids:
      image_paths = glob.glob(path_to_images + f"/{id_}/*.jpeg", recursive = True)
      for image_path in image_paths:
        visual_embs = self._get_embedding_from_image_path(image_path)
        embeddings['embeddings'].append(visual_embs)
        embeddings['paths'].append(image_path)
    with open('/content/mma-project/fromage/fromage_model/embeddings/funda_sample_updated.pkl', 'wb') as f:
          pickle.dump(embeddings, f)
    return embeddings

  def _filter_embeddings_from_id(self, ids):
    '''
    This method returns embedding matrix and path array to initialize the fromage models' embeddings matrix
    and path array.
    '''
    train_embs_data = self._filters_embeddings_from_id(ids)
    path_array = []
    emb_matrix = []
    # These were precomputed for all funda dataset images.
    path_array.extend(train_embs_data['paths'])
    emb_matrix.append(train_embs_data['embeddings'])
    emb_matrix = np.concatenate(emb_matrix[0], axis=0)
    assert len(path_array) == emb_matrix.shape[0], (len(path_array), emb_matrix.shape[0])
    return path_array, emb_matrix
  
  def _updates_embeddings_from_id(self, ids):
    '''
    Utility function to update embeddings from the  ids extracted from the dataframe.
    This function initialized embeddings dictionary as fromage models expects and
    embeds all the images corresponding to the ids retrieved from the data.
    '''
    # This is used for passing the ids data to embeddings, then we can retrive the images for the id
    # and embed them.
    embeddings = {'embeddings': [], 'paths': []}
    # Open the json file
    for id_ in ids:
      image_paths = glob.glob(path_to_images + f"/{id_}/*.jpeg", recursive = True)
      for image_path in image_paths:
        print(image_path)
        p = utils.get_image_from_url(image_path)
        with torch.no_grad():
          pixel_values = utils.get_pixel_values_for_model(model.model.feature_extractor, p)
          pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
          pixel_values = pixel_values[None, ...]
          visual_embs = model.model.get_visual_embs(pixel_values, mode='retrieval')
        embeddings['embeddings'].append(visual_embs.float().cpu().numpy())
        embeddings['paths'].append(image_path)
    with open('/content/mma-project/fromage/fromage_model/embeddings/funda_sample_updated.pkl', 'wb') as f:
          pickle.dump(embeddings, f)
    return embeddings

  def _update_embeddings_from_id(self, ids):
    '''
    This method returns embedding matrix and path array to initialize the fromage models' embeddings matrix
    and path array.
    '''
    train_embs_data = self._updates_embeddings_from_id(ids)
    path_array = []
    emb_matrix = []
    # These were precomputed for all funda dataset images.
    path_array.extend(train_embs_data['paths'])
    emb_matrix.append(train_embs_data['embeddings'])
    emb_matrix = np.concatenate(emb_matrix[0], axis=0)
    assert len(path_array) == emb_matrix.shape[0], (len(path_array), emb_matrix.shape[0])
    return path_array, emb_matrix
          
  def refresh_embeddings(self):
    '''
    Refresh the embeddings of the fromage model, this is done so that when user presses restart
    fromage model starts with fresh embeddings.
    '''
    # Load the pre-computed embeddings and paths
    path_array,emb_matrix = self._load_embeddings()
    self.model.path_array = path_array
    # Normalize the embeddings
    with torch.no_grad():
      logit_scale = self.model.model.logit_scale.exp()
      emb_matrix = torch.tensor(emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
      emb_matrix = emb_matrix / emb_matrix.norm(dim=1, keepdim=True)
      emb_matrix = logit_scale * emb_matrix
      self.model.emb_matrix = emb_matrix

  def filter_embeddings(self, dataframe: pd.DataFrame):
    '''
    Fetchesbthe embeddings of the ids from a database and creates a new embeddings matrix.
    '''
    # dataframe is assumed to be the funda dataset, but filtered.
    assert self.model.emb_matrix != None and self.model.path_array != None, "Please initialize the embeddings first"
    # Load the pre-computed embeddings and paths
    path_array,emb_matrix = self._filter_embeddings_from_id(dataframe['id'])
    self.model.path_array = path_array
    self.model.emb_matrix = emb_matrix
  
  def update_embeddings(self, dataframe: pd.DataFrame):
    '''
    Fetchesbthe embeddings of the ids from a database and creates a new embeddings matrix.
    '''
    # dataframe is assumed to be the funda dataset, but filtered.
    assert self.model.emb_matrix != None and self.model.path_array != None, "Please initialize the embeddings first"
    # Load the pre-computed embeddings and paths
    path_array,emb_matrix = self._update_embeddings_from_id(dataframe['id'])
    self.model.path_array = path_array
    with torch.no_grad():
      logit_scale = self.model.model.logit_scale.exp()
      emb_matrix = torch.tensor(emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
      emb_matrix = emb_matrix / emb_matrix.norm(dim=1, keepdim=True)
      emb_matrix = logit_scale * emb_matrix
      self.model.emb_matrix = emb_matrix

  def prompt(self, text:str, image: Image.Image, nr_of_retrieval: int = 1, debug = False) -> List[Union[Image.Image, int]]:
    '''
    Prompt is expected to be a PIL image and a string of text. We want to adjust 
    how many images we want to retrive with nr_of_retrieval.
    Returns
    --------
        List of tuples of PIL.Image and int. PIL.Image is the retrieved image and int is the unique identifier
        of the data.
    '''
    # Append the ret token to the text.
    text = text + " [RET]"
    prompt = [image, text]
    # Set number of words to only generate images.
    # This output is list of images and ints
    print(prompt)
    outputs, ids = self.model.generate_for_images_and_texts(prompts = prompt,
                                            max_img_per_ret=nr_of_retrieval
                                            )
    print(outputs)
    if debug:
      self._display_interleaved_outputs(outputs, nr_of_retrieval == 1)
    returns = []
    for output in outputs:
      if type(output) != str:
        returns.append(output)
    return list(zip(returns, ids))


if __name__ == "__main__":
    model_dir = '/content/mma-project/fromage/fromage_model'
    model = models.load_fromage_for_embeddings(model_dir)
    path_to_embeddings = "/content/mma-project/fromage/fromage_model/embeddings/funda_sample.pkl" 
    path_to_images = "/content/mma-project/funda"

    adapter = FromageModel(model, path_to_embeddings, path_to_images)
    print("Loaded the adapter fromage model!")

    adapter.refresh_embeddings()
    output = adapter.prompt("Similar images", utils.get_image_from_url("/content/mma-project/funda/42194096/image1.jpeg"))
    print(output)

    umap_embeddings = umap_utils.compute_umap(adapter.model.emb_matrix)
    fig = umap_utils.draw_umap(umap_embeddings)
    fig.savefig("All_embeddings.png")

    data_dummy_filter = data = {'id': [42194016, 42194023]}
    adapter.update_embeddings(pd.DataFrame.from_dict(data_dummy_filter))
    output = adapter.prompt("Similar images", utils.get_image_from_url("/content/mma-project/funda/42194096/image1.jpeg"))
    print(output)

    fig = umap_utils.draw_umap(umap_embeddings)
    fig.savefig("Reduced_embeddings.png")
