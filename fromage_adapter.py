import numpy as np 
import pickle
import pandas as pd 
import torch 
import sys
sys.path.append("./fromage")
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

  def _load_embeddings(self):
    '''
    Utility function to load embeddings from the .pkl pre-computed funda embeddings.
    '''
    # The embeddings path
    path_array = []
    emb_matrix = []
    # These were precomputed for all funda dataset images.
    with open(self.path_to_embeddings, 'rb') as wf:
          train_embs_data = pickle.load(wf)
          path_array.extend(train_embs_data['paths'])
          emb_matrix.append(train_embs_data['embeddings'])
    emb_matrix = np.concatenate(emb_matrix, axis=0)
    assert len(path_array) == emb_matrix.shape[0], (len(path_array), emb_matrix.shape[0])
    return path_array,emb_matrix
  
  def _update_embeddings_from_id(self, ids):
    '''
    Utility function to update embeddings from the  ids extracted from the dataframe.
    This function initialized embeddings dictionary as fromage models expects and
    embeds all the images corresponding to the ids retrieved from the data.
    '''
    # This is used for passing the ids data to embeddings, then we can retrive the images for the id
    # and embed them.
    embeddings = {"embeddings": [], "paths": []}
    # For each id, there exists a directory with images.
    for id in ids:
      # Path to images needs to contain a seperate folder for each id in the sub-folder
      search_path = os.path.join(self.path_to_images, id)
      # Glob is used here to retrieve the images
      image_paths = glob.glob("*.png", root_dir = search_path)
      for image_path in image_paths:
            p = get_image_from_png(path=image_path)
            pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, p)
            pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
            pixel_values = pixel_values[None, ...]
            visual_embs = self.model.get_visual_embs(pixel_values, mode='retrieval')
            embeddings['embeddings'].append(visual_embs)
            embeddings['paths'].append(image_path)
      return embeddings

  def _compute_embeddings_from_id(self, ids):
    '''
    This method returns embedding matrix and path array to initialize the fromage models' embeddings matrix 
    and path array.
    '''
    train_embs_data = self._update_embeddings_from_id(ids)
    path_array = []
    emb_matrix = []
    # These were precomputed for all funda dataset images.
    with open(self.path_to_embeddings, 'rb') as wf:
          train_embs_data = pickle.load(wf)
          path_array.extend(train_embs_data['paths'])
          emb_matrix.append(train_embs_data['embeddings'])
    emb_matrix = np.concatenate(emb_matrix, axis=0)
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
    logit_scale = self.model.logit_scale.exp()
    emb_matrix = torch.tensor(emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
    emb_matrix = emb_matrix / emb_matrix.norm(dim=1, keepdim=True)
    emb_matrix = logit_scale * emb_matrix
    self.model.emb_matrix = emb_matrix
  
  def update_embeddings(self, dataframe: pd.DataFrame):
    '''
    Update the embeddings of the fromage model. We re-write everything because 
    it is faster due to L1 cache retrieval. This method goes through the data and computes
    the embeddings of the images again. In the future we can replace this with a method that only fetches
    the embeddings of the ids from a database because those embeddings in principle should never be different.
    '''
    # dataframe is assumed to be the funda dataset, but filtered.
    assert self.model.emb_matrix != None and self.model.path_array != None, "Please initialize the embeddings first"
    # Load the pre-computed embeddings and paths
    path_array,emb_matrix = self._compute_embeddings_from_id(dataframe['id'])
    self.model.path_array = path_array  
    # Normalize the embeddings
    logit_scale = self.model.logit_scale.exp()
    emb_matrix = torch.tensor(emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
    emb_matrix = emb_matrix / emb_matrix.norm(dim=1, keepdim=True)
    emb_matrix = logit_scale * emb_matrix
    self.model.emb_matrix = emb_matrix

if __name__ == "__main__":
    model_dir = './fromage/fromage_model/'
    model = models.load_fromage(model_dir)
    path_to_embeddings = "./embeddings" 
    path_to_images = "./images"
    adapter = FromageModel(model, path_to_embeddings, path_to_images)
    print("Loaded the fromage model!")