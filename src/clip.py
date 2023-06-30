import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel

import utils


class FrozenArgs:
    visual_encoder: str = 'openai/clip-vit-base-patch32'  # 'openai/clip-vit-large-patch14'
    n_visual_tokens: int = 1
    emb_dim: int = 512


class ClipModel(nn.Module):
    def __init__(self, args: FrozenArgs = FrozenArgs()):
        super().__init__()
        self.feature_extractor = utils.get_feature_extractor_for_model(args.visual_encoder, train=False)
        self.args = args

        visual_encoder = args.visual_encoder
        n_visual_tokens = args.n_visual_tokens
        print(f"Using {visual_encoder} for the visual model with {n_visual_tokens} visual tokens.")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.visual_model = CLIPVisionModel.from_pretrained(visual_encoder, torch_dtype=torch.float32)

        # freeze
        self.visual_model.eval()
        for param in self.visual_model.parameters():
            param.requires_grad = False

        hidden_size = self.visual_model.config.hidden_size
        embedding_dim = args.emb_dim
        self.visual_fc = nn.Linear(hidden_size, embedding_dim)
        # dropout=0

    @torch.no_grad()
    def get_visual_embs(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes the visual embeddings for the pixel values tensor.
        Assumes batch dimension in the first dimension.
        Resulting shape: (batch_size, 1, embedding_dim)
        """
        # inference through clip
        outputs = self.visual_model(pixel_values)
        encoder_outputs = outputs.pooler_output
        visual_embs = self.visual_fc(encoder_outputs)  # (2, D * n_visual_tokens)

        # transform to (B, C=1, embedding_dim)
        visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1))
        visual_embs = visual_embs.float()
        return visual_embs

    @torch.no_grad()
    def get_visual_emb_for_img(self, img: Image.Image) -> np.ndarray:
        """
        Computes the visual embeddings for the given image.
        img is a PIL Image
        """
        img = img.resize((224, 224))
        img = img.convert('RGB')

        # convert to tensor (B=1, C=3, width, height), single image
        pixel_values = utils.get_pixel_values_for_model(self.feature_extractor, img)
        pixel_values = pixel_values.to(device=self.logit_scale.device, dtype=self.logit_scale.dtype)
        pixel_values: torch.FloatTensor = pixel_values[None, ...]

        # back to numpy (B=1, C=1, hdim) -> (hdim,), single embedding
        visual_embs = self.get_visual_embs(pixel_values)
        visual_emb = visual_embs[0, 0, :].cpu().numpy()

        return visual_emb


def load_clip_model(gpu=False) -> ClipModel:
    """
    Loads the CLIP model.
    Downloads the model if necessary.
    """
    args = FrozenArgs()

    model = ClipModel(args=args)
    model = model.eval()
    model = model.float()
    if gpu:
        model = model.cuda()
    print("Loaded the model")
    return model


def compute_emb_distances(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity between the query embedding and the document embeddings.
    The output is in range [-1, 1], where 1 means the embeddings are identical.

    :param query_embedding: (embedding_dim,)
    :param document_embeddings: (num_documents, embedding_dim)
    """
    # normalize
    query_embedding /= np.linalg.norm(query_embedding)
    document_embeddings /= np.linalg.norm(document_embeddings, axis=1, keepdims=True)

    # compute cosine similarity
    cosine_similarities = np.dot(document_embeddings, query_embedding)
    return cosine_similarities
