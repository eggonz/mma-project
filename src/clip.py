import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from fromage import utils


class FrozenArgs:
    visual_encoder: str = 'openai/clip-vit-large-patch14'
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

        hidden_size = self.visual_model.config.hidden_size

        # freeze
        self.visual_model.eval()
        for param in self.visual_model.parameters():
            param.requires_grad = False

        embedding_dim = args.emb_dim
        self.visual_fc = nn.Linear(hidden_size, embedding_dim)
        # dropout=0

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        # mode=retrieval

        outputs = self.visual_model(pixel_values)
        encoder_outputs = outputs.pooler_output

        visual_embs = self.visual_fc(encoder_outputs)  # (2, D * n_visual_tokens)
        visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1))

        return visual_embs


def load_clip_for_embeddings() -> ClipModel:
    args = FrozenArgs()

    model = ClipModel(args=args)
    model = model.eval()
    model = model.float()
    model = model.cuda()

    return model
