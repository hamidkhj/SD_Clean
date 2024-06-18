# import argparse, os
# import cv2
# import torch
# import numpy as np
# from omegaconf import OmegaConf
# from PIL import Image
# from tqdm import tqdm, trange
# from itertools import islice
# from einops import rearrange
# from torchvision.utils import make_grid
# from pytorch_lightning import seed_everything
# from torch import autocast
# from contextlib import nullcontext
# from imwatermark import WatermarkEncoder

# from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# torch.set_grad_enabled(False)

# def chunk(it, size):
#     it = iter(it)
#     return iter(lambda: tuple(islice(it, size)), ())


# def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     if "global_step" in pl_sd:
#         print(f"Global Step: {pl_sd['global_step']}")
#     sd = pl_sd["state_dict"]
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)

#     if device == torch.device("cuda"):
#         model.cuda()
#     elif device == torch.device("cpu"):
#         model.cpu()
#         model.cond_stage_model.device = "cpu"
#     else:
#         raise ValueError(f"Incorrect device name. Received: {device}")
#     model.eval()
#     return model




# def main(opt):
#     seed_everything(opt.seed)

#     config = OmegaConf.load(f"{opt.config}")
#     device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
#     model = load_model_from_config(config, f"{opt.ckpt}", device)

    
# def test ():
#     checkpoint = torch.load('checkpoints/v2-1_768-ema-pruned.ckpt', map_location="cpu")
#     print('model here')
#     if 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#     for x in state_dict.keys():
#         print(x)


# parser = argparse.ArgumentParser()
# parser.add_argument('--ali', action='store_true')
# parser.add_argument('--bar', action='store_false')
# parser.add_argument('--baz', action='store_false')
# opt = parser.parse_args()
# print(opt)