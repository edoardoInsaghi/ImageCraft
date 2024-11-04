import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
# from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


if __name__ == '__main__':
    seed_everything(42)

    config = OmegaConf.load('src/configs/v2-inference-v.yaml')
    device_name = 'cuda'
    device = torch.device(device_name) # if opt.device == 'cuda' else torch.device('cpu')
    # model = load_model_from_config(config, '/home/adryw/dataset/imagecraft/sd21-unclip-h.ckpt', device)
    # model = load_model_from_config(config, '/home/adryw/dataset/imagecraft/v2-1_512-ema-pruned.ckpt', device)
    model = load_model_from_config(config, '/u/dssc/adonninelli/scratch/sd21-unclip-h.ckpt', device)

    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
    # https://stable-diffusion-art.com/samplers/
    sampler = DDIMSampler(model, device=device)
    # sampler = PLMSSampler(model, device=device)
    # sampler = DPMSolverSampler(model, device=device)
    ddim_eta = 0  # "ddim eta (eta=0.0 corresponds to deterministic sampling"

    # Out folders
    outpath = './output/'
    os.makedirs(outpath, exist_ok=True)

    # Watermark?
    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # Hardcoded batches and prompts (can be read from file)
    batch_size = 1
    n_rows = 1
    prompt = 'A cat battle mage, sword and shild with runes, dramatic lighting, dynamic pose, dynamic camera, masterpiece, best quality, dark shadows, ((dark fantasy) ), detailed, realistic, 8k uhd, high quality'
    n_prompt = '(worst quality:2) , (low quality:2) , (mutated hands and fingers) , (poorly drawn hands) , (mutation)'
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # Can be different
    C = 4  # Latent channels
    H = 768
    W = 768
    f = 8  # Downsampling factor
    shape = [C, H // f, W // f]
    diff_steps = 40

    # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
    scale = 7

    for i in range(30):
        seed_everything(i * 123456789)
        start_code = torch.randn([batch_size, *shape], device=device)

        # Warmup
        uc = None
        with torch.no_grad(), model.ema_scope():
            for prompts in tqdm(data, desc="data"):
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [n_prompt])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)

                samples, _ = sampler.sample(S=diff_steps,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=start_code)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1
                    sample_count += 1
        # =======