import argparse, os
from IPython import embed
import cv2
import torch
import PIL
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
# from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

torch.set_grad_enabled(False)


def load_img(path, w=768, h=768):
    image = Image.open(path).convert("RGB")
    if w is None or h is None:
        w, h = image.size
        print(f"loaded input image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


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

    config = OmegaConf.load('configs/v2-1-stable-unclip-h-bind-inference.yaml')
    device_name = 'cuda'
    device = torch.device(device_name) # if opt.device == 'cuda' else torch.device('cpu')
    model = load_model_from_config(config, 'ldm/sd21-unclip-h.ckpt', device)
    model_imagebind = imagebind_model.imagebind_huge(ckpt_path='imagebind/imagebind_huge.pth', pretrained=True).to(device)

    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
    # https://stable-diffusion-art.com/samplers/
    sampler = DDIMSampler(model, device=device)
    # sampler = PLMSSampler(model, device=device)
    # sampler = DPMSolverSampler(model, device=device)
    ddim_eta = 0  # "ddim eta (eta=0.0 corresponds to deterministic sampling"

    # Out folders
    outpath = 'output/'
    os.makedirs(outpath, exist_ok=True)

    # Watermark?
    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # Hardcoded batches and prompts (can be read from file)
    batch_size = 1
    n_rows = 1
    prompt = ''
    prompt = prompt + ', best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, creepy'
    prompts = [batch_size * [prompt]]

    # Init image
    #init_image1 = load_img('samples/human_face.jpg')
    #init_image1 = repeat(init_image1, '1 ... -> b ...', b=batch_size).to(device)  # Propag. over batch
    #init_latent1 = model.get_first_stage_encoding(model.encode_first_stage(init_image1))  # move to latent space

    #init_image2 = load_img('samples/dog_image.jpg')
    #init_image2 = repeat(init_image2, '1 ... -> b ...', b=batch_size).to(device)  # Propag. over batch
    #init_latent2 = model.get_first_stage_encoding(model.encode_first_stage(init_image2))  # move to latent space

    init_image1 = data.load_and_transform_vision_data(['samples/human_face.jpg'], device)
    init_image2 = data.load_and_transform_vision_data(['samples/dog_image.jpg'], device)
    inputs1 = { ModalityType.VISION: init_image1 }
    inputs2 = { ModalityType.VISION: init_image2 }
    outs1 = model_imagebind(inputs1)
    outs2 = model_imagebind(inputs2)
    embeddings1 = outs1[ModalityType.VISION]
    embeddings2 = outs2[ModalityType.VISION]
    print(embeddings1, embeddings2)
    print(embeddings1.shape, embeddings2.shape)
    print(embeddings1 == embeddings2)

    embeddings = 0.5 * embeddings1 + 0.5 * embeddings2

    strength = 1
    c_adm = repeat(embeddings, '1 ... -> b ...', b=batch_size) * strength

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
    start_code = torch.randn([batch_size, *shape], device=device)
    diff_steps = 200

    # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
    scale = 9

    sampler.make_schedule(ddim_num_steps=diff_steps, ddim_eta=ddim_eta, verbose=False)

    with torch.no_grad(), model.ema_scope():
        uc = None
        if scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [n_prompt])
        uc = {"c_crossattn": [uc], "c_adm": torch.zeros_like(c_adm)}

        if isinstance(prompt, tuple):
            prompt = list(prompt)

        c = {"c_crossattn": [model.get_learned_conditioning(prompt)], "c_adm": c_adm}
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
            # img = put_watermark(img, wm_encoder)
            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1
            sample_count += 1
# =======
