import argparse, os
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

def load_img_mask(img_file, mask_file, down_factor):
    image = Image.open(img_file).convert('RGB')
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.
    image = image.to(device.type)

    mask = Image.open(mask_file).convert('L')
    mask = mask.resize((w // down_factor, h // down_factor), resample=Image.Resampling.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask = torch.from_numpy(mask).to(device.type)
    return image, mask

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

    config = OmegaConf.load('src/configs/v2-1-stable-unclip-h-bind-inference.yaml')
    device_name = 'cuda'
    device = torch.device(device_name) # if opt.device == 'cuda' else torch.device('cpu')
    model = load_model_from_config(config, '/home/adryw/dataset/imagecraft/sd21-unclip-h.ckpt', device)

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
    prompt = 'a dog, photo'
    data_prompts = [batch_size * [prompt]]

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
    diff_steps = 100

    # Load img bind stuff
    image_paths = ["samples/dog_image.jpg"]
    audio_paths = ["samples/dog_sound.wav"]
    # depth_paths = "samples/01441.h5"

    # with h5py.File(depth_paths, 'r') as f:
    #     depth = np.array(f['depth'])
    #     rgb = np.array(f['rgb'])
    # depth = ((depth / depth.max()) * 255).astype(np.uint8)
    # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
        # ModalityType.DEPTH: data.load_and_transform_thermal_data([depth], device),
    }

    # Prepare embeddings cond
    with torch.no_grad():  # TODO: In main_bind2 they use norm=True for audio
        embeddings_imagebind = model.embedder(inputs, normalize=False)
    strength = 0.25
    noise_level = 0.0

    alpha = 0.5
    # embeddings_imagebind = 0.1 * embeddings_imagebind[ModalityType.DEPTH]
    # embeddings_imagebind = alpha * embeddings_imagebind[ModalityType.AUDIO] + (1-alpha) * embeddings_imagebind[ModalityType.VISION]
    embeddings_imagebind = embeddings_imagebind[ModalityType.VISION]

    c_adm = repeat(embeddings_imagebind, '1 ... -> b ...', b=batch_size) * strength
    # c_adm = (c_adm / c_adm.norm()) * 20

    if model.noise_augmentor is not None:
        c_adm, noise_level_emb = model.noise_augmentor(c_adm, noise_level=(
                torch.ones(batch_size) * model.noise_augmentor.max_noise_level *
                noise_level).long().to(c_adm.device))
        # assume this gives embeddings of noise levels
        c_adm = torch.cat((c_adm, noise_level_emb), 1)

    n_prompt = "text, watermark, blurry, number, poor quality, low quality, cropped"

    strength = 0.9  # How strong is the init_image transformed during diffusion 1.0 full destruction [0,1]
    t_enc = int(strength * diff_steps)

    # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
    # it should be higher -> closer to prompt
    scale = 3

    # Init image and load mask :)
    init_image, mask = load_img_mask('samples/sample_inpaint.png', 'samples/sample_inpaint_mask.png', f)
    print(f'>>> Loaded img and mask with shape {init_image.shape}, {mask.shape}')

    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size).to(device)  # Propag. over batch
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=diff_steps, ddim_eta=ddim_eta, verbose=False)

    # Transform z_enc based on mask
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
    random = torch.randn(mask.shape, device=model.device)
    z_enc = (mask * random) + ((1 - mask) * z_enc)

    # fiuuu
    model.embedder.to('cpu')

    # Warmup
    with torch.no_grad(), model.ema_scope():
        for prompts in tqdm(data_prompts, desc="data"):
            uc_cross = None
            if scale != 1.0:
                uc_cross = model.get_learned_conditioning(batch_size * [n_prompt])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)

            c = {'c_crossattn': [c], 'c_adm': c_adm}
            uc = {"c_crossattn": [uc_cross], 'c_adm': c_adm}

            samples = sampler.decode_inpaint(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                             z_mask=mask, x0=init_latent, unconditional_conditioning=uc)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample in x_samples:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1
                sample_count += 1
    # =======