import cog

import os, sys
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange

# tqdm_auto_model = __import__("tqdm.auto", fromlist=[None])
# sys.modules["tqdm"] = tqdm_auto_model
from einops import rearrange
from torchvision.utils import make_grid
import gc

sys.path.append("/latent-diffusion")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.diffusionmodules.util import noise_like
from dotmap import DotMap

import io
import sys
import lpips
import requests
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import time
import re
import base64
import sys

sys.path.append(".")

# suppress mmc warmup outputs
from resize_right import resize

import mmc.loaders
from mmc.multimmc import MultiMMC
from mmc.modalities import TEXT, IMAGE
import mmc
from mmc.registry import REGISTRY
import mmc.loaders  # force trigger model registrations
from mmc.mock.openai import MockOpenaiClip


model_path = "/content/models/"
outputs_path = "/outputs"  # TODO check


torch.backends.cudnn.benchmark = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_from_config(
    config, ckpt, verbose=False, latent_diffusion_model="original"
):
    print(f"Loading model from {ckpt}")
    print(latent_diffusion_model)
    model = instantiate_from_config(config.model)
    if latent_diffusion_model != "finetuned":
        sd = torch.load(ckpt, map_location="cuda")["state_dict"]
        m, u = model.load_state_dict(sd, strict=False)

    if latent_diffusion_model == "finetuned":
        sd = torch.load(
            f"{model_path}/txt2img-f8-large-jack000-finetuned-fp16.ckpt",
            map_location="cuda",
        )
        m, u = model.load_state_dict(sd, strict=False)
        # model.model = model.model.half().eval().to(device)

    if latent_diffusion_model == "ongo (fine tuned in art)":
        del sd
        sd_finetuned = torch.load(f"{model_path}/ongo.pt")
        sd_finetuned["input_blocks.0.0.weight"] = sd_finetuned[
            "input_blocks.0.0.weight"
        ][:, 0:4, :, :]
        model.model.diffusion_model.load_state_dict(sd_finetuned, strict=False)
        del sd_finetuned
        torch.cuda.empty_cache()
        gc.collect()

    if latent_diffusion_model == "erlich (fine tuned in logos)":
        del sd
        sd_finetuned = torch.load(f"{model_path}/erlich.pt")
        sd_finetuned["input_blocks.0.0.weight"] = sd_finetuned[
            "input_blocks.0.0.weight"
        ][:, 0:4, :, :]
        model.model.diffusion_model.load_state_dict(sd_finetuned, strict=False)
        del sd_finetuned
        torch.cuda.empty_cache()
        gc.collect()

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.requires_grad_(False).half().eval().to("cuda")
    return model


# @title Choose your perceptor models

# suppress mmc warmup outputs
import mmc.loaders

clip_load_list = []
# @markdown #### Open AI CLIP models
ViT_B32 = False  # @param {type:"boolean"}
ViT_B16 = True  # @param {type:"boolean"}
ViT_L14 = True  # @param {type:"boolean"}
ViT_L14_336px = False  # @param {type:"boolean"}
# RN101 = False #@param {type:"boolean"}
# RN50 = False #@param {type:"boolean"}
RN50x4 = False  # @param {type:"boolean"}
RN50x16 = False  # @param {type:"boolean"}
RN50x64 = False  # @param {type:"boolean"}

# @markdown #### OpenCLIP models
ViT_B16_plus = False  # @param {type: "boolean"}
ViT_B32_laion2b = True  # @param {type: "boolean"}
ViT_L14_laion = False  # @param {type:"boolean"}

# @markdown #### Multilangual CLIP models
clip_farsi = False  # @param {type: "boolean"}
clip_korean = False  # @param {type: "boolean"}

# @markdown #### CLOOB models
cloob_ViT_B16 = False  # @param {type: "boolean"}

# @markdown Load even more CLIP and CLIP-like models (from [Multi-Modal-Comparators](https://github.com/dmarx/Multi-Modal-Comparators))
model1 = ""  # @param ["[clip - mlfoundations - RN50--openai]","[clip - mlfoundations - RN101--openai]","[clip - mlfoundations - RN50--yfcc15m]","[clip - mlfoundations - RN50--cc12m]","[clip - mlfoundations - RN50-quickgelu--yfcc15m]","[clip - mlfoundations - RN50-quickgelu--cc12m]","[clip - mlfoundations - RN101--yfcc15m]","[clip - mlfoundations - RN101-quickgelu--yfcc15m]","[clip - mlfoundations - ViT-B-32--laion400m_e31]","[clip - mlfoundations - ViT-B-32--laion400m_e32]","[clip - mlfoundations - ViT-B-32--laion400m_avg]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e31]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e32]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_avg]","[clip - mlfoundations - ViT-B-16--laion400m_e31]","[clip - mlfoundations - ViT-B-16--laion400m_e32]","[clip - sbert - ViT-B-32-multilingual-v1]","[clip - facebookresearch - clip_small_25ep]","[simclr - facebookresearch - simclr_small_25ep]","[slip - facebookresearch - slip_small_25ep]","[slip - facebookresearch - slip_small_50ep]","[slip - facebookresearch - slip_small_100ep]","[clip - facebookresearch - clip_base_25ep]","[simclr - facebookresearch - simclr_base_25ep]","[slip - facebookresearch - slip_base_25ep]","[slip - facebookresearch - slip_base_50ep]","[slip - facebookresearch - slip_base_100ep]","[clip - facebookresearch - clip_large_25ep]","[simclr - facebookresearch - simclr_large_25ep]","[slip - facebookresearch - slip_large_25ep]","[slip - facebookresearch - slip_large_50ep]","[slip - facebookresearch - slip_large_100ep]","[clip - facebookresearch - clip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc12m_35ep]","[clip - facebookresearch - clip_base_cc12m_35ep]"] {allow-input: true}
model2 = ""  # @param ["[clip - mlfoundations - RN50--openai]","[clip - mlfoundations - RN101--openai]","[clip - mlfoundations - RN50--yfcc15m]","[clip - mlfoundations - RN50--cc12m]","[clip - mlfoundations - RN50-quickgelu--yfcc15m]","[clip - mlfoundations - RN50-quickgelu--cc12m]","[clip - mlfoundations - RN101--yfcc15m]","[clip - mlfoundations - RN101-quickgelu--yfcc15m]","[clip - mlfoundations - ViT-B-32--laion400m_e31]","[clip - mlfoundations - ViT-B-32--laion400m_e32]","[clip - mlfoundations - ViT-B-32--laion400m_avg]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e31]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e32]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_avg]","[clip - mlfoundations - ViT-B-16--laion400m_e31]","[clip - mlfoundations - ViT-B-16--laion400m_e32]","[clip - sbert - ViT-B-32-multilingual-v1]","[clip - facebookresearch - clip_small_25ep]","[simclr - facebookresearch - simclr_small_25ep]","[slip - facebookresearch - slip_small_25ep]","[slip - facebookresearch - slip_small_50ep]","[slip - facebookresearch - slip_small_100ep]","[clip - facebookresearch - clip_base_25ep]","[simclr - facebookresearch - simclr_base_25ep]","[slip - facebookresearch - slip_base_25ep]","[slip - facebookresearch - slip_base_50ep]","[slip - facebookresearch - slip_base_100ep]","[clip - facebookresearch - clip_large_25ep]","[simclr - facebookresearch - simclr_large_25ep]","[slip - facebookresearch - slip_large_25ep]","[slip - facebookresearch - slip_large_50ep]","[slip - facebookresearch - slip_large_100ep]","[clip - facebookresearch - clip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc12m_35ep]","[clip - facebookresearch - clip_base_cc12m_35ep]"] {allow-input: true}
model3 = ""  # @param ["[clip - openai - RN50]","[clip - openai - RN101]","[clip - mlfoundations - RN50--yfcc15m]","[clip - mlfoundations - RN50--cc12m]","[clip - mlfoundations - RN50-quickgelu--yfcc15m]","[clip - mlfoundations - RN50-quickgelu--cc12m]","[clip - mlfoundations - RN101--yfcc15m]","[clip - mlfoundations - RN101-quickgelu--yfcc15m]","[clip - mlfoundations - ViT-B-32--laion400m_e31]","[clip - mlfoundations - ViT-B-32--laion400m_e32]","[clip - mlfoundations - ViT-B-32--laion400m_avg]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e31]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e32]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_avg]","[clip - mlfoundations - ViT-B-16--laion400m_e31]","[clip - mlfoundations - ViT-B-16--laion400m_e32]","[clip - sbert - ViT-B-32-multilingual-v1]","[clip - facebookresearch - clip_small_25ep]","[simclr - facebookresearch - simclr_small_25ep]","[slip - facebookresearch - slip_small_25ep]","[slip - facebookresearch - slip_small_50ep]","[slip - facebookresearch - slip_small_100ep]","[clip - facebookresearch - clip_base_25ep]","[simclr - facebookresearch - simclr_base_25ep]","[slip - facebookresearch - slip_base_25ep]","[slip - facebookresearch - slip_base_50ep]","[slip - facebookresearch - slip_base_100ep]","[clip - facebookresearch - clip_large_25ep]","[simclr - facebookresearch - simclr_large_25ep]","[slip - facebookresearch - slip_large_25ep]","[slip - facebookresearch - slip_large_50ep]","[slip - facebookresearch - slip_large_100ep]","[clip - facebookresearch - clip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc12m_35ep]","[clip - facebookresearch - clip_base_cc12m_35ep]"] {allow-input: true}

if ViT_B32:
    clip_load_list.append("[clip - mlfoundations - ViT-B-32--openai]")
if ViT_B16:
    clip_load_list.append("[clip - mlfoundations - ViT-B-16--openai]")
if ViT_L14:
    clip_load_list.append("[clip - mlfoundations - ViT-L-14--openai]")
if RN50x4:
    clip_load_list.append("[clip - mlfoundations - RN50x4--openai]")
if RN50x64:
    clip_load_list.append("[clip - mlfoundations - RN50x64--openai]")
if RN50x16:
    clip_load_list.append("[clip - mlfoundations - RN50x16--openai]")
if ViT_L14_laion:
    clip_load_list.append("[clip - mlfoundations - ViT-L-14--laion400m_e32]")
if ViT_L14_336px:
    clip_load_list.append("[clip - mlfoundations - ViT-L-14-336--openai]")
if ViT_B16_plus:
    clip_load_list.append("[clip - mlfoundations - ViT-B-16-plus-240--laion400m_e32]")
if ViT_B32_laion2b:
    clip_load_list.append("[clip - mlfoundations - ViT-B-32--laion2b_e16]")
if clip_farsi:
    clip_load_list.append("[clip - sajjjadayobi - clipfa]")
if clip_korean:
    clip_load_list.append("[clip - navervision - kelip_ViT-B/32]")
if cloob_ViT_B16:
    clip_load_list.append("[cloob - crowsonkb - cloob_laion_400m_vit_b_16_32_epochs]")

if model1:
    clip_load_list.append(model1)
if model2:
    clip_load_list.append(model2)
if model3:
    clip_load_list.append(model3)


i = 0
temp_perceptor = MultiMMC(TEXT, IMAGE)


def get_mmc_models(clip_load_list):
    mmc_models = []
    for model_key in clip_load_list:
        if not model_key:
            continue
        arch, pub, m_id = model_key[1:-1].split(" - ")
        mmc_models.append(
            {
                "architecture": arch,
                "publisher": pub,
                "id": m_id,
            }
        )
    return mmc_models


mmc_models = get_mmc_models(clip_load_list)


normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


def set_custom_schedules(schedule):
    custom_schedules = []
    for schedule_item in schedule:
        if isinstance(schedule_item, list):
            custom_schedules.append(np.arange(*schedule_item))
        else:
            custom_schedules.append(schedule_item)

    return custom_schedules


def parse_prompt(prompt):
    if (
        prompt.startswith("http://")
        or prompt.startswith("https://")
        or prompt.startswith("E:")
        or prompt.startswith("C:")
        or prompt.startswith("D:")
    ):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


padargs = {"mode": "constant", "value": -1}
cut_blur_kernel = 3


class MakeCutouts(nn.Module):
    def __init__(
        self,
        cut_size,
        Overview=4,
        WholeCrop=0,
        WC_Allowance=10,
        WC_Grey_P=0.2,
        InnerCrop=0,
        IC_Size_Pow=0.5,
        IC_Grey_P=0.2,
        cut_blur_n=0,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.WholeCrop = WholeCrop
        self.WC_Allowance = WC_Allowance
        self.WC_Grey_P = WC_Grey_P
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.cut_blur_n = cut_blur_n
        self.augs = T.Compose(
            [
                # T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    # scale=(0.9,0.95),
                    fill=-1,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.RandomPerspective(p=1, interpolation = T.InterpolationMode.BILINEAR, fill=-1,distortion_scale=0.2),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            ]
        )

    def forward(self, input):
        gray = transforms.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [input.shape[0], 3, self.cut_size, self.cut_size]
        output_shape_2 = [input.shape[0], 3, self.cut_size + 2, self.cut_size + 2]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2 + round(max_size * 0.055),
                (sideY - max_size) // 2 + round(max_size * 0.055),
                (sideX - max_size) // 2 + round(max_size * 0.055),
                (sideX - max_size) // 2 + round(max_size * 0.055),
            ),
            **padargs,
        )
        cutouts_list = []

        if self.Overview > 0:
            cutouts = []
            cutout = resize(pad_input, out_shape=output_shape, antialiasing=True)
            output_shape_all = list(output_shape)
            output_shape_all[0] = self.Overview * input.shape[0]
            pad_input = pad_input.repeat(input.shape[0], 1, 1, 1)
            cutout = resize(pad_input, out_shape=output_shape_all)
            cutout = self.augs(cutout)
            if self.cut_blur_n > 0:
                cutout[0 : self.cut_blur_n, :, :, :] = TF.gaussian_blur(
                    cutout[0 : self.cut_blur_n, :, :, :], cut_blur_kernel
                )
            cutouts_list.append(cutout)

        if self.InnerCrop > 0:
            cutouts = []
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size)
                    + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            cutouts_tensor = torch.cat(cutouts)
            cutouts = []
            cutouts_list.append(cutouts_tensor)
        cutouts = torch.cat(cutouts_list)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


# def range_loss(input, range_min, range_max):
#    return ((input - input.clamp(range_min,range_max)).abs()*10).pow(2).mean([1, 2, 3])
def range_loss(input, range_min, range_max):
    return ((input - input.clamp(range_min, range_max)).abs()).mean([1, 2, 3])


def symmetric_loss(x):
    w = x.shape[3]
    diff = (x - torch.flip(x, [3])).square().mean().sqrt() / (
        x.shape[2] * x.shape[3] / 1e4
    )
    return diff


def fetch(url_or_path):
    """Fetches a file from an HTTP or HTTPS url, or opens the local file."""
    if str(url_or_path).startswith("http://") or str(url_or_path).startswith(
        "https://"
    ):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def to_pil_image(x):
    """Converts from a tensor to a PIL image."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)


def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub("^data:image/.+;base64,", "", base64_str)
    binary_data = base64.b64decode(base64_data)
    img_data = io.BytesIO(binary_data)
    img = Image.open(img_data)
    if image_path:
        img.save(image_path)
    return img


normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


def centralized_grad(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


def null_fn(x_in):
    return torch.zeros_like(x_in)


def make_schedule(t_start, t_end, step_size=1):
    schedule = []
    par_schedule = []
    t = t_start
    while t > t_end:
        schedule.append(t)
        t -= step_size
    schedule.append(t_end)
    return np.array(schedule)


lpips_model = lpips.LPIPS(net="vgg").to(device)


def list_mul_to_array(list_mul):
    i = 0
    mul_count = 0
    mul_string = ""
    full_list = list_mul
    full_list_len = len(full_list)
    for item in full_list:
        if i == 0:
            last_item = item
        if item == last_item:
            mul_count += 1
        if item != last_item or full_list_len == i + 1:
            mul_string = mul_string + f" [{last_item}]*{mul_count} +"
            mul_count = 1
        last_item = item
        i += 1
    return mul_string[1:-2]


class Predictor(cog.BasePredictor):
    def setup(self):

        config = OmegaConf.load(
            "/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        )  # TODO: self.optionally download from same location as ckpt and chnage this logic
        model = load_model_from_config(
            config,
            f"{model_path}/latent_diffusion_txt2img_f8_large.ckpt",
            False,
            "finetuned",
        )  # TODO: check path
        self.model = model.half().eval().to(device)

        # Alstro's aesthetic model
        self.aesthetic_model_336 = torch.nn.Linear(768, 1).cuda()
        self.aesthetic_model_336.load_state_dict(
            torch.load(f"{model_path}/ava_vit_l_14_336_linear.pth")
        )

        self.aesthetic_model_224 = torch.nn.Linear(768, 1).cuda()
        self.aesthetic_model_224.load_state_dict(
            torch.load(f"{model_path}/ava_vit_l_14_linear.pth")
        )

        self.aesthetic_model_16 = torch.nn.Linear(512, 1).cuda()
        self.aesthetic_model_16.load_state_dict(
            torch.load(f"{model_path}/ava_vit_b_16_linear.pth")
        )

        self.aesthetic_model_32 = torch.nn.Linear(512, 1).cuda()
        self.aesthetic_model_32.load_state_dict(
            torch.load(f"{model_path}/sa_0_4_vit_b_32_linear.pth")
        )

        self.opt = DotMap()

        # Change it to false to not use CLIP Guidance at all
        self.use_cond_fn = True

        # Custom cut schedules and super-resolution. Check out the guide on how to use it a https://multimodal.art/majestydiffusion
        self.custom_schedule_setting = [
            [50, 1000, 8],
            "gfpgan:1.5",
            "scale:.9",
            "noise:.55",
            [50, 200, 5],
        ]

        # Cut settings
        # clamp_index = [2.1,1.6] #linear variation of the index for clamping the gradient
        self.cut_overview = [8] * 500 + [4] * 500
        self.cut_innercut = [0] * 500 + [4] * 500
        self.cut_ic_pow = 0.2
        self.cut_icgray_p = [0.1] * 300 + [0] * 1000
        self.cutn_batches = 1
        self.cut_blur_n = [0] * 300 + [0] * 1000
        self.cut_blur_kernel = cut_blur_kernel
        self.range_index = [0] * 200 + [5e4] * 400 + [0] * 1000
        self.var_index = [2] * 300 + [0] * 700
        self.var_range = 0.5
        self.mean_index = [0] * 400 + [0] * 600
        self.mean_range = 0.75
        self.active_function = (
            "softsign"  # function to manipulate the gradient - help things to stablize
        )
        self.ths_method = "clamp"  # clamp is another self.option
        self.tv_scales = [150] * 1 + [0] * 1 + [0] * 2

        self.symmetric_loss_scale = 0  # Apply symmetric loss

        # Latent Diffusion Advanced Settings
        self.scale_div = 1  # Use when latent upscale to correct satuation problem
        self.opt_mag_mul = 20  # Magnify grad before clamping
        # PLMS Currently not working, working on a fix
        self.opt_plms = False  # Experimental. It works but does not lookg good
        self.opt_ddim_eta, self.opt_eta_end = [1.3, 1.1]  # linear variation of eta
        self.opt_temperature = 0.98

        # Grad advanced settings
        self.grad_center = False
        self.grad_scale = 0.25  # Lower value result in more coherent and detailed result, higher value makes it focus on more dominent concept

        # Restraints the model from explodign despite larger clamp
        self.score_modifier = True
        self.threshold_percentile = 0.85
        self.threshold = 1
        self.score_corrector_setting = ["latent", ""]

        # self.init image advanced settings
        self.init_rotate, self.mask_rotate = [False, False]
        self.init_magnitude = 0.18215

        # Noise settings
        self.upscale_noise_temperature = 1
        self.upscale_xT_temperature = 1

        # More settings
        self.RGB_min, self.RGB_max = [-0.95, 0.95]
        self.padargs = padargs  # How to pad the image with cut_overview
        self.flip_aug = False
        self.cutout_debug = False
        self.opt.outdir = outputs_path

        # Experimental aesthetic embeddings, work only with OpenAI ViT-B/32 and ViT-L/14
        self.experimental_aesthetic_embeddings = True
        # How mself.uch you want this to influence your result
        self.experimental_aesthetic_embeddings_weight = 0.3
        # 9 are good aesthetic embeddings, 0 are bad ones
        self.experimental_aesthetic_embeddings_score = 8

        # For fun dont change except if you really know what your are doing
        self.grad_blur = False
        self.compress_steps = 200
        self.compress_factor = 0.1
        self.punish_steps = 200
        self.punish_factor = 0.5

        self.unpurge()
        clip_load_list_universal = clip_load_list
        torch.cuda.empty_cache()
        gc.collect()

    def unpurge(self):
        (
            self.clip_model,
            self.clip_size,
            self.clip_tokenize,
            self.clip_normalize,
            self.clip_list,
        ) = self.full_clip_load(clip_load_list)

        self.has_purged = False

    def load_clip_models(self, mmc_models):
        self.clip_model, self.clip_size, self.clip_tokenize, self.clip_normalize = {}, {}, {}, {}
        clip_list = []
        for item in mmc_models:
            print("Loaded ", item["id"])
            clip_list.append(item["id"])
            model_loaders = REGISTRY.find(**item)
            for model_loader in model_loaders:
                self.clip_model_loaded = model_loader.load()
                self.clip_model[item["id"]] = MockOpenaiClip(self.clip_model_loaded)
                self.clip_size[item["id"]] = self.clip_model[
                    item["id"]
                ].visual.input_resolution
                self.clip_tokenize[item["id"]] = self.clip_model[
                    item["id"]
                ].preprocess_text()
                self.clip_normalize[item["id"]] = normalize
        return self.clip_model, self.clip_size, self.clip_tokenize, self.clip_normalize, clip_list

    def full_clip_load(self, clip_load_list):
        torch.cuda.empty_cache()
        gc.collect()
        try:
            del (
                self.clip_model,
                self.clip_size,
                self.clip_tokenize,
                self.clip_normalize,
                self.clip_list,
            )
        except:
            pass
        mmc_models = get_mmc_models(clip_load_list)
        (
            self.clip_model,
            self.clip_size,
            self.clip_tokenize,
            self.clip_normalize,
            self.clip_list,
        ) = self.load_clip_models(mmc_models)
        return (
            self.clip_model,
            self.clip_size,
            self.clip_tokenize,
            self.clip_normalize,
            self.clip_list,
        )

    def predict(
        self,
        Prompt: str = cog.Input(description="Your text prompt.", default=""),
        latent_diffusion_model: str = cog.Input(
            description='Original is the previous LAION model. Finetuned should be better but cannot do text. One of:  ["original", "finetuned", "ongo (fine tuned in paintings)", "erlich (fine tuned in logos)"]',
            default="finetuned",
        ),  # @param ["original", "finetuned", "ongo (fine tuned in paintings)", "erlich (fine tuned in logos)"]
        latent_diffusion_guidance_scale: int = cog.Input(
            description="Balance between creativity and coherent composition. Try values between 0-15. Lower values help with text interpretation and creativity, higher help with composition. ",
            default=12,
        ),
    ) -> None:
        text_prompt = Prompt
        clip_prompts = [text_prompt]

        # Prompt for Latent Diffusion
        latent_prompts = [text_prompt]

        # Negative prompts for Latent Diffusion
        latent_negatives = [""]

        image_prompts = []

        width = 256  # @param{type: 'integer'}
        height = 256  # @param{type: 'integer'}

        # @markdown The `clamp_index` will determine how much of the `clip_prompts` affect the image, it is a linear scale that will decrease from the first to the second value. Try values between 3-1
        clamp_index = [2.4, 2.1]  # @param{type: 'raw'}
        clip_guidance_scale = 16000  # @param{type: 'integer'}
        how_many_batches = 1  # @param{type: 'integer'}
        aesthetic_loss_scale = 400  # @param{type: 'integer'}
        augment_cuts = True  # @param{type:'boolean'}

        # @markdown

        # @markdown  ### Init image settings
        # @markdown `init_image` requires the path of an image to use as init to the model
        self.init_image = None  # @param{type: 'string'}
        # @markdown `starting_timestep`: How much noise do you want to add to your init image for it to then be difused by the model
        starting_timestep = 0.9  # @param{type: 'number'}
        # @markdown `init_mask` is a mask same width and height as the original image with the color black indicating where to inpaint
        self.init_mask = None  # @param{type: 'string'}
        # @markdown `init_scale` controls how much the init image should influence the final result. Experiment with values around `1000`
        self.init_scale = 1000  # @param{type: 'integer'}
        self.init_brightness = 0.0  # @param{type: 'number'}
        #  @markdown How much extra noise to add to the init image, independently from skipping timesteps (use it also if you are upscaling)
        # @markdown ### Custom saved settings
        # @markdown If you choose custom saved settings, the settings set by the preset overrule some of your choices. You can still modify the settings not in the preset. <a href="https://github.com/multimodalart/majesty-diffusion/tree/main/latent_settings_library">Check what each preset modifies here</a>

        prompts = clip_prompts

        self.opt.prompt = latent_prompts
        self.opt.uc = latent_negatives
        custom_schedules = set_custom_schedules(self.custom_schedule_setting)
        self.aes_scale = aesthetic_loss_scale
        self.clip_guidance_index = [clip_guidance_scale] * 1000

        self.image_grid = None

        for n in trange(how_many_batches, desc="Sampling"):
            print(f"Sampling images {n+1}/{how_many_batches}")
            self.opt.W = (width // 64) * 64
            self.opt.H = (height // 64) * 64
            if self.opt.W != width or self.opt.H != height:
                print(
                    f"Changing output size to {self.opt.W}x{self.opt.H}. Dimensions must by multiples of 64."
                )

            self.opt.mag_mul = self.opt_mag_mul
            self.opt.ddim_eta = self.opt_ddim_eta
            self.opt.eta_end = self.opt_eta_end
            self.opt.temperature = self.opt_temperature

            self.opt.scale = latent_diffusion_guidance_scale
            self.opt.plms = self.opt_plms
            aug = augment_cuts

            # Checks if it's not a normal schedule (legacy purposes to keep old configs compatible)
            if len(clamp_index) == 2:
                self.clamp_index_variation = np.linspace(
                    clamp_index[0], clamp_index[1], 1000
                )
            else:
                self.clamp_index_variation = clamp_index

            score_corrector = DotMap()

            def modify_score(e_t, e_t_uncond):
                if self.score_modifier is False:
                    return e_t
                else:
                    e_t_d = e_t - e_t_uncond
                    s = torch.quantile(
                        rearrange(e_t_d, "b ... -> b (...)").abs().float(),
                        self.threshold_percentile,
                        dim=-1,
                    )

                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (e_t_d.ndim - 1)))
                if self.ths_method == "softsign":
                    e_t_d = F.softsign(e_t_d) / s
                elif self.ths_method == "clamp":
                    e_t_d = e_t_d.clamp(-s, s) / s * 1.3  # 1.2
                e_t = e_t_uncond + e_t_d
                return e_t

            score_corrector.modify_score = modify_score

            def dynamic_thresholding(pred_x0, t):
                return pred_x0

            self.opt.n_iter = 1  # Old way for batching, avoid touching
            self.opt.n_samples = 1  # How many implaes in parallel. Breaks upscaling
            torch.cuda.empty_cache()
            gc.collect()

        if self.has_purged:
            self.unpurge()

        self.cur_step = 0
        self.scale_factor = 1
        make_cutouts = {}
        for i in self.clip_list:
            make_cutouts[i] = MakeCutouts(
                self.clip_size[i][0]
                if type(self.clip_size[i]) is tuple
                else self.clip_size[i],
                Overview=1,
            )
        self.target_embeds, self.weights, self.zero_embed = {}, {}, {}
        for i in self.clip_list:
            self.target_embeds[i] = []
            self.weights[i] = []

        for prompt in prompts:
            txt, weight = parse_prompt(prompt)
            for i in self.clip_list:
                if "cloob" not in i:
                    with torch.cuda.amp.autocast():
                        embeds = self.clip_model[i].encode_text(
                            self.clip_tokenize[i](txt).to(device)
                        )
                        self.target_embeds[i].append(embeds)
                        self.weights[i].append(weight)
                else:
                    embeds = self.clip_model[i].encode_text(
                        self.clip_tokenize[i](txt).to(device)
                    )
                    self.target_embeds[i].append(embeds)
                    self.weights[i].append(weight)

        for prompt in image_prompts:
            if prompt.startswith("data:"):
                img = base64_to_image(prompt).convert("RGB")
                weight = 1
            else:
                print(f"processing{prompt}", end="\r")
                path, weight = parse_prompt(prompt)
                img = Image.open(fetch(path)).convert("RGB")
            img = TF.resize(
                img,
                min(self.opt.W, self.opt.H, *img.size),
                transforms.InterpolationMode.LANCZOS,
            )
            for i in self.clip_list:
                if "cloob" not in i:
                    with torch.cuda.amp.autocast():
                        batch = make_cutouts[i](
                            TF.to_tensor(img).unsqueeze(0).to(device)
                        )
                        embed = self.clip_model[i].encode_image(
                            self.clip_normalize[i](batch)
                        )
                        self.target_embeds[i].append(embed)
                        self.weights[i].extend([weight])
                else:
                    batch = make_cutouts[i](TF.to_tensor(img).unsqueeze(0).to(device))
                    embed = self.clip_model[i].encode_image(
                        self.clip_normalize[i](batch)
                    )
                    self.target_embeds[i].append(embed)
                    self.weights[i].extend([weight])
        # if anti_jpg != 0:
        #    self.target_embeds["ViT-B-32--openai"].append(torch.tensor([np.load(f"{model_path}/openimages_512x_png_embed224.npz")['arr_0']-np.load(f"{model_path}/imagenet_512x_jpg_embed224.npz")['arr_0']], device = device))
        #    self.weights["ViT-B-32--openai"].append(anti_jpg)

        for i in self.clip_list:
            self.target_embeds[i] = torch.cat(self.target_embeds[i])
            self.weights[i] = torch.tensor([self.weights[i]], device=device)
        shape = [4, self.opt.H // 8, self.opt.W // 8]
        self.init = None
        mask = None
        transform = T.GaussianBlur(kernel_size=3, sigma=0.4)
        if self.init_image is not None:
            if self.init_image.startswith("data:"):
                img = base64_to_image(self.init_image).convert("RGB")
            else:
                img = Image.open(fetch(self.init_image)).convert("RGB")
            self.init = TF.to_tensor(img).to(device).unsqueeze(0)
            if self.init_rotate:
                self.init = torch.rot90(self.init, 1, [3, 2])
            x0_original = torch.tensor(self.init)
            self.init = resize(
                self.init, out_shape=[self.opt.n_samples, 3, self.opt.H, self.opt.W]
            )
            self.init = self.init.mul(2).sub(1).half()
            self.init_encoded = (
                self.model.first_stage_model.encode(self.init).sample()
                * self.init_magnitude
                + self.init_brightness
            )
            # self.init_encoded = self.init_encoded + noise_like(self.init_encoded.shape,device,False).mul(self.init_noise)
            upscaled_flag = True
        else:
            self.init = None
            self.init_encoded = None
            upscale_flag = False
        if self.init_mask is not None:
            mask = Image.open(fetch(self.init_mask)).convert("RGB")
            mask = TF.to_tensor(mask).to(device).unsqueeze(0)
            if self.mask_rotate:
                mask = torch.rot90(mask, 1, [3, 2])
            mask = F.interpolate(mask, [self.opt.H // 8, self.opt.W // 8]).mean(1)
            mask = transform(mask)
            print(mask)

        if self.opt.plms:
            sampler = PLMSSampler(self.model)
        else:
            sampler = DDIMSampler(self.model)

        os.makedirs(self.opt.outdir, exist_ok=True)
        outpath = self.opt.outdir

        prompt = self.opt.prompt
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        all_samples = list()
        last_step_upscale = False
        eta1 = self.opt.ddim_eta
        eta2 = self.opt.eta_end
        with torch.enable_grad():
            with torch.cuda.amp.autocast():
                with self.model.ema_scope():
                    self.uc = None
                    if self.opt.scale != 1.0:
                        self.uc = self.model.get_learned_conditioning(
                            self.opt.n_samples * self.opt.uc
                        ).cuda()

                    for n in range(self.opt.n_iter):
                        torch.cuda.empty_cache()
                        gc.collect()
                        self.c = self.model.get_learned_conditioning(
                            self.opt.n_samples * prompt
                        ).cuda()
                        if self.init_encoded is None:
                            x_T = torch.randn(
                                [self.opt.n_samples, *shape], device=device
                            )
                            upscaled_flag = False
                            x0 = None
                        else:
                            x_T = self.init_encoded
                            x0 = torch.tensor(x_T)
                            upscaled_flag = True
                        last_step_uspcale_list = []
                        diffusion_stages = 0
                        for custom_schedule in custom_schedules:
                            if type(custom_schedule) != type(""):
                                diffusion_stages += 1
                                torch.cuda.empty_cache()
                                gc.collect()
                                last_step_upscale = False
                                samples_ddim, _ = sampler.sample(
                                    S=self.opt.ddim_steps,
                                    conditioning=self.c,
                                    batch_size=self.opt.n_samples,
                                    shape=shape,
                                    custom_schedule=custom_schedule,
                                    verbose=False,
                                    unconditional_guidance_scale=self.opt.scale,
                                    unconditional_conditioning=self.uc,
                                    eta=eta1
                                    if diffusion_stages == 1 or last_step_upscale
                                    else eta2,
                                    eta_end=eta2,
                                    img_callback=None
                                    if self.use_cond_fn
                                    else self.display_handler,
                                    cond_fn=self.cond_fn if self.use_cond_fn else None,
                                    temperature=self.opt.temperature,
                                    x_adjust_fn=self.cond_clamp,
                                    x_T=x_T,
                                    x0=x0,
                                    mask=mask,
                                    score_corrector=score_corrector,
                                    corrector_kwargs=self.score_corrector_setting,
                                    x0_adjust_fn=dynamic_thresholding,
                                    clip_embed=self.target_embeds["ViT-L-14--openai"]
                                    if "ViT-L-14--openai" in self.clip_list
                                    else None,
                                )
                                # x_T = samples_ddim.clamp(-6,6)
                                x_T = samples_ddim
                                last_step_upscale = False
                            else:
                                torch.cuda.empty_cache()
                                gc.collect()
                                method, self.scale_factor = custom_schedule.split(":")
                                if method == "RGB":
                                    self.scale_factor = float(self.scale_factor)
                                    temp_file_name = (
                                        "temp_" + f"{str(round(time.time()))}.png"
                                    )
                                    temp_file = os.path.join(
                                        sample_path, temp_file_name
                                    )
                                    self.im.save(temp_file, format="PNG")
                                    self.init = Image.open(fetch(temp_file)).convert(
                                        "RGB"
                                    )
                                    self.init = (
                                        TF.to_tensor(self.init).to(device).unsqueeze(0)
                                    )
                                    self.opt.H, self.opt.W = (
                                        self.opt.H * self.scale_factor,
                                        self.opt.W * self.scale_factor,
                                    )
                                    self.init = resize(
                                        self.init,
                                        out_shape=[
                                            self.opt.n_samples,
                                            3,
                                            self.opt.H,
                                            self.opt.W,
                                        ],
                                        antialiasing=True,
                                    )
                                    self.init = self.init.mul(2).sub(1).half()
                                    x_T = (
                                        self.model.first_stage_model.encode(
                                            self.init
                                        ).sample()
                                        * self.init_magnitude
                                    )
                                    upscaled_flag = True
                                    last_step_upscale = True
                                    # x_T += noise_like(x_T.shape,device,False)*self.init_noise
                                    # x_T = x_T.clamp(-6,6)
                                if method == "gfpgan":
                                    self.scale_factor = float(self.scale_factor)
                                    last_step_upscale = True
                                    temp_file_name = (
                                        "temp_" + f"{str(round(time.time()))}.png"
                                    )
                                    temp_file = os.path.join(
                                        sample_path, temp_file_name
                                    )
                                    self.im.save(temp_file, format="PNG")
                                    GFP_factor = 2 if self.scale_factor > 1 else 1
                                    GFP_ver = 1.3  # if GFP_factor == 1 else 1.2

                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    os.system(
                                        f"cd /GFPGAN && python inference_gfpgan.py -i {temp_file} -o results -v {GFP_ver} -s {GFP_factor}"
                                    )
                                    face_corrected = Image.open(
                                        fetch(
                                            f"/GFPGAN/results/restored_imgs/{temp_file_name}"
                                        )
                                    )
                                    with io.BytesIO() as output:
                                        face_corrected.save(output, format="PNG")
                                    self.init = Image.open(
                                        fetch(
                                            f"/GFPGAN/results/restored_imgs/{temp_file_name}"
                                        )
                                    ).convert("RGB")
                                    self.init = (
                                        TF.to_tensor(self.init).to(device).unsqueeze(0)
                                    )
                                    self.opt.H, self.opt.W = (
                                        self.opt.H * self.scale_factor,
                                        self.opt.W * self.scale_factor,
                                    )
                                    self.init = resize(
                                        self.init,
                                        out_shape=[
                                            self.opt.n_samples,
                                            3,
                                            self.opt.H,
                                            self.opt.W,
                                        ],
                                        antialiasing=True,
                                    )
                                    self.init = self.init.mul(2).sub(1).half()
                                    x_T = (
                                        self.model.first_stage_model.encode(
                                            self.init
                                        ).sample()
                                        * self.init_magnitude
                                    )
                                    upscaled_flag = True
                                    # x_T += noise_like(x_T.shape,device,False)*self.init_noise
                                    # x_T = x_T.clamp(-6,6)
                                if method == "scale":
                                    self.scale_factor = float(self.scale_factor)
                                    x_T = x_T * self.scale_factor
                                if method == "noise":
                                    self.scale_factor = float(self.scale_factor)
                                    x_T += (
                                        noise_like(x_T.shape, device, False)
                                        * self.scale_factor
                                    )
                                if method == "purge":
                                    self.has_purged = True
                                    for i in self.scale_factor.split(","):
                                        if i in clip_load_list:
                                            arch, pub, m_id = i[1:-1].split(" - ")
                                            print("Purge ", i)
                                            del self.clip_list[
                                                self.clip_list.index(m_id)
                                            ]
                                            del self.clip_model[m_id]
                                            del self.clip_size[m_id]
                                            del self.clip_tokenize[m_id]
                                            del self.clip_normalize[m_id]
                        # last_step_uspcale_list.append(last_step_upscale)
                        self.scale_factor = 1
                        current_time = str(round(time.time()))
                        if last_step_upscale and method == "gfpgan":
                            latest_upscale = Image.open(
                                fetch(f"/GFPGAN/results/restored_imgs/{temp_file_name}")
                            ).convert("RGB")
                            latest_upscale.save(
                                os.path.join(outpath, f"{current_time}.png"),
                                format="PNG",
                            )
                        else:
                            Image.fromarray(self.image_grid.astype(np.uint8)).save(
                                os.path.join(outpath, f"{current_time}.png"),
                                format="PNG",
                            )

                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        all_samples.append(x_samples_ddim)

        if len(all_samples) > 1:
            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, "n b c h w -> (n b) c h w")
            grid = make_grid(grid, nrow=self.opt.n_samples)

            # to image
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(
                os.path.join(outputs_path, f"output.png")
            )

    def display_handler(self, x, i, cadance=5, decode=True):
        # global  image_grid, writer, img_tensor, im
        img_tensor = x
        if i % cadance == 0:
            if decode:
                x = self.model.decode_first_stage(x)
            grid = make_grid(
                torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0),
                round(x.shape[0] ** 0.5 + 0.2),
            )
            grid = 255.0 * rearrange(grid, "c h w -> h w c").detach().cpu().numpy()
            self.image_grid = grid.copy(order="C")
            # with io.BytesIO() as output:
            self.im = Image.fromarray(grid.astype(np.uint8))
            self.im.save(os.path.join(outputs_path, f"output.png"))
            # if generate_video:
            #    im.save(p.stdin, 'PNG')

    def cond_fn(self, x, t):
        self.cur_step += 1
        t = 1000 - t
        t = t[0]
        x = x.detach()
        with torch.enable_grad():
            global clamp_start_, clamp_max
            x = x.requires_grad_()
            x_in = self.model.decode_first_stage(x)
            self.display_handler(x_in, t, 1, False)
            n = x_in.shape[0]
            clip_guidance_scale = self.clip_guidance_index[t]
            make_cutouts = {}
            # rx_in_grad = torch.zeros_like(x_in)
            for i in self.clip_list:
                make_cutouts[i] = MakeCutouts(
                    self.clip_size[i][0]
                    if type(self.clip_size[i]) is tuple
                    else self.clip_size[i],
                    Overview=self.cut_overview[t],
                    InnerCrop=self.cut_innercut[t],
                    IC_Size_Pow=self.cut_ic_pow,
                    IC_Grey_P=self.cut_icgray_p[t],
                    cut_blur_n=self.cut_blur_n[t],
                )
                cutn = self.cut_overview[t] + self.cut_innercut[t]
            for j in range(self.cutn_batches):
                losses = 0
                for i in self.clip_list:
                    clip_in = self.clip_normalize[i](
                        make_cutouts[i](x_in.add(1).div(2)).to("cuda")
                    )
                    image_embeds = (
                        self.clip_model[i]
                        .encode_image(clip_in)
                        .float()
                        .unsqueeze(0)
                        .expand([self.target_embeds[i].shape[0], -1, -1])
                    )
                    self.target_embeds_temp = self.target_embeds[i]
                    if (
                        i == "ViT-B-32--openai"
                        and self.experimental_aesthetic_embeddings
                    ):
                        aesthetic_embedding = torch.from_numpy(
                            np.load(
                                f"/aesthetic-predictor/vit_b_32_embeddings/rating{self.experimental_aesthetic_embeddings_score}.npy"
                            )
                        ).to(device)
                        aesthetic_query = (
                            self.target_embeds_temp
                            + aesthetic_embedding
                            * self.experimental_aesthetic_embeddings_weight
                        )
                        self.target_embeds_temp = (aesthetic_query) / torch.linalg.norm(
                            aesthetic_query
                        )
                    if (
                        i == "ViT-L-14--openai"
                        and self.experimental_aesthetic_embeddings
                    ):
                        aesthetic_embedding = torch.from_numpy(
                            np.load(
                                f"/aesthetic-predictor/vit_l_14_embeddings/rating{self.experimental_aesthetic_embeddings_score}.npy"
                            )
                        ).to(device)
                        aesthetic_query = (
                            self.target_embeds_temp
                            + aesthetic_embedding
                            * self.experimental_aesthetic_embeddings_weight
                        )
                        self.target_embeds_temp = (aesthetic_query) / torch.linalg.norm(
                            aesthetic_query
                        )
                    self.target_embeds_temp = self.target_embeds_temp.unsqueeze(
                        1
                    ).expand([-1, cutn * n, -1])
                    dists = spherical_dist_loss(image_embeds, self.target_embeds_temp)
                    dists = dists.mean(1).mul(self.weights[i].squeeze()).mean()
                    losses += (
                        dists * clip_guidance_scale
                    )  # * (2 if i in ["ViT-L-14-336--openai", "RN50x64--openai", "ViT-B-32--laion2b_e16"] else (.4 if "cloob" in i else 1))
                    if i == "ViT-L-14-336--openai" and self.aes_scale != 0:
                        aes_loss = (
                            self.aesthetic_model_336(F.normalize(image_embeds, dim=-1))
                        ).mean()
                        losses -= aes_loss * self.aes_scale
                    if i == "ViT-L-14--openai" and self.aes_scale != 0:
                        aes_loss = (
                            self.aesthetic_model_224(F.normalize(image_embeds, dim=-1))
                        ).mean()
                        losses -= aes_loss * self.aes_scale
                    if i == "ViT-B-16--openai" and self.aes_scale != 0:
                        aes_loss = (
                            self.aesthetic_model_16(F.normalize(image_embeds, dim=-1))
                        ).mean()
                        losses -= aes_loss * self.aes_scale
                    if i == "ViT-B-32--openai" and self.aes_scale != 0:
                        aes_loss = (
                            self.aesthetic_model_32(F.normalize(image_embeds, dim=-1))
                        ).mean()
                        losses -= aes_loss * self.aes_scale
                # x_in_grad += torch.autograd.grad(losses, x_in)[0] / cutn_batches / len(self.clip_list)
                # losses += dists
                # losses = losses / len(self.clip_list)
                # gc.collect()

            loss = losses
            # del losses
            if self.symmetric_loss_scale != 0:
                loss += symmetric_loss(x_in) * self.symmetric_loss_scale
            if self.init_image is not None and self.init_image:
                lpips_loss = (
                    (lpips_model(x_in, self.init) * self.init_scale).squeeze().mean()
                )
                # print(lpips_loss)
                loss += lpips_loss
            range_scale = self.range_index[t]
            range_losses = (
                range_loss(x_in, self.RGB_min, self.RGB_max).sum() * range_scale
            )
            loss += range_losses
            # loss_grad = torch.autograd.grad(loss, x_in, )[0]
            # x_in_grad += loss_grad
            # grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            loss.backward()
            grad = -x.grad
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0, neginf=0)
            if self.grad_center:
                grad = centralized_grad(grad, use_gc=True, gc_conv_only=False)
            mag = grad.square().mean().sqrt()
            if mag == 0 or torch.isnan(mag):
                print("ERROR")
                print(t)
                return grad
            if t >= 0:
                if self.active_function == "softsign":
                    grad = F.softsign(grad * self.grad_scale / mag)
                if self.active_function == "tanh":
                    grad = (grad / mag * self.grad_scale).tanh()
                if self.active_function == "clamp":
                    grad = grad.clamp(
                        -mag * self.grad_scale * 2, mag * self.grad_scale * 2
                    )
            if grad.abs().max() > 0:
                grad = grad / grad.abs().max() * self.opt.mag_mul
                magnitude = grad.square().mean().sqrt()
            else:
                return grad
            clamp_max = self.clamp_index_variation[t]
            # print(magnitude, end = "\r")
            grad = grad * magnitude.clamp(max=clamp_max) / magnitude  # 0.2
            grad = grad.detach()
            grad = self.grad_fn(grad, t)
            x = x.detach()
            x = x.requires_grad_()
            var = x.var()
            var_scale = self.var_index[t]
            var_losses = (var.pow(2).clamp(min=self.var_range) - 1) * var_scale
            mean_scale = self.mean_index[t]
            mean_losses = (x.mean().abs() - self.mean_range).abs().clamp(
                min=0
            ) * mean_scale
            tv_losses = (
                tv_loss(x).sum() * self.tv_scales[0]
                + tv_loss(F.interpolate(x, scale_factor=1 / 2)).sum()
                * self.tv_scales[1]
                + tv_loss(F.interpolate(x, scale_factor=1 / 4)).sum()
                * self.tv_scales[2]
                + tv_loss(F.interpolate(x, scale_factor=1 / 8)).sum()
                * self.tv_scales[3]
            )
            adjust_losses = tv_losses + var_losses + mean_losses
            adjust_losses.backward()
            grad -= x.grad
            # print(grad.abs().mean(), x.grad.abs().mean(), end = "\r")
        return grad

    def grad_fn(self, x, t):
        if t <= 500 and self.grad_blur:
            x = TF.gaussian_blur(
                x, 2 * round(int(max(self.grad_blur - t / 150, 1))) - 1, 1.5
            )
        return x

    def cond_clamp(self, image, t):
        t = 1000 - t[0]
        if t <= max(self.punish_steps, self.compress_steps):
            s = torch.quantile(
                rearrange(image, "b ... -> b (...)").abs(),
                self.threshold_percentile,
                dim=-1,
            )
            s = s.view(-1, *((1,) * (image.ndim - 1)))
            ths = s.clamp(min=self.threshold)
            im_max = image.clamp(min=ths) - image.clamp(min=ths, max=ths)
            im_min = image.clamp(max=-ths, min=-ths) - image.clamp(max=-ths)
        if t <= self.punish_steps:
            image = (
                image.clamp(min=-ths, max=ths) + (im_max - im_min) * self.punish_factor
            )  # ((im_max-im_min)*punish_factor).tanh()/punish_factor
        if t <= self.compress_steps:
            image = image / (ths / self.threshold) ** self.compress_factor
            image += noise_like(image.shape, device, False) * (
                (ths / self.threshold) ** self.compress_factor - 1
            )
        return image
