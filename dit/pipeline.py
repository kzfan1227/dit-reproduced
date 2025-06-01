import torch
import numpy as np
from dit.dit import DiffusionTransformer, DiffusionTransformerModel
from dit.encoder import VAE_Encoder
from dit.decoder import VAE_Decoder
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
import torch.nn.functional as F
import argparse

def generate(
        ckpt_path: str,
        image_size: int,
        class_labels,
        cfg_scale: float,
        num_sampling_steps: int,
        output_path: str,
        seed: int,
        ):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = image_size // 8
    dit = DiffusionTransformer(
        patch_size=8,
        num_attention_heads=6,
        num_blocks=12,
    )
    model = DiffusionTransformerModel(dit, time_embed_dim=256).to(device)

    # Load checkpoint if provided
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

    diffusion = create_diffusion(str(num_sampling_steps))

    # Setup VAE
    vae_weights = "stabilityai/sd-vae-ft-ema"
    vae = AutoencoderKL.from_pretrained(vae_weights).to(device)
    vae.eval()
    
    # Convert class labels to tensor and one-hot encode
    if isinstance(class_labels, (list, tuple)):
        class_labels = torch.tensor(class_labels, device=device)
    elif isinstance(class_labels, int):
        class_labels = torch.tensor([class_labels], device=device)
    
    # Convert to one-hot encoding (B, 1000)
    y = class_labels.long()
    
    # create sampling noise, latent sized
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance
    z = torch.cat([z, z], 0)
    # Create null class labels (B, 1000) with all zeros
    y_null = torch.zeros(n, 1000, device=device)
    y = torch.cat([F.one_hot(y,1000).float(), y_null],0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Generate samples
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    )
    samples, _ = samples.chunk(2, dim=0) #remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    save_image(samples, output_path, nrow=4, normalize=True, value_range=(-1,1))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        type=str,   required=False)
    p.add_argument("--image-size",  type=int,   default=256)
    p.add_argument("--classes",     nargs="+",  type=int, default=[207,360,387,974,88,979,417,279])
    p.add_argument("--cfg-scale",   type=float, default=4.0)
    p.add_argument("--steps",       type=int,   default=250)
    p.add_argument("--output",      type=str,   default="sample.png")
    p.add_argument("--seed",        type=int,   default=0)
    args = p.parse_args()

    generate(
        ckpt_path=args.ckpt,
        image_size=args.image_size,
        class_labels=args.classes,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.steps,
        output_path=args.output,
        seed=args.seed,
    )

