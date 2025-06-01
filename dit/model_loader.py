import torch
from diffusers.models import AutoencoderKL
from dit.dit import DiffusionTransformer, DiffusionTransformerModel

def load_pretrained_model(
        image_size=256,
        vae_model="stabilityai/sd-vae-ft-ema",
        device="cpu"):
    
    latent_size = int(image_size) // 8
    
    dit = DiffusionTransformer(
        patch_size=8,
        num_attention_heads=6,
        num_blocks=12,
    )

    model = DiffusionTransformerModel(dit)

    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    with torch.no_grad():
        latent = vae.encode(dummy).latent_dist.sample()
        recon = vae.decode(latent).sample
    print(f"latent shape: {latent.shape}, recon shape: {recon.shape}")

    return model, vae