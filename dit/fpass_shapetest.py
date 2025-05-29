import torch
import numpy as np
from dit.dit import DiffusionTransformer, DiffusionTransformerModel
from dit.encoder import VAE_Encoder
from dit.decoder import VAE_Decoder

def get_time_embedding(timestep):
    # (128, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=128, dtype=torch.float32) / 128)
    # (1, 128)
    x = torch.tensor([timestep], dtype=torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    # (1, 256)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def test_forward():
    batch_size = 1
    image_size = 256
    num_channels = 3
    latent_size = 32

    #latent = torch.randn(batch_size, num_channels, latent_size, latent_size)
    image = torch.randn(batch_size, num_channels, image_size, image_size)
    time_tensor = get_time_embedding(1)
    class_tensor = torch.zeros(batch_size, 1000)
    class_tensor[:,0]=1

    encoder = VAE_Encoder()
    decoder = VAE_Decoder()

    dit = DiffusionTransformer(
        patch_size=8,
        num_attention_heads=6,
        num_blocks=12,
    )

    model = DiffusionTransformerModel(dit)

    try:
        noise = torch.randn(batch_size, 4, latent_size, latent_size)
        latent = encoder(image, noise)
        print(f"Encoder output shape: {latent.shape}")

        output = model(latent, time_tensor, class_tensor)
        print(f"Successful DIT forward pass, output shape {output.shape}")

        decoded = decoder(latent)
        print(f"Decoder output shape: {decoded.shape}")

        print("Successful fpass!")
    except Exception as e:
        print(f"Error {e} encountered")

if __name__ == "__main__":
    test_forward()

