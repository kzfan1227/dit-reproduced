import torch, time
from dit.pipeline import generate

t0 = time.time()
# all zeros means random weights – just check shape & no crash
generate(ckpt_path=None,
         image_size=256,
         class_labels=[0,1],
         cfg_scale=1.0,
         num_sampling_steps=2,
         output_path="tmp.png",
         seed=42)
print("✓ forward OK in", round(time.time()-t0,1),"s")