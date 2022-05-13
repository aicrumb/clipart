from clipart import CLIPart
import clip 
import torch
import sys
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip.load("ViT-B/32", jit=False)[0]
z = torch.randn((1, 3, 32, 32), device=device, requires_grad=True)
prompt = clip_model.encode_text(clip.tokenize(sys.argv[1]).to(device))
z, pz = CLIPart(steps=250, clip_model=clip_model, device=device, lr=0.12)(prompt, z)
pz.resize((256,256)).save("output.png")