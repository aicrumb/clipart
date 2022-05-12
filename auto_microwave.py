import sys
import clip
from clipart import CLIPart
from clipart.metrics import sph_dist
import torch 
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = sys.argv[1]
clip_model = clip.load("ViT-B/32", jit=False)[0].to(device)
embedding = clip_model.encode_text(clip.tokenize(prompt).to(device))

paths = os.listdir("baked")
names = [path.replace("_", " ").replace(".pt", "") for path in paths]
embeddings = [clip_model.encode_text(clip.tokenize(name).to(device)) for name in names]
dists = [sph_dist(embedding, i).item() for i in embeddings]
best_ind = dists.index(min(dists))
path = torch.load("baked/"+paths[best_ind])
print(f"microwaving '{prompt}' with '{paths[best_ind]}'")

steps = 25
stages = [i for i in path]
render = CLIPart(steps=steps, clip_model=clip_model, device=device, lr=0.12)
def lerp(x,y,t):
    return (1 - t) * x + t * y
z = torch.randn((1,3,64,64), device=device, requires_grad=True)

for stage in stages:
	if prompt:
		z, pz = render(lerp(stage, embedding, .3), z)
	else:
		z, pz = render(stage, z)

pz.resize((256,256)).save("output.png")