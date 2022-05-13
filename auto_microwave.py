import sys
import clip
from clipart import CLIPart
from clipart.metrics import sph_dist
import torch 
import os
import random
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = sys.argv[1]
clip_model = clip.load("ViT-B/32", jit=False)[0].to(device)
embedding = clip_model.encode_text(clip.tokenize(prompt).to(device))

paths = os.listdir("baked")
names = [path.replace("_", " ").replace(".pt", "") for path in paths]
embeddings = [clip_model.encode_text(clip.tokenize(name).to(device)) for name in names]

dists = [sph_dist(embedding, i).item() for i in embeddings]
dists_clone = [i for i in dists]
dists_clone.sort()


to_sample = 2

best = dists_clone[:to_sample]
best = [paths[dists.index(b)] for b in best]
print(best)
best = [torch.load("baked/"+i) for i in best]
print([len(i) for i in best])
steps = 25
render = CLIPart(steps=steps, clip_model=clip_model, device=device, lr=0.12)
def lerp(x,y,t):
    return (1 - t) * x + t * y

z = torch.randn((1,3,128,128), device=device, requires_grad=True)
lengths = [len(i) for i in best]
for step in range(min(lengths)):
	# stage = best[random.randint(0,to_sample-1)][step]
	stage = torch.cat([i.unsqueeze(0) for i in best]).mean()
	z, pz = render(lerp(stage, embedding, .7), z)

pz.resize((256,256)).save("output.png")