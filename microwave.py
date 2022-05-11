import sys
import clip
from clipart import CLIPart
import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = sys.argv[1] if ".pth" not in sys.argv[1] else None
path = sys.argv[2] if prompt else sys.argv[1]
path = torch.open(path).to(device)
steps = 25
stages = [i for i in path]
clip_model = clip.load("ViT-B/32", jit=False)[0].to(device)
render = CLIPart(steps=steps, clip_model=clip_model, device=device, lr=0.12)
def lerp(x,y,t):
    return (1 - t) * x + t * y
z = torch.randn((512), device=device, requires_grad=True)

if prompt:
	embedding = clip_model.encode_text(clip.tokenize(prompt).to(device))

for stage in stages:
	if prompt:
		z, pz = render(lerp(stage, embedding, .5), z)
	else
		z, pz = render(stage, z)
		
pz.save("output.png")