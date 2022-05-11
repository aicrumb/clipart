import clip
from clipart import CLIPart
from clipart.metrics import sph_dist 
import torch 
from torch import optim, nn
from tqdm.auto import trange
from torchvision import transforms
import wandb
import sys
from PIL import Image
wandb.init(project="guide-optimization")
device = "cuda" if torch.cuda.is_available() else "cpu"

stages = 10
resolution = (64,64)
steps = int(250/stages)
clip_model = clip.load("ViT-B/32",jit=False)[0].to(device)

timeout = 0.015  # sleep a little, dont run everything so fast, my computer will overheat doing this for extended time
render = CLIPart(clip_model=clip_model, device=device, steps=steps, cutn=2, verbose=False, timeout=timeout, lr=0.15)
render_norm = CLIPart(clip_model=clip_model, device=device, steps=steps*stages, cutn=2, verbose=False, timeout=timeout)
prompt_o = sys.argv[1]
print(f"Baking {prompt_o}")
prompt = clip_model.encode_text(clip.tokenize(prompt_o).to(device))
z = torch.randn((1,3,resolution[0],resolution[1]), requires_grad=True, device=device)
_, norm = render_norm(prompt, z)
del render_norm


fix = transforms.Resize(224)
guides = [torch.randn((512), device=device) for i in range(stages)]
for i in range(len(guides)):
    guides[i].requires_grad = True
guide_opt = optim.Adam(guides, 0.01)

steps = 100
for step in trange(steps):
    z = torch.randn((1,3,resolution[0],resolution[1]), requires_grad=True, device=device)
    loss = 0
    for i in range(len(guides)):
        z, pz = render(guides[i], z)
        loss+=(sph_dist(guides[i], prompt)/(len(guides)))*(1-step/steps)*(1-i/len(guides))
    z_emb = clip_model.encode_image(fix(z))
    loss += sph_dist(z_emb, prompt)

    guide_opt.zero_grad()
    loss.backward(retain_graph=True) 
    guide_opt.step()

    # idk why the image isnt updating
    pz.save("output.png")
    pz = Image.open("output.png")
    wandb.log({"loss": loss.item(), "output": [wandb.Image(pz, caption=prompt_o+" (ex)"), wandb.Image(norm, caption=prompt_o+" (og)")]})

name=prompt_o.replace(" ", "_")
torch.save(torch.stack(guides), f"baked/{name}.pt")