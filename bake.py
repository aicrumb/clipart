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
wandb.init(project="10STAGE-metaclip")
device = "cuda" if torch.cuda.is_available() else "cpu"

stages = 10
resolution = (128,128)
steps = int(300/stages)
clip_model = clip.load("ViT-B/32",jit=False)[0].to(device)

timeout = 0.01  # sleep a little, dont run everything so fast, my computer will overheat doing this for extended time
render = CLIPart(clip_model=clip_model, device=device, steps=steps, cutn=1, verbose=False, timeout=timeout, lr=0.12)
render_norm = CLIPart(clip_model=clip_model, device=device, steps=steps*stages, cutn=1, verbose=False, timeout=timeout)
prompt_o = sys.argv[1]
print(f"Baking {prompt_o}")
fix = transforms.Resize((224,224))
if ".jpg" not in prompt_o and ".png" not in prompt_o:
    prompt = clip_model.encode_text(clip.tokenize(prompt_o).to(device))
    prompt_type = 'txt'
else:
    base_image = fix(transforms.ToTensor()(Image.open(prompt_o).convert("RGB")).unsqueeze(0)).to(device)
    prompt = clip_model.encode_image(base_image)
    prompt_o = prompt_o.split("/")[-1].replace("_", " ").replace(".jpg", "").replace(".png", "")
    prompt_type = 'img'

z = torch.randn((1,3,resolution[0],resolution[1]), requires_grad=True, device=device)
_, norm = render_norm(prompt, z)
del render_norm

guides = [torch.randn((512), device=device) for i in range(stages)]
for i in range(len(guides)):
    guides[i].requires_grad = True
guide_opt = optim.AdamW(guides, 0.12)

steps = 100
for step in trange(steps):
    z = torch.randn((1,3,resolution[0],resolution[1]), requires_grad=True, device=device)
    loss = 0
    for i in range(len(guides)):
        z, pz = render(guides[i], z)

        # a loss to push it in the right semantic direction, but it's magnitude decays over time so
        # that in the early bit it doesnt have to focus on pushin it in the right direction
        # on it's own, but in the late bit it can focus on the things that are more important than
        # being globally semantically close
        loss+=(sph_dist(guides[i], prompt)/(len(guides)))*(1-step/steps)*(1-i/len(guides))
    z_emb = clip_model.encode_image(fix(z))
    loss += sph_dist(z_emb, prompt)
    if prompt_type=='img':
        print(base_image.shape)
        print(fix(z).shape)
        loss += (fix(z)-fix(base_image)).pow(2).mean()


    guide_opt.zero_grad()
    loss.backward(retain_graph=True) 
    guide_opt.step()

    # idk why the image isnt updating so lets save and reopen it ??? that's gotta be a big time waster
    # but honestly it's the one thing that's fixed it so far

    pz.save("output.png")
    pz = Image.open("output.png")
    wandb.log({"loss": loss.item(), 
               "output": [
                            wandb.Image(pz, caption=prompt_o+" (ex)"), 
                            wandb.Image(norm, caption=prompt_o+" (og)")
                          ]})

name=prompt_o.replace(" ", "_")
torch.save(torch.stack(guides), f"baked/{name}.pt")
