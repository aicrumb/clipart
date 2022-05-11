"""
from testing, this is the code that produces the best 32x32 images in my opinion.
"""
from tqdm import tqdm
import torch as t
import clip 
import PIL
import gc
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision import transforms
from .augment import MakeCutouts
from .metrics import sph_dist, tv_loss
import time
class CLIPart(t.nn.Module):
    def __init__(self, clip_model, steps=450, lr=0.12, cutn=8, penalize_graffiti=True, device='cuda', verbose=True, timeout=None):
        super().__init__()
        self.clip_model = clip_model
        self.lr = lr
        self.cutn = cutn
        self.penalize_graffiti = penalize_graffiti
        self.device=device
        self.verbose=verbose
        self.steps = steps
        self.timeout=timeout
    def forward(self, prompt, z):
        z.requires_grad_(True)
        # assert z.shape[-1]==z.shape[-2], "z should be square"

        o=t.optim.Adam((z,),self.lr)
        make_cutouts = MakeCutouts(self.clip_model.visual.input_resolution, 8)

        if self.penalize_graffiti:
            te = prompt
            te=te-self.clip_model.encode_text(clip.tokenize("text, graffitti, words").to(self.device))*0.5
            te=te+self.clip_model.encode_text(clip.tokenize("#pixelart simplistic 8-bit sprite art").to(self.device))*0.5
        else:
            te = prompt
        te = te.to(self.device)
        if self.verbose:
            range_n = tqdm(range(self.steps))
        else:
            range_n = range(self.steps)
        for i in range_n:
            l = sph_dist(self.clip_model.encode_image(make_cutouts(z)), te.unsqueeze(0))
            l = l +  (z - z.clamp(-1, 1)).pow(2).mean()/2
            l = l + (tv_loss(z).mean()*1.4) 
            l = l + F.mse_loss(z.std(),t.ones(1,device=self.device)*.4)*.13 
            o.zero_grad()
            l.backward(retain_graph=True)
            o.step()
            if self.timeout:
                time.sleep(self.timeout)
        image = t.clamp(z/2+0.5,0,1).squeeze(0)
        image = t.round(image*16)/16
        image = (image*1.1-0.1).clamp(0,1)
        pil = transforms.ToPILImage()(image).resize((512,512), PIL.Image.NEAREST)
        if self.device=='cuda':
            t.cuda.empty_cache()
        gc.collect()
        return z, pil 