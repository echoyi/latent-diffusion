import argparse
import os
import torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler



def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def custom_to_pil(sample):
    x = sample.clone().detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    model = get_model()
    sampler = DDIMSampler(model)



    num_classes = 1000
    # classes = [25, 187, 448, 992]   # define classes to be sampled here
    class_labels_all = [i for i in range(num_classes)]
    n_samples_per_class = args.n_samples_per_class
    img_id = args.start_img_id

    ddim_steps = args.ddim_steps
    ddim_eta = 0.0
    scale = args.scale  # for unconditional guidance


    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            
            for class_label in class_labels_all:
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                for sample in x_samples_ddim:
                    path = f'res/{class_label}/{img_id}.png'
                    if not os.path.exists(f'res/{class_label}'): os.makedirs(f'res/{class_label}')
                    img = custom_to_pil(sample)
                    img.save(path)
                    img_id += 1
                    

    # # display as grid
    # grid = torch.stack(all_samples, 0)
    # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    # grid = make_grid(grid, nrow=n_samples_per_class)

    # # to image
    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    # img = Image.fromarray(grid.astype(np.uint8))
    # img.save('img.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--ddim-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples-per-class", type=int, default=3, help='number of samples per class, default 3')
    parser.add_argument("--start_img_id", type=int, default=0, help='img id of the first generated image, default 0')
    args = parser.parse_args()
    main(args)