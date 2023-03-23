# conda env create -f environment.yaml
# conda acitvate ldm
# pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
# pip install git+https://github.com/arogozhnikov/einops.git
# mkdir -p models/ldm/cin256-v2/
# wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt 

python3 sample.py --seed 2023 --n-samples-per-class 3
zip -r LDM-4-human-experiments res/

# python sample.py --seed 0 --n-samples-per-class 10 --start_img_id 0 
# zip -r LDM-4-0-9999 res/

# python sample.py --seed 1 --n-samples-per-class 10 --start_img_id 10000 
# zip -r LDM-4-10000-19999 res/

# python sample.py --seed 2 --n-samples-per-class 10 --start_img_id 20000  
# zip -r LDM-4-20000-29999 res/

# python sample.py --seed 3 --n-samples-per-class 10 --start_img_id 30000 
# zip -r LDM-4-30000-39999 res/

# python sample.py --seed 4 --n-samples-per-class 10 --start_img_id 40000 
# zip -r LDM-4-40000-49999 res/

# python sample.py --seed 5 --n-samples-per-class 10 --start_img_id 50000 
# zip -r LDM-4-50000-59999 res/

# python sample.py --seed 6 --n-samples-per-class 10 --start_img_id 60000 
# zip -r LDM-4-60000-69999 res/

# python sample.py --seed 7 --n-samples-per-class 10 --start_img_id 70000 
# zip -r LDM-4-70000-79999 res/

# python sample.py --seed 8 --n-samples-per-class 10 --start_img_id 80000 
# zip -r LDM-4-80000-89999 res/

# python sample.py --seed 9 --n-samples-per-class 10 --start_img_id 90000 
# zip -r LDM-4-90000-99999 res/