import glob
from PIL import Image

# filepaths
fp_in = "out/vae/*.jpg"
fp_out = "vae.gif"

for f in glob.glob(fp_in):
    print(f)

img, *imgs = [Image.open(f) for f in glob.glob(fp_in)]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)
