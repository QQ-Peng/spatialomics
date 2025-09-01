# Author: Qianqian Peng
# Date: 2025-09-01
# %%
from PIL import Image
import argparse
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    image = Image.open(args.input)
    image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    # %%
    # image.show()
    # %%
    # image_flip.show()
    # %%
    image_flip.save(args.output)