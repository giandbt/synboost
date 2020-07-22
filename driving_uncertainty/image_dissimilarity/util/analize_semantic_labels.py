import numpy as np
from PIL import Image


if __name__ == '__main__':
    semantic_img = '/media/giandbt/Samsung_T5/master_thesis/data/wild_dash/wd_val_01/wd_val_01/cn0000_100000_labelIds.png'
    img = np.array(Image.open(semantic_img))
    mask = np.where(img==5, 255, 0)
    print(mask)
    mask_img = Image.fromarray(mask.astype(np.uint8)).convert('RGB').save('/home/giandbt/Desktop/test.png')
    print(np.unique(img, return_counts=True))
