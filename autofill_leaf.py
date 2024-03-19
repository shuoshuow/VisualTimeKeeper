import cv2
import os
import random
import numpy as np
from glob import glob
import imgaug.augmenters as iaa


def get_bbox_from_mask(mask):
    # Get bounding box coordinates from a mask
    non_zero = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(non_zero)
    return x, y, w, h

def load_binary_mask(mask_path, w, h):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (w, h))
    mask = mask.astype(float) / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return mask


def load_textures(texture_files, texture_height):
    # Load textures
    textures = []
    for texture_file in texture_files:
        texture_img = cv2.imread(texture_file, -1)
        
        # crop the non-empty part of the texture
        texture_mask = texture_img[..., 3]    
        bbox = get_bbox_from_mask(texture_mask)        
        texture_img = texture_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], ...]
                
        # reshape the texture to the target height
        hm, wm, _ = texture_img.shape
        new_h = texture_height
        new_w = int((new_h / hm) * wm)        
        texture_img = cv2.resize(texture_img.astype(np.float32),  (new_w, new_h), cv2.INTER_CUBIC)
               
        # pad the texture to avoid cutting off through rotation augmentation
        texture_img = np.pad(texture_img, ((new_h//2, new_h//2), (new_w//2, new_w//2), (0, 0)), 'constant', constant_values=0)

        textures.append(texture_img)
        
    return textures


# Read input image and mask
img = cv2.imread('./data/testframe.jpg', 1)
img = cv2.resize(img, (1280, 720))
h, w = img.shape[:2]
mask_fg = load_binary_mask('./data/testmask.png', w, h)
fg_height = get_bbox_from_mask(mask_fg)[1]

# Constants
FPS = 25
T = 5
fragments = 18
texture_num_per_row = 10
blend_pre_frame = 1.0 / 30
texture_height = int((h - fg_height) / fragments) * 2
frame_num = FPS * T
# shift = int((h - fg_height) / frame_num)
shift = int(h / frame_num)

output_dir = './output/leaf_pile_2'
os.makedirs(output_dir, exist_ok=True)

texture_files = sorted(glob('./data/leaf/*.png'))
textures = load_textures(texture_files, texture_height)

fill_mask = np.zeros_like(mask_fg)
generating_mask = np.zeros_like(mask_fg)
generating_cache = []
generating_cache_mask = []
generating_cache_weight = []
generating_cache_box = []

aug_geometric = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Fliplr(1.0)),
    iaa.Affine(
        scale=(0.7, 1.4),
        rotate=(-20, 20),
        mode='constant',
        cval=0)
    ])

# Generate frames
for t in range(1, frame_num):    
    # img_display = img.copy().astype(np.float32)
    img_effect = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    alpha_effect = np.zeros(img.shape[:2], dtype=np.float32)
    
    py = int(h - t * shift)

    i = 0
    trials = 0
    fill_mask_frame = np.zeros_like(mask_fg)
    while(i < texture_num_per_row and trials < 50):
        trials += 1
        
        # Check if the center is in the foreground
        px = np.random.randint(0, w)                
        if mask_fg[py, px] == 0:
            continue
        
        # randomly select a texture and augment it
        texture_img = random.choice(textures).copy()        
        texture_img = aug_geometric.augment_image(texture_img)
        texture_mask = texture_img[..., 3]
        texture_mask = texture_mask.astype(float) / 255
        texture_mask[texture_mask < 0.5] = 0
        texture_mask[texture_mask >= 0.5] = 1

        cx_t, cy_t = texture_img.shape[1] // 2, texture_img.shape[0] // 2
        
        # left-top and right-bottom coordinates of the image
        x0 = max(0, px - cx_t)
        y0 = max(0, py - cy_t)
        x1 = min(w, px + cx_t)
        y1 = min(h, py + cy_t)
        
        # left-top and right-bottom coordinates of the texture
        x0_t = max(0, cx_t - px)
        y0_t = max(0, cy_t - py)
        x1_t = x1 - x0
        y1_t = y1 - y0
        
        crop_fill = fill_mask_frame[y0:y1, x0:x1]
        crop_texture = texture_img[y0_t:y1_t, x0_t:x1_t, 0:3]
        crop_mask = texture_mask[y0_t:y1_t, x0_t:x1_t]
        
        if t == 70:
            t = t
            
        # skip if overlap with existing textures
        intersection = np.logical_and(crop_fill, crop_mask)
        if np.sum(intersection) / np.sum(crop_mask) > 0.6:
            continue           
        
        # update mask        
        fill_mask_frame[y0:y1, x0:x1] = np.logical_or(fill_mask_frame[y0:y1, x0:x1], crop_mask)
        fill_mask[y0:y1, x0:x1] = np.logical_or(fill_mask[y0:y1, x0:x1], crop_mask)
        
        generating_cache.append(crop_texture)
        generating_cache_mask.append(crop_mask)
        generating_cache_weight.append(blend_pre_frame)
        generating_cache_box.append((x0, y0, x1, y1)) #h_b, h_e, w_b, w_e
        
        i += 1
        
        # # for debug
        # weight = blend_pre_frame
        # crop_mask = crop_mask[..., np.newaxis]        
        # img_display[y0:y1, x0:x1] = img_display[y0:y1, x0:x1] * (1.0 - crop_mask) \
        #         + img_display[y0:y1, x0:x1] * crop_mask * (1.0 - weight) \
        #         + crop_texture * crop_mask * weight
                
        

    for i, (weight_prev, crop_texture, crop_mask) in enumerate(zip(generating_cache_weight, generating_cache, generating_cache_mask)):
        weight_curr = weight_prev + blend_pre_frame
        weight_curr = min(weight_curr, 0.9)
        generating_cache_weight[i] = weight_curr
        
        x0, y0, x1, y1= generating_cache_box[i]
        
        alpha_effect[y0:y1, x0:x1] = alpha_effect[y0:y1, x0:x1] * (1.0 - crop_mask) \
                + alpha_effect[y0:y1, x0:x1] * crop_mask * (1.0 - weight_curr) \
                + crop_mask * weight_curr
                
        crop_mask = crop_mask[..., np.newaxis]
        img_effect[y0:y1, x0:x1] = img_effect[y0:y1, x0:x1] * (1.0 - crop_mask) \
                + crop_texture * crop_mask 
        
        alpha_effect = np.clip(alpha_effect, 0, 1.0)
        img_effect = np.clip(img_effect, 0, 255.0)
    
    img_final = np.concatenate([img_effect, alpha_effect[..., np.newaxis] * 255], axis=2)
    cv2.imwrite('{}/frame_{:04d}.png'.format(output_dir, t), img_final.astype(np.uint8))
        
        
