
import cv2
import os
import random
import numpy as np
from manhattan_distance_transform import manhattan_distance_transform
from face_features import FaceFeature
from glob import glob


def select_faces(faces, mask):
    faces_selected = []
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        mask_face = np.zeros_like(mask)
        mask_face[y1:y2, x1:x2] = 1
        intersection = cv2.bitwise_and(mask_face, mask)
        face_iou = cv2.countNonZero(intersection) / cv2.countNonZero(mask_face)
        
        if face_iou >= 0.5:
            faces_selected.append(face)
    return faces_selected

def create_face_mask(landmark, mask):
    mask_face = np.zeros_like(mask, dtype=np.float32)
    _, w = mask.shape[:2]
    points = [(landmark.part(n).x, landmark.part(n).y) for n in range(4, 13)] # jawline
    points += [(w-1, points[-1][1]), (w-1, 0), (0, 0), (0, points[0][1])]     
    points = np.array(points, dtype=np.int32)
    
    mask_face = cv2.fillPoly(mask_face, [points], 1)    
    mask_face = mask_face * mask
    
    return mask_face

def mask_find_bboxs(mask):
    xs, ys = np.where(mask > 0)
    return np.min(xs), np.max(xs), np.min(ys), np.max(ys)

frames = sorted(glob('D:\\workspace\\VisualTimeKeeper-shuowan-froze_effect\\VisualTimeKeeper-shuowan-froze_effect\\frames\\*.jpeg'))
img = cv2.imread(frames[0], 1)
print(img.shape)
h, w, _ = img.shape
frame_num = len(frames)
FPS = 30
T = len(frames) / 30
fragments = 8
first_num = 10
blend_pre_frame = 1.0 / 30
ice_size = int(h / fragments)

output_dir = '.\\output_dir_video'
os.makedirs(output_dir, exist_ok=True)

fill_mask = np.zeros((h, w))
ices_3_names = sorted(glob('.\\VisualTimeKeeper-shuowan-froze_effect\\VisualTimeKeeper-shuowan-froze_effect\\textures\\kisspng*_3.png'))
ices_1_names = sorted(glob('.\\VisualTimeKeeper-shuowan-froze_effect\\VisualTimeKeeper-shuowan-froze_effect\\textures\\kisspng*_1.png'))
ices_3, ices_1 = [], []
for name in ices_3_names:
    ice_img = cv2.imread(name, -1)
    # print(ice_img.shape)
    ice_mask = ice_img[..., 3]
    bbox = mask_find_bboxs(ice_mask)
    ice_img = ice_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    hm, wm, _ = ice_img.shape
    new_h = ice_size * 2
    new_w = int((new_h / hm) * wm)
    ice_img = ice_img.astype(np.float32)
    ice_img = cv2.resize(ice_img,  (new_w, new_h), cv2.INTER_LINEAR)
    ices_3.append(ice_img)

for name in ices_1_names:
    ice_img = cv2.imread(name, -1)
    # print(ice_img.shape)
    ice_mask = ice_img[..., 3]
    bbox = mask_find_bboxs(ice_mask)
    ice_img = ice_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    hm, wm, _ = ice_img.shape
    new_h = ice_size
    new_w = int((ice_size / hm) * wm)
    ice_img = ice_img.astype(np.float32)
    ice_img = cv2.resize(ice_img,  (new_w, new_h), cv2.INTER_CUBIC)
    ices_1.append(ice_img)

generating_cache = []
generating_cache_weight = []
generating_cache_box = []
shift = h / frame_num
for t in range(1, frame_num):

    img = cv2.imread(frames[t], 1)
    # mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    # mask = mask.astype(float) / 255
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1

    # face_feature = FaceFeature()
    # faces = face_feature.detect_face(img)
    # faces = select_faces(faces, mask)
    # landmarks = face_feature.detect_landmark(faces, img)

    # assert len(faces) == 1, 'Only one face should be detected'
    # face = faces[0]
    # landmark = landmarks[0]

    # mask_face = create_face_mask(landmark, mask)
    # mask_body = mask - mask_face

    # dist_map_face = manhattan_distance_transform(mask_face)
    # dist_map = manhattan_distance_transform(mask)

    # d_max_face = np.max(dist_map_face)
    # d_max_body = np.max(dist_map * mask_body)
    # factor = d_max_body / d_max_face
    # dist_map_face = dist_map_face * factor
    # dist_map = np.maximum(dist_map, dist_map_face)

    center_h = int(h - t * shift)
    img_display = img.copy().astype(np.float32)
    human_mask = np.zeros((h, w), dtype=np.float32) #TBD

    for i in range(first_num):
        idx = np.random.choice(2)
        scale = np.random.uniform(0.6, 1.4)
        center_w = np.random.randint(0, w)
        if idx == 0:
            ice = random.sample(ices_3, 1)[0].copy()
            flipflag = np.random.choice(2)
            if flipflag:
                ice = cv2.flip(ice, 1)
        else:
            ice = random.sample(ices_1, 1)[0].copy()
            flipflag = np.random.choice(2)
            #TBD rotate
            if flipflag:
                ice = cv2.flip(ice, 1)
        # ice = cv2.resize(ice, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        hi, wi, _ = ice.shape
        h_b, h_e, w_b, w_e = max(0, center_h - hi // 2), min(h, center_h + hi // 2), max(0, center_w - wi // 2), min(w, center_w + wi // 2)
        ch_b, ch_e, cw_b, cw_e = max(0, 0 - (center_h - hi // 2)), max(0, center_h + hi // 2 - h), max(0, 0 - (center_w - wi // 2)), max(0, center_w + wi // 2 - w)
        crop_fill = fill_mask[h_b:h_e, w_b:w_e]
        hf, wf = crop_fill.shape
        crop_ice = ice[ch_b:(hi-ch_e), cw_b:(wi-cw_e), ...]
        crop_ice = crop_ice[:hf, :wf]
        crop_mask = crop_ice[..., 3] / 255.
        xs, ws = np.where(crop_mask > 0)
        ice_area = len(xs)
        xs, ws = np.where(crop_mask * crop_fill > 0)
        union_area = len(xs)
        if union_area / ice_area > 0.6:
            continue
        weight = blend_pre_frame
        fill_mask[h_b:h_e, w_b:w_e][crop_mask > 0.04] = 255
        crop_mask = np.expand_dims(crop_mask, axis=-1)
        img_display[h_b:h_e, w_b:w_e] = img_display[h_b:h_e, w_b:w_e] * (1.0 - crop_mask) + img_display[h_b:h_e, w_b:w_e] * crop_mask * (1.0 - weight) + crop_ice[..., 0:3] * crop_mask * weight
        generating_cache.append(crop_ice)
        generating_cache_weight.append(weight)
        generating_cache_box.append((h_b, h_e, w_b, w_e))


    if t == 100:
        pass

    for i, ice in enumerate(generating_cache):
        weight = generating_cache_weight[i] + blend_pre_frame
        weight = 0.6 if weight > 0.6 else weight
        generating_cache_weight[i] = weight
        h_b, h_e, w_b, w_e = generating_cache_box[i]
        crop_mask = ice[..., 3] / 255.
        crop_mask = np.expand_dims(crop_mask, axis=-1)
        crop_human_mask = np.expand_dims(human_mask[h_b:h_e, w_b:w_e], axis=-1)
        img_display[h_b:h_e, w_b:w_e] = img_display[h_b:h_e, w_b:w_e] * (1.0 - crop_mask) + (img_display[h_b:h_e, w_b:w_e] * (1.0 - weight) + ice[..., 0:3] * weight) * (1 - crop_human_mask) * crop_mask \
                                        + img[h_b:h_e, w_b:w_e].astype(np.float32) * crop_human_mask * crop_mask
        

    img_display = np.clip(img_display, 0, 255.0)
    # mask_display_curr = dist_map <= s * t
    # mask_display_curr = mask_display_curr * mask  # only display the fg region
    
    # mask_delta = mask_display_curr - mask_display_prev
    
    # intensity_map[mask_display_curr != 0] += s_intensity
    # intensity_map = np.clip(intensity_map, 0, 1)
    # intensity_map = np.minimum(intensity_map, texture_alpha)

    # for c in range(3):
    #     img_display[:,:,c] = img_display[:,:,c] * (1 - intensity_map) + texture[:,:,c] * intensity_map
    
    # mask_display_prev = mask_display_curr
    cv2.imwrite('{}/frame_{:04d}.jpg'.format(output_dir, t), img_display.astype(np.uint8))
    
    
