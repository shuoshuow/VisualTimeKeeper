
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
    points = [(landmark.part(n).x, landmark.part(n).y) for n in range(17, 27)] # eyebrows
    points += [(landmark.part(n).x, landmark.part(n).y) for n in range(16, 0, -1)] # jawline
    points += [points[0]]     
    points = np.array(points, dtype=np.int32)
    
    mask_face = cv2.fillPoly(mask_face, [points], 1)    
    mask_face = mask_face * mask
    point4 = [(landmark.part(0).x, landmark.part(0).y), 
                (landmark.part(16).x, landmark.part(16).y),
                (landmark.part(11).x, landmark.part(11).y), 
                (landmark.part(5).x, landmark.part(5).y)]
    point4 = np.array(point4, dtype=np.float32)
    return point4, mask_face

def mask_find_bboxs(mask):
    xs, ys = np.where(mask > 0)
    return np.min(xs), np.max(xs), np.min(ys), np.max(ys)

SCALE=2
frames = sorted(glob('D:\\workspace\\VisualTimeKeeper\\frames\\*.jpeg'))
masks = sorted(glob('D:\\workspace\\VisualTimeKeeper\\segmentation\\*.png'))
img = cv2.imread(frames[0], 1)

print(img.shape)
h, w, _ = img.shape
h *= SCALE
w *= SCALE
frame_num = len(frames)
FPS = 30
T = len(frames) / 30
fragments = 8
first_num = 10
blend_pre_frame = 1.0 / 30
ice_size = int(h / fragments)

output_dir = '.\\output_dir_video'
os.makedirs(output_dir, exist_ok=True)

face_feature = FaceFeature()
fill_mask = np.zeros((h, w))
ices_3_names = sorted(glob('D:\\workspace\\VisualTimeKeeper\\textures\\kisspng*_3.png'))
ices_1_names = sorted(glob('D:\\workspace\\VisualTimeKeeper\\textures\\kisspng*_1.png'))
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
frozen_mask = np.zeros((h, w), dtype=np.uint8)
# frozen_color_adjust = np.array([[[1.3, 1.1, 0.9]]])

body_freeze_frame = 146
body_freeze_h = 348
body_freeze_img = cv2.imread(frames[body_freeze_frame], 1).astype(np.float32)
body_freeze_mask = cv2.imread(masks[body_freeze_frame], 0).astype(np.float32) / 255.

kernel = np.ones((3, 3), np.uint8)
# body_freeze_mask_d = cv2.dilate(body_freeze_mask, kernel, iterations=5)
# body_freeze_mask_d = np.expand_dims(body_freeze_mask_d, axis=-1)

head_freeze_frame = 400
head_freeze_h = 44
head_freeze_img = cv2.imread(frames[head_freeze_frame], 1).astype(np.float32)
head_freeze_mask = cv2.imread(masks[head_freeze_frame], 0).astype(np.float32)
head_freeze_mask[head_freeze_mask < 0.5] = 0
head_freeze_mask[head_freeze_mask >= 0.5] = 1
faces = face_feature.detect_face(head_freeze_img.astype(np.uint8))
faces = select_faces(faces, head_freeze_mask)
head_landmarks = face_feature.detect_landmark(faces, img.astype(np.uint8))

assert len(faces) == 1, 'Only one face should be detected'
face = faces[0]
head_landmark = head_landmarks[0]
headpoint, _ = create_face_mask(head_landmark, head_freeze_mask)
retval = cv2.getPerspectiveTransform(headpoint, headpoint)

# tmp_dir = '.\\output_dir_withoutice'
# os.makedirs(tmp_dir, exist_ok=True)

for t in range(1, frame_num):
    print(f'frame {t}')
    img = cv2.imread(frames[t], 1).astype(np.float32)
    mask = cv2.imread(masks[t], 0).astype(np.float32)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    center_h = int(h - t * shift)
    frozen_mask[center_h] = 1
    human_mask = mask

    if t == 415:
        pass

    if t > body_freeze_frame: # Freeze body
        human_mask[body_freeze_h:] = body_freeze_mask[body_freeze_h:]
        img[body_freeze_h:] = body_freeze_img[body_freeze_h:]
    
    if t >360 and t < 370:
        human_mask[:, 450:] = body_freeze_mask[:, 450:]
        img[:, 450:] = body_freeze_img[:, 450:]

    if t > head_freeze_frame:
        faces = face_feature.detect_face(img.astype(np.uint8))
        faces = select_faces(faces, mask)
        landmarks = face_feature.detect_landmark(faces, img.astype(np.uint8))

        assert len(faces) == 1, 'Only one face should be detected'
        face = faces[0]
        landmark = landmarks[0]

        point, mask_face = create_face_mask(landmark, mask)
        
        retval_cur = cv2.getPerspectiveTransform(point, headpoint)
        retval = retval_cur * 0.2 + retval * 0.8
        mask_face_warp = cv2.warpPerspective(mask_face, retval, (w//SCALE, h//SCALE), flags=cv2.INTER_CUBIC).astype(np.float32)
        mask_face_warp = cv2.erode(mask_face_warp, kernel, iterations=3)
        mask_face_warp = cv2.GaussianBlur(mask_face_warp, (7, 7), 21)
        img_warp = cv2.warpPerspective(img, retval, (w//SCALE, h//SCALE), flags=cv2.INTER_CUBIC).astype(np.float32)

        mask_face_warp = np.expand_dims(mask_face_warp[head_freeze_h:body_freeze_h], -1)
        
        img[head_freeze_h:body_freeze_h] = img_warp[head_freeze_h:body_freeze_h] * mask_face_warp + head_freeze_img[head_freeze_h:body_freeze_h] * (1.0 - mask_face_warp)
        human_mask[head_freeze_h:body_freeze_h] = head_freeze_mask[head_freeze_h:body_freeze_h]

    img = cv2.resize(img, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    human_mask = cv2.resize(human_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img_display = img.copy()
    human_mask *= 0.75

    for i in range(first_num):
        idx = np.random.choice(2)
        scale = np.random.uniform(0.8, 1.2)
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
        ice = cv2.resize(ice, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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


    # cv2.imwrite('{}/frame_{:04d}.jpg'.format(tmp_dir, t), img_display.astype(np.uint8))

    for i, ice in enumerate(generating_cache):
        weight = generating_cache_weight[i] + blend_pre_frame
        weight = 0.6 if weight > 0.6 else weight
        generating_cache_weight[i] = weight
        h_b, h_e, w_b, w_e = generating_cache_box[i]
        crop_mask = ice[..., 3] / 255.
        crop_mask = np.expand_dims(crop_mask, axis=-1)
        # crop_human_mask = np.expand_dims(human_mask[h_b:h_e, w_b:w_e], axis=-1)
        # img_display[h_b:h_e, w_b:w_e] = img_display[h_b:h_e, w_b:w_e] * (1.0 - crop_mask) + (img_display[h_b:h_e, w_b:w_e] * (1.0 - weight) + ice[..., 0:3] * weight) * (1 - crop_human_mask) * crop_mask \
        #                                 + img[h_b:h_e, w_b:w_e] * crop_human_mask * crop_mask
        img_display[h_b:h_e, w_b:w_e] = img_display[h_b:h_e, w_b:w_e] * (1.0 - crop_mask) + (img_display[h_b:h_e, w_b:w_e] * (1.0 - weight) + ice[..., 0:3] * weight) * crop_mask
    
    human_mask = np.expand_dims(human_mask, axis=-1)
    # img[center_h:] *= frozen_color_adjust
    img_display = img_display * (1 - human_mask) + img * human_mask
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
    
    
