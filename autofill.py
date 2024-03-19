import cv2
import numpy as np
from manhattan_distance_transform import manhattan_distance_transform
from face_features import FaceFeature


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

img = cv2.imread('D:\\Work_Teams\\engine\\demo\\data\\testframe.jpg', 1)
mask = cv2.imread('D:\\Work_Teams\\engine\\demo\\data\\testmask.png', 0)
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
mask = mask.astype(float) / 255
mask[mask < 0.5] = 0
mask[mask >= 0.5] = 1

face_feature = FaceFeature()
faces = face_feature.detect_face(img)
faces = select_faces(faces, mask)
landmarks = face_feature.detect_landmark(faces, img)

assert len(faces) == 1, 'Only one face should be detected'
face = faces[0]
landmark = landmarks[0]

mask_face = create_face_mask(landmark, mask)
mask_body = mask - mask_face

dist_map_face = manhattan_distance_transform(mask_face)
dist_map = manhattan_distance_transform(mask)

d_max_face = np.max(dist_map_face)
d_max_body = np.max(dist_map * mask_body)
factor = d_max_body / d_max_face
dist_map_face = dist_map_face * factor
dist_map = np.maximum(dist_map, dist_map_face)

FPS = 30
T = 5
s = max(1.0, np.max(dist_map) / (FPS * T)) # growth speed
s_intensity = 1.0 / FPS # intensity speed

texture = cv2.imread('frozen_texture.png', cv2.IMREAD_UNCHANGED)
texture_alpha = texture[:,:,3] / 255

intensity_map = np.zeros_like(dist_map)
mask_display_prev = np.zeros_like(mask)

for t in range(FPS * T):
    img_display = img.copy()
    
    mask_display_curr = dist_map <= s * t
    mask_display_curr = mask_display_curr * mask  # only display the fg region
    
    mask_delta = mask_display_curr - mask_display_prev
    
    intensity_map[mask_display_curr != 0] += s_intensity
    intensity_map = np.clip(intensity_map, 0, 1)
    intensity_map = np.minimum(intensity_map, texture_alpha)

    for c in range(3):
        img_display[:,:,c] = img_display[:,:,c] * (1 - intensity_map) + texture[:,:,c] * intensity_map
    
    mask_display_prev = mask_display_curr
    cv2.imwrite('output/frame_{:04d}.jpg'.format(t), img_display)
    
    
