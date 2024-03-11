import numpy as np

def manhattan_distance_transform(mask):
    rows, cols = mask.shape
    distance = np.zeros_like(mask)
    
    # first pass
    for i in range(1, rows):
        if mask[i, 0] == 1:
            distance[i, 0] = distance[i-1, 0] + 1

    for j in range(1, cols):
        if mask[0, j] == 1:
            distance[0, j] = distance[0, j-1] + 1
            
    for i in range(1, rows):
        for j in range(1, cols):
            if mask[i, j] == 1:
                distance[i, j] = min(distance[i-1, j] + 1, distance[i, j-1] + 1)

    # set the last row of fg to be 1
    for j in range(0, cols):
        if mask[rows-1, j] == 1:
            distance[rows-1, j] = 1
            
    # second pass
    for i in range(rows-2, -1, -1):
        if mask[i, cols-1] == 1:
            distance[i, cols-1] = min(distance[i+1, cols-1] + 1, distance[i, cols-1])

    for j in range(cols-2, -1, -1):
        if mask[rows-1, j] == 1:
            distance[rows-1, j] = min(distance[rows-1, j+1] + 1, distance[rows-1, j])
            
    for i in range(rows-2, -1, -1):
        for j in range(cols-2, -1, -1):
            if mask[i, j] == 1:
                distance[i, j] = min(distance[i+1, j] + 1, distance[i, j+1] + 1, distance[i, j])
    
    return distance

if __name__ == '__main__':
    import cv2
    mask = cv2.imread('D:\\Work_Teams\\engine\\demo\\data\\testmask.png', 0)
    mask = mask.astype(np.float32) / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    dist_map = manhattan_distance_transform(mask)
    cv2.imwrite('DistanceMap.jpg', dist_map/np.max(dist_map)*255)