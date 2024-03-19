import dlib
import cv2


class FaceFeature():
    def __init__(self):
        p = "D:\\workspace\\VisualTimeKeeper\\shape_predictor_68_face_landmarks.dat"
        self._face_detector = dlib.get_frontal_face_detector()
        self._landmark_detector = dlib.shape_predictor(p)
        
    def detect_face(self, img):
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(gray)
        return faces
        
    def detect_landmark(self, faces, img):
        if len(faces) == 0:
            return []
        
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = [self._landmark_detector(gray, face) for face in faces]
        return landmarks
    
    def draw_face_bboxes(self, img, faces):
        vis_img = img.copy()
        for face in faces:
            x1=face.left()
            y1=face.top()
            x2=face.right()
            y2=face.bottom()
            cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0),3)
        return vis_img
    
    def draw_landmarks(self, img, landmarks):
        vis_img = img.copy()
        for landmark in landmarks:
            for n in range(0, 68):
                x=landmark.part(n).x
                y=landmark.part(n).y
                cv2.putText(vis_img, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return vis_img
    
    
if __name__ == '__main__':
    face_feature = FaceFeature()
    img = cv2.imread('D:\\Work_Teams\\engine\\demo\\data\\testframe.jpg', 1)
    faces = face_feature.detect_face(img)
    landmarks = face_feature.detect_landmark(faces, img)
    vis_img = face_feature.draw_landmarks(img, landmarks)
    cv2.imwrite('landmarks.jpg', vis_img)