import cv2
import time
import mediapipe as mp

video_name = "videoplayback.mp4"
cap = cv2.VideoCapture(video_name)

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.20)
mpDraw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    sonuc = faceDetection.process(imgRGB)
    
    if success == True:
        if sonuc.detections:
            for id, detection in enumerate(sonuc.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
                cv2.rectangle(frame, bbox, (0,0,255), 2)
        
        time.sleep(0.01)
        cv2.imshow("MediaPipe Face Detection - (Laura Branigan - Self Control)", frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release() 
cv2.destroyAllWindows()