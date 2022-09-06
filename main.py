#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import mediapipe
import time
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot,QObject
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QMainWindow,QApplication
from gui_1 import Ui_MainWindow
import sys




class Detecter(QThread):
    
    signal=pyqtSignal(object)
    signal_2=pyqtSignal(bool)


    def run(self):

        # Opencv DNN
        net = cv2.dnn.readNet("yolo_weights/custom-yolov4-tiny-detector_best.weights", "yolo_weights/custom-yolov4-tiny-detector.cfg")
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(320, 320), scale=1/255)

        # Mediapipe hand pose
        mpHands = mediapipe.solutions.hands 
        self.hands = mpHands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

        # Mediapipe face mesh
        self.mp_face_mesh = mediapipe.solutions.face_mesh

        # Font for cv images
        self.font = cv2.FONT_HERSHEY_PLAIN
        # colors = np.random.uniform(0, 255, size=(100, 3))


        # Load class lists
        classes = []
        with open("yolo_weights/coco.names", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()  

                classes.append(class_name)
        
        # List which contains last five points of finger 
        liste=[1,2,3,4,5]
        # Control variable hand found or not
        self.check=False
        # Control variable hand in position or not
        sure=True

        # Initialize camera
        cap = cv2.VideoCapture(0)

        # Define Facemesh
        with self.mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

            while True:
            # Get frames
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                h, w = frame.shape[:2]
                try:
                    # Get nose and forehead landmarks
                    nose_y=results.multi_face_landmarks[0].landmark[4].y*h
                    forehead_y=results.multi_face_landmarks[0].landmark[9].y*h
                    # Check head noded or not
                    if nose_y-forehead_y<30:    
                        self.signal_2.emit(True)
                except:
                    pass
                
                # get last five finger landmark and draw 
                try:
                    finger_x,finger_y=self.eltespit(frame)
                    if len(liste)<6:
                        liste.pop(0)
                        liste.append([finger_x,finger_y])
                    points = np.array(liste)
                    points = points.reshape((-1, 1, 2))
                    for num,a in enumerate(liste):
                        x=liste[num][0]
                        y=liste[num][1]
                        cv2.circle(frame,(int(x),int(y)),5,(0,255,0),-1)
                except:
                    pass

                    
                
                
                # Object Detection
                (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
                for class_id, score, bbox in zip(class_ids, scores, bboxes):
                    (x, y, w, h) = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (200,0,50), 3)

                    class_name = classes[class_id]
                    
                    #Control hand on screen
                    if self.getColor(x, w, y, h, frame):

                        # ilk sinyal gelince süreyi alıyor.
                        if sure:
                            print("calistim")
                            time_function_done = time.time()
                            sure = False

                        # Sürenin üstüne 2 saniye geçene kadar elimi kontrol ediyor.
                        # Eğer elim telefona değiyorsa süreyi o ana eşitler.
                        if time.time() < (time_function_done + 2):
                            try:
                                if (x < finger_x < x + w) and (y < finger_y < y + h):
                    
                                    cv2.putText(frame, "elim sende", (x, y), self.font, 2, (255, 255, 255), 2)
                                    time_function_done = time.time()
                            except:
                                pass

                        # Eğer sinyalden sonra 5 saniye boyunca elim
                        # telefona değmediyse ekranda elini vermedin yazıyor.
                        # sure değişkeni True olarak değiştiriliyor.
                        # Sonraki 5 saniyeyi kontrol edecek.
                        else:
                            print(time_function_done, time.time())
                            cv2.putText(frame, "elini vermedin", (x, y), self.font, 2, (255, 255, 255), 2)
                            if (x < finger_x < x + w) and (y < finger_y < y + h):
                                sure = True
                    else:
                        print("maviler patladi")
                        sure = True
            
                self.signal.emit(frame)


          
      

    def getColor(self,x, w, y, h, frame):
        blueLower = (85, 100, 100)
        blueUpper = (135, 255, 255)
        kare = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)

        maske = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imwrite("maske1.jpg", maske)
        maske = cv2.dilate(maske, None, iterations=2)
        cv2.imwrite("maske_dilated.jpg", maske)
        maske = cv2.erode(maske, None, iterations=2)
        cv2.imwrite("maske2.jpg", maske)

        color_ratio = (maske.sum() // 255) / (kare.shape[0] * kare.shape[1])

        if color_ratio > 0.35:
            return True
        else:
            return False

    def eltespit(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hlms = self.hands.process(imgRGB) 

        h, w, channel = img.shape
        if hlms.multi_hand_landmarks:  
            positionX=hlms.multi_hand_landmarks[0].landmark[8].x*w
            positionY=hlms.multi_hand_landmarks[0].landmark[8].y*h
            
            self.check=True
            return positionX,positionY
        else:
            return None
            
            

class App(QMainWindow):
    def __init__(self):
        super(App,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.disply_width = 640
        self.display_height = 480
        self.count=0
        self.setWindowTitle("ArVis Proje")
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.ui.label.resize(self.disply_width, self.display_height)
        self.th=Detecter(self)
        self.th.signal.connect(self.getImage)
        self.th.signal_2.connect(self.exit)
        self.ui.pushButton.clicked.connect(self.savePhoto)
        self.ui.pushButton_2.clicked.connect(self.exit)

        self.th.start()
        self.show()

    @pyqtSlot(bool)
    def exit(self,check):
        self.th.quit()
        self.close()

    @pyqtSlot(object)
    def getImage(self, image):
        self.image=image
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.ui.label.setPixmap(QPixmap.fromImage(p))
    
    def savePhoto(self):
            self.count += 1
            filename = 'Snapshot_'+str(self.count)+'.png'
            cv2.imwrite(filename,self.image)
            self.ui.label_2.setPixmap(QPixmap('./Snapshot_'+str(self.count) + '.png'))
            print('Image saved as:',filename)
    

app=QApplication(sys.argv)
win=App()
win.show()
app.exec()