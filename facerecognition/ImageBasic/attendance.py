from importlib.resources import path
import cv2
import numpy as np
import face_recognition
from sympy import factor_terms
import os
from datetime import datetime

path = 'Images'
images= []
classNames= []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to write the name and time of person in csv file
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
      myDataList = f.readline()
      nameList = []
      for line in myDataList:
        entry = line.split(',')
        nameList.append(entry[0])

      if name not in nameList:
        now = datetime.now()
        tStr = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{tStr}')
      


def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListknown = findEncodings(images)
print('Encoding Completed')

cap=cv2.VideoCapture(0)

if (cap.isOpened() == False): 
  print("Unable to read camera feed")

while (True):
    success,img=cap.read()
    imgs= cv2.resize(img,(0,0),None,0.25,0.25)
    imgs= cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgs)
    encodesCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListknown,encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1 ,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2, y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255 ,255, 255),2)
            markAttendance(name)
        else:
              print('Unknown Person\n')
    else:
        break
    
cv2.imshow('face-recognition-cam',img)
cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
