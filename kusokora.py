import cv2
import numpy as np


path = "lena.png"
f_path = "face.png"
img = cv2.imread(path)
f_img = cv2.imread(f_path)
f_img = np.array(f_img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f_img.shape)
face_path = "haarcascade_frontalface_alt.xml"

face_cascade = cv2.CascadeClassifier(face_path)
faces = face_cascade.detectMultiScale(img_gray)
print(faces)

for (x, y, w, h) in faces:
    f_img = cv2.resize(f_img,(w,h),interpolation=cv2.INTER_LINEAR)  
    dist = img[y:y+h,x:x+w]
    cv2.imwrite("kuso.png",dist)
    for i,n in enumerate(range(y, h+y)):
        for j,k in enumerate(range(x, w+x)):
            if (i-h/2)**2+(j-w/2)**2 < (w/2)**2:
                img[n][k]=f_img[i][j]
img = cv2.blur(img,(5,5))
img = cv2.medianBlur(img,5)
cv2.imshow("tsts",dist)
cv2.imshow("tst",img)
cv2.waitKey(0)
