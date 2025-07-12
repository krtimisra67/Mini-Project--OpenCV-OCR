import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Image Read
image_path="C:\\Users\\hp\\Desktop\\Elite Work 2025\\OPEN CV + OCR\\Sample Images\\Datacluster Indian Signboard (2).jpg"
img=cv2.imread(image_path)


#Instances of image
reader=easyocr.Reader(['en'],gpu=True)


#Detect text on the image
text_=reader.readtext(img)


threshold=0.25
#Draw bounding Box and Text
for t_,t in enumerate(text_):
    print(t)
    
    bbox,text,score=t
    
    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0) ,5) #(0,255,0) is color which is green for the box and 5 is the thickness for the box only
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65,(255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()


