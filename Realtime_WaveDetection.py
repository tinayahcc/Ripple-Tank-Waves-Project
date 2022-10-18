# space to take screenshots / esc to escape

import cv2

cam = cv2.VideoCapture(2) #Webcam

cv2.namedWindow("test")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

from tkinter import image_names
import cv2
import numpy as np
from PIL import Image
import requests
import io

img = cv2.imread("opencv_frame_0.jpg") 

img = cv2.resize(img,(300,300))
img1 = cv2.resize(img,(300,300))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur (gray, (3, 3),0)

_,thresh = cv2.threshold(gray_blur, 175, 255, cv2.THRESH_BINARY)

kernel = np.ones ((3,3),np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations=5)

result = closing.copy()

contours, hierachy = cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cv2.imshow('contour',result)

result_ = img.copy()

line = cv2.imread("white.jpg")
line = cv2.resize(line,(300,300))

y_avg= []
for cnt in contours:
    areas = cv2.contourArea(cnt)
    if areas<500 :
        continue

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    x1 = (box[0][0] + box[1][0]) // 2
    y1 = (box[0][1] + box[1][1]) // 2
    x2 = (box[3][0] + box[2][0]) // 2
    y2 = (box[3][1] + box[2][1]) // 2

    cv2.line(result_, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.line(line, (x1,y1), (x2,y2), (255,0,0), 2)

    cv2.circle(result_,(int((x1+x2)/2),int((y1+y2)/2)),1,(0,0,255),1)

    y_avg.append(int((y1+y2)/2))
print('---------------------')
print('Average Y :',y_avg)

y_differcence = []
for n in range(len(y_avg)):

    y_differcence.append(y_avg[n-1]-y_avg[n])
y_differcence.pop(0)
distance_ = y_differcence
print('Y value difference :',distance_)
a = sum(distance_) / len(distance_)
b = round(a,2)
c = round(b*0.0264583333*10,2) # MM

print('Ramda : ',b,'pixel')
print('Ramda : ',c,'MM')
print('---------------------')

final = cv2.imread("Final3.png")
cv2.putText(final,str(c)+' Millimeter',(227,445),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,0),1)
result2 = cv2.resize(result_,(140,140))
x_offset = 230
y_offset = 283
x_end = x_offset + result2.shape[1]
y_end = y_offset + result2.shape[0]
final[y_offset:y_end,x_offset:x_end] = result2

img2 = cv2.resize(img1,(140,140))
x_offset2 = 40
y_offset2 = 283
x_end2 = x_offset2 + img2.shape[1]
y_end2 = y_offset2 + img2.shape[0]
final[y_offset2:y_end2,x_offset2:x_end2] = img2
line2 = cv2.resize(line,(140,140))
x_offset3 = 135
y_offset3 = 455
x_end3 = x_offset3 + line2.shape[1]
y_end3 = y_offset3 + line2.shape[0]
final[y_offset3:y_end3,x_offset3:x_end3] = line2

token = "9V04nddlKiqwTAWZpVuBiARZuMsY3Mgzyes3Ntw9cUG" # Give me a WAVE!
url = 'https://notify-api.line.me/api/notify'
HEADERS = {'Authorization': 'Bearer ' + token}

cv2.imshow("FINAL",final)
cv2.imwrite('capture.png',final)
            
ramda = 'Ramda',c,'mm'
ramda = str(ramda)
ramda = ramda.replace(',',"")
ramda = ramda.replace('(',"")
ramda = ramda.replace(')',"")

msg = ramda

cv2.imwrite('finalline.png',line)
cv2.imwrite('finalimg1.png',img1)
cv2.imwrite('finalresult.png',result_)

stickerPackageId = 446
stickerId = 2010

imgd = Image.open('capture.png')
imgd.load()
myimg = np.array(imgd)
f = io.BytesIO()
f = io.BytesIO()

Image.fromarray(myimg).save(f, 'png')
data = f.getvalue()

response = requests.post(url,headers=HEADERS,params={"message": msg},
                        files={"imageFile" : data})
print(response)

imga = Image.open('finalimg1.png')
imga.load()
myimga = np.array(imga)
fa = io.BytesIO()
fa = io.BytesIO()

msg = 'original wave image'

Image.fromarray(myimga).save(fa, 'png')
dataa = fa.getvalue()

responsea = requests.post(url,headers=HEADERS,params={"message": msg},
                        files={"imageFile" : dataa})
print(responsea)

imgb = Image.open('finalresult.png')
imgb.load()
myimgb = np.array(imgb)
fb = io.BytesIO()
fb = io.BytesIO()

msg = 'final result'

Image.fromarray(myimgb).save(fb, 'png')
datab = fb.getvalue()

responseb = requests.post(url,headers=HEADERS,params={"message": msg},
                        files={"imageFile" : datab})
print(responseb)

imgc = Image.open('finalline.png')
imgc.load()
myimgc = np.array(imgc)
fc = io.BytesIO()
fc = io.BytesIO()

msg = 'wave line only'

Image.fromarray(myimgc).save(fc, 'png')
datac = fc.getvalue()

responsec = requests.post(url,headers=HEADERS,params={"message": msg},
                        files={"imageFile" : datac})
print(responsec)

cv2.waitKey(0)
cv2.destroyAllWindows()