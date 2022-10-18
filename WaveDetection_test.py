import cv2
import numpy as np

img = cv2.imread("project\msteam_wave.png")

img = cv2.resize(img,(300,300))
img1 = cv2.resize(img,(300,300))
cv2.imshow('Original',img1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur (gray, (3, 3),0)

_,thresh = cv2.threshold(gray_blur, 170, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh',thresh)

result = img.copy()

contours, hierachy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
image = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow('CONTOUR',image)

y_avg= []
for cnt in contours:
    areas = cv2.contourArea(cnt)
    if areas<500 :
        continue
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    x1 = (box[0][0] + box[3][0]) // 2
    y1 = (box[0][1] + box[3][1]) // 2
    x2 = (box[1][0] + box[2][0]) // 2
    y2 = (box[1][1] + box[2][1]) // 2

    cv2.line(result, (x1,y1), (x2,y2), (255,0,0), 2)

    cv2.circle(result,(int((x1+x2)/2),int((y1+y2)/2)),1,(0,0,255),1)
   # print(int((y1+y2)/2))
    y_avg.append(int((y1+y2)/2))
print('---------------------')
print('Average Y :',y_avg)

y_differcence = []
for n in range(len(y_avg)):
    # print(n)
    # print(y1_avg[n-1]-y1_avg[n]) # distance between dots
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

cv2.putText(result,'Ramda(MM) :',(100,290),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)
cv2.putText(result,str(c),(220,290),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)

cv2.imshow("RESULT", result) 
cv2.waitKey(0)
cv2.destroyAllWindows()
