import cv2
import numpy as np
import ScannerSupport
import imutils
image_=cv2.imread("images/receipt.jpg")
(h,w,c)=image_.shape
r=h/500
dim=(int(w/r),int(h/r))
image=cv2.resize(image_,dim)

# image=imutils.resize(image_,height=500)

cv2.imshow("Image",image)
cv2.waitKey(0)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(gray,75,200)

cv2.imshow("Edges",edged)
cv2.waitKey(0)

contour,hierarchy=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
output = image.copy()
contour=sorted(contour,key=cv2.contourArea,reverse=True)[:5]

for c in contour:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if(len(approx)==4):
		screenCnt=approx
		break

cv2.drawContours(output,  [screenCnt], -1, (0, 250, 0), 2)
cv2.imshow("Outline",output)
cv2.waitKey(0)

warped=ScannerSupport.four_point_transform(image_,screenCnt.reshape(4,2)*r)
warped= cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped=cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,199,5)		#199, 5
# warped=cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)		#199, 5
warped=imutils.resize(warped,height=500)
cv2.imshow("Scanned",warped)
cv2.waitKey(0)