import numpy as np
import cv2 as cv
import pickle

import math



def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        if area > 50:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4 :
                biggest = approx
                max_area = area
    return biggest,max_area
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew
def Clean(img):

    img =cv.GaussianBlur(img,(5,5),1)
    edited=cv.adaptiveThreshold(img, 255, 1, 1, 11, 8)
    countores,_=cv.findContours(edited,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE )
    #Board =np.array([])
    #MaxArea=0
    # for c in countores:
    #         area=cv.contourArea(c)
    #         if area >50:
    #             arc_len= cv.arcLength(c,True)
    #             approx=cv.approxPolyDP(c,0.02*arc_len,Tr
    #             ue)
    #             if  area>MaxArea  and len(approx)==4 :
    #                 Board = approx
    #                 MaxArea=area
    Board,_=biggestContour(countores)
    Order_Board=np.zeros((4,1,2),dtype=np.int32)
    pt=Board.reshape((4,2))
    Max=pt.sum(1)
    Order_Board[0]=pt[np.argmin(Max)]
    Order_Board[3]=pt[np.argmax(Max)]
    Min=np.diff(pt,axis=1)
    Order_Board[1]=pt[np.argmin(Min)]
    Order_Board[2]=pt[np.argmax(Min)]
    print(Order_Board)
    pts=np.float32(Order_Board)
    pts1=np.float32([[0,0],[300,0],[0,300],[300,300]])
    Mat=cv.getPerspectiveTransform(pts,pts1)
    edited=cv.warpPerspective(edited,Mat,(300,300))
    cv.imshow("wr",edited)

    edited=cv.morphologyEx(edited,cv.MORPH_OPEN,(9,9),iterations=1)
    edited=cv.erode(edited,(5,5),iterations=1)
    #tr=np.ones_like(edited)
    #edited=cv.bitwise_not(edited,tr)
    return edited
def img2digitsimg(img):
    imges=[]
    for i in range(6,297,33):
        for j in range(2,297,33):
            pts=[[j,i],[j+33,i],[j,i+33],[j+33,i+33]]
            pts1=[[0,0],[100,0],[0,100],[100,100]]
            pts =np.float32(pts)
            pts1=np.float32(pts1)
            m=cv.getPerspectiveTransform(pts,pts1)
            imges.append(cv.warpPerspective(img,m,(100,100)))
    return imges
def CheckEmptyBox(img,index):
    for i in range(40):
        for j in range(40):
            if img[34+i][34+j]==255:
                return False
    return True
    # if img[50][50]==255 or img[51][51]==255 or img[49][49]==255 or:
    #     return False
    # else:
    #     return True
    # contour,_=cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2:]
    # print(i,"   ",len(contour))
    # Flag=True
    # if len(contour) >2:
    #     for c in contour:
    #         area=cv.contourArea(c)
    #         if area >500:
    #             Flag =True
    #         else:
    #             Flag=False
    #     return Flag
    # else:
    #     return False
def most_frequent(List):
    return max(set(List), key = List.count)
def predictNumbers(imgs):
    orb = cv.SIFT_create()
    bf = cv.BFMatcher()
    kp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    des = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, 10):
        tmp = cv.imread("G:\\ITE-FIFTH\\Computer Vision\\Dataset\\SIFT\\" + str(i) + ".jpg", 0)
        tmp=cv.resize(tmp,(100,100),interpolation=cv.INTER_CUBIC)
        kp[i - 1], des[i - 1] = orb.detectAndCompute(tmp, None)
        res=np.zeros(81)
    for index,img in enumerate(imgs):
        if not CheckEmptyBox(img,index):
            # img=cv.dilate(img,(3,3),iterations=1)
            cv.imshow(str(index),img)
            kp1,des1=orb.detectAndCompute(img,None)
            pres=[]
            for i,n in enumerate(des):
                matches=bf.knnMatch(des1,n,k=2)
                for a,b in matches:
                    if a.distance <0.75*b.distance:
                        pres.append(i+1)
            print(index ,"  res ", most_frequent(pres))
            res[index]=most_frequent(pres)
        else:
            res[index]=0
    res=np.array(res).reshape(9,9)
    return res
     # for index,img in enumerate(imgs):
     #    if not CheckEmptyBox(img,index):
     #        kp1,des1=orb.det
    # fil = open(r'G:\ITE-FIFTH\Computer Vision\knnpickle_file1.model', 'rb')
    # models = pickle.load(fil)
    # res=np.zeros_like(imgs)
    # print(res.shape)
    #  for index,img in enumerate(imgs):
    #     if not CheckEmptyBox(img,index):
    #         ac = cv.resize(img, (128, 128), interpolation=cv.INTER_LANCZOS4)
    #         tt = np.ones_like(ac)
    #         ac = cv.bitwise_not(ac, tt)
    #         cv.imshow(str(index),ac)
    #         #ac=cv.erode(ac,(13,13),iterations=6)
    #         tmp=models.predict([ac.flatten()])
    #         print(index, "   ",tmp[0] )
    #         res[index]=tmp[0]

    # return res


def imgCuts(img):
    countores,_=cv.findContours(img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:]
    t =[]
    imges=[]
    for c in countores :
            area=cv.contourArea(c)
            if area >700 and area <1000:
                peri = cv.arcLength(c,True)
                approx = cv.approxPolyDP(c, 0.02*peri, True)
                if len(approx)==4 :
                    t.append(approx)
                    pts=reorder(approx)
                    pts1=[[0,0],[100,0],[0,100],[100,100]]
                    pts = np.float32(pts)
                    pts1 = np.float32(pts1)
                    m = cv.getPerspectiveTransform(pts, pts1)
                    imges.append([cv.warpPerspective(img, m, (100, 100))])
    print("Hi ", len(t))
    d=cv.drawContours(img,t,-1,color=(255,0,0),thickness=1)
    cv.imshow("d",d)
    return imges


def onMouse(event,x,y,a,b):
    if event==cv.EVENT_LBUTTONDOWN:
        print("x=",x," y=",y)

Original_Sudoku = cv.imread(r"G:\ITE-FIFTH\Computer Vision\Pictures\Sudoku1.jpg",0)
Empty_Grid=cv.imread(r"G:\ITE-FIFTH\Computer Vision\Pictures\grid1.jpg",0)
Empty_Grid=cv.resize(Empty_Grid,(300,300),interpolation=cv.INTER_LANCZOS4)
_,Empty_Grid=cv.threshold(Empty_Grid,200,255,cv.THRESH_BINARY)
Empty_Grid=cv.erode(Empty_Grid,(9,9),iterations=2)
Empty_Grid=cv.morphologyEx(Empty_Grid,cv.MORPH_CLOSE,(5,5))
cv.imshow("r",Empty_Grid)
mask=np.zeros_like(Empty_Grid)
Empty_Grid=cv.bitwise_not(Empty_Grid,mask)
cv.imshow("test",Original_Sudoku)
cv.setMouseCallback("test",onMouse)
ims=Clean(Original_Sudoku)
ims=cv.morphologyEx(ims,cv.MORPH_CLOSE,(7,7),iterations=2)

cv.imshow("edited",ims)
c=cv.bitwise_or(ims,Empty_Grid)
cv.imshow("c",c)
#z=imgCuts(c)
z=img2digitsimg(Clean(Original_Sudoku))
grid=predictNumbers(z)
print(grid)
cnt=0
# for i in range(81):
#     if not CheckEmptyBox(z[i],i):
#         cv.imshow(str(i),z[i])
#         cnt=cnt+1
print(cnt)
#cv.imshow("z",z[79][0])
tt=z[3]
tt=cv.dilate(tt,(5,5),iterations=5)
cv.imshow("q",tt)
cv.imshow("av",z[3])

cv.setMouseCallback("q",onMouse)
#print(grid)
# if(CheckEmptyBox(z[1],1)):
#     print("Yess")
#a=z[79]-z[0]
#cv.imwrite(r"G:\ITE-FIFTH\Computer Vision\Projects\Computer_Vision_Sudokou\9.jpg",z[3])

cv.waitKey(0)