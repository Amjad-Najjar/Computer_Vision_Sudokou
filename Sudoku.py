import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize


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
    flag=False
    # img =cv.GaussianBlur(img,(5,5),1)
    edited=cv.adaptiveThreshold(img, 255, 1, 1, 11, 14)
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
    if Board.size !=0 :
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
        return True ,edited

    # edited=cv.morphologyEx(edited,cv.MORPH_OPEN,(9,9),iterations=1)
    # edited=cv.dilate(edited,(7,7),iterations=1)
    #tr=np.ones_like(edited)
    #edited=cv.bitwise_not(edited,tr)
    return False , edited
def img2digitsimg(img):
    imges=[]
    cnt=0
    for i in range(6,297,33):
        for j in range(2,297,33):
            pts=[[j,i],[j+33,i],[j,i+33],[j+33,i+33]]
            pts1=[[0,0],[100,0],[0,100],[100,100]]
            pts =np.float32(pts)
            pts1=np.float32(pts1)
            m=cv.getPerspectiveTransform(pts,pts1)
            box=cv.warpPerspective(img,m,(100,100))
            imges.append(box)
            cnt+=1
            # if not CheckEmptyBox(box,1):
            #     cv.imwrite(r"G:\ITE-FIFTH\Computer Vision\Dataset\SIFT\\a"+str(cnt)+".jpg",box)
    return imges
def skel(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        close = cv.morphologyEx(img, cv.MORPH_CLOSE, element)

        # Step 3: Substract open from the original image
        open=cv.subtract(open, close)

        temp = cv.subtract(img, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv.erode(img, element)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv.countNonZero(img) == 0:
            break
    return skel
def CheckEmptyBox(img,index):
    for i in range(30):
        for j in range(30):
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
def TemplateMatching(imgs):
    res=np.zeros(81)
    for i in range(1, 10):
        tmp = cv.imread("G:\\ITE-FIFTH\\Computer Vision\\Dataset\\SIFT1\\" + str(i) + ".jpg", 0)
        for index,img in enumerate(imgs):
            if not CheckEmptyBox(img,index):
                    match=cv.matchTemplate(img,tmp,cv.TM_SQDIFF_NORMED)
                    threshold = 0.8
                    flag = False
                    if np.amax(match) > threshold:
                        flag = True
                    if flag:
                        res[index]=i
            else:
                res[index]=0
    return res.reshape(9,9)

def predictNumbersSift(imgs):
    orb = cv.FastFeatureDetector_create()
    orb1 = cv.SIFT_create()

    kp = np.ndarray(9).tolist()
    des = np.ndarray(9).tolist()
    for i in range(1, 10):
        tmp = cv.imread("G:\\ITE-FIFTH\\Computer Vision\\Dataset\\SIFT2\\" + str(i) + ".jpg", 0)
        tmp=skel(tmp)
        # plt.imshow(c)
        # cv.imshow(str(i),c)
        #tmp=cv.resize(tmp,(100,100),interpolation=cv.INTER_AREA)
        # tmp=cv.erode(tmp,(7,7),iterations=2)
        kp[i-1] = orb.detect(tmp, None)
        des[i-1]=orb1.compute(tmp,kp[i-1])[1]
    res=np.zeros(81)
    for index,img in enumerate(imgs):
         img=skel(img)
         # cv.imshow(str(index),t)
         if not CheckEmptyBox(img,index):
            kp1=orb.detect(img,None)
            des1=orb1.compute(img,kp1)[1]
            pres=[]
            for i,n in enumerate(des):
                index_params = dict(algorithm = 0,trees=5)
                search_params = dict(checks=100)
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1,n, k=2)
                # Need to draw only good matches, so create a mask
                matchesMask = [0 for i in range(len(matches))]

                # ratio test as per Lowe's paper
                for g,  (m, n) in enumerate(matches):
                    if m.distance < 0.75 * n.distance:
                        matchesMask[g] = 1

                for a,b in matches:
                    if a.distance <0.7*b.distance :
                        pres.append((i+1))

            print(pres, ' Hello',most_frequent(pres))

            res[index]=most_frequent(pres)
         else:
             res[index]=0
    res=np.array(res).reshape(9,9)
    return res
def getmax(a):
    max=0
    tmp=object()
    for i in a:
        if i.response > max:
            max=i.response
            tmp=i
    return tmp
def predictNumbersKnnModel(imgs):
    fil = open(r'G:\ITE-FIFTH\Computer Vision\knnpickle_file1.model', 'rb')
    models = pickle.load(fil)
    res=np.zeros(81)
    for index,img in enumerate(imgs):
        if not CheckEmptyBox(img,index):
            ac = cv.resize(img, (128, 128), interpolation=cv.INTER_LANCZOS4)
            tt = np.ones_like(ac)
            ac = cv.bitwise_not(ac, tt)
            #cv.imshow(str(index),ac)
            #ac=cv.erode(ac,(13,13),iterations=6)
            tmp=models.predict([ac.flatten()])
            res[index]=tmp
            print(tmp)


    return res.resize(9,9)


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

def video():
    PicPath = "G:\\ITE-FIFTH\\Computer Vision\\Video\\"
    vid = cv.VideoCapture(0)
    cnt = 0
    while (True):
        ret, frame = vid.read()
        c = vid.get(cv.CAP_PROP_FPS)
        tt = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flag, z = Clean(tt)
        z = np.float32(z)
        if cnt<0:
            cnt=0
        if flag and not np.all(z == 0) and not np.all(z == 255) and not np.count_nonzero(z == 255) > 10000:
            cnt += 1
            cv.imwrite(PicPath + str(cnt) + ".jpg", z)
        else:
            cnt-=1
        if cnt ==8:
            break
            #c = img2digitsimg(z)
            #a = predictNumbersSift(c)
        #     tmp=cv.imread(PicPath+"15.jpg",0)

            # print(a)
        #     break
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    z=cv.imread(PicPath+"5.jpg",0)
    # z=cv.dilate(z,(5,5),iterations=1)
    z=cv.dilate(z, (4, 4), iterations=2)
    z=img2digitsimg(z)
    # grid=TemplateMatching(z)
    # grid=predictNumbersSift(z)
    grid=predictNumbersSift(z)
    print(grid)

    vid.release()
    # # Destroy all the windows
    cv.destroyAllWindows()


video()
# debug=cv.imread(r"G:\ITE-FIFTH\Computer Vision\Video\13.jpg",0)
#
# cv.imshow("debug",debug)
# # debug=cv.erode(debug,(5,5))
# debug=cv.dilate(debug,(4,4),iterations=2)
# # debug=cv.morphologyEx(debug,cv.MORPH_CLOSE,(5,5),iterations=1)
#
#
# cv.imshow("debug edited",debug)
# debug=img2digitsimg(debug)
# print(predictNumbersSift(debug))
cv.waitKey(0)