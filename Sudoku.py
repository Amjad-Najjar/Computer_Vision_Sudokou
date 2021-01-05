import time

import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize

import sudukoSolver


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        if area > 50:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def Clean(img):
    flag = False
    edited = cv.adaptiveThreshold(img, 255, 1, 1, 11, 14)
    countores, _ = cv.findContours(edited, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    Board, _ = biggestContour(countores)
    if Board.size != 0:
        Order_Board = np.zeros((4, 1, 2), dtype=np.int32)
        pt = Board.reshape((4, 2))
        Max = pt.sum(1)
        Order_Board[0] = pt[np.argmin(Max)]
        Order_Board[3] = pt[np.argmax(Max)]
        Min = np.diff(pt, axis=1)
        Order_Board[1] = pt[np.argmin(Min)]
        Order_Board[2] = pt[np.argmax(Min)]
        pts = np.float32(Order_Board)
        pts1 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        Mat = cv.getPerspectiveTransform(pts, pts1)
        edited = cv.warpPerspective(edited, Mat, (300, 300))
        edited = cv.dilate(edited, (4, 4), iterations=2)
        cv.imshow("Extracted Image", edited)
        return True, edited

    return False, edited


def img2digitsimg(img):
    imges = []
    for i in range(6, 297, 33):
        for j in range(2, 297, 33):
            pts = [[j, i], [j + 33, i], [j, i + 33], [j + 33, i + 33]]
            pts1 = [[0, 0], [100, 0], [0, 100], [100, 100]]
            pts = np.float32(pts)
            pts1 = np.float32(pts1)
            m = cv.getPerspectiveTransform(pts, pts1)
            box = cv.warpPerspective(img, m, (100, 100))
            imges.append(box)
    return imges


def CheckEmptyBox(img, index):
    for i in range(30):
        for j in range(30):
            if img[34 + i][34 + j] == 255:
                return False
    return True


def most_frequent(List):
    return max(set(List), key=List.count)


def TemplateMatching(imgs):
    res = np.zeros(81)
    for i in range(1, 10):
        tmp = cv.imread("G:\\ITE-FIFTH\\Computer Vision\\Dataset\\SIFT1\\" + str(i) + ".jpg", 0)
        for index, img in enumerate(imgs):
            if not CheckEmptyBox(img, index):
                match = cv.matchTemplate(img, tmp, cv.TM_SQDIFF_NORMED)
                threshold = 0.8
                flag = False
                if np.amax(match) > threshold:
                    flag = True
                if flag:
                    res[index] = i
            else:
                res[index] = 0
    return res.reshape(9, 9)


def predictNumbersSift(imgs):
    orb = cv.FastFeatureDetector_create()
    orb1 = cv.SIFT_create()

    kp = np.ndarray(9).tolist()
    des = np.ndarray(9).tolist()
    for i in range(1, 10):
        tmp = cv.imread("G:\\ITE-FIFTH\\Computer Vision\\Dataset\\SIFT2\\" + str(i) + ".jpg", 0)
        kp[i - 1] = orb.detect(tmp, None)
        des[i - 1] = orb1.compute(tmp, kp[i - 1])[1]
    res = np.zeros(81)
    for index, img in enumerate(imgs):
        if not CheckEmptyBox(img, index):
            kp1 = orb.detect(img, None)
            des1 = orb1.compute(img, kp1)[1]
            pres = []
            for i, n in enumerate(des):
                index_params = dict(algorithm=0, trees=5)
                search_params = dict(checks=100)
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, n, k=2)
                for a, b in matches:
                    if a.distance < 0.7 * b.distance:
                        pres.append((i + 1))
            res[index] = most_frequent(pres)
        else:
            res[index] = 0
    res = np.array(res).reshape(9, 9)
    return res


def getmax(a):
    max = 0
    tmp = object()
    for i in a:
        if i.response > max:
            max = i.response
            tmp = i
    return tmp


def predictNumbersKnnModel(imgs):
    fil = open(r'G:\ITE-FIFTH\Computer Vision\knnpickle_file1.model', 'rb')
    models = pickle.load(fil)
    res = np.zeros(81)
    for index, img in enumerate(imgs):
        if not CheckEmptyBox(img, index):
            ac = cv.resize(img, (128, 128), interpolation=cv.INTER_LANCZOS4)
            tt = np.ones_like(ac)
            ac = cv.bitwise_not(ac, tt)
            tmp = models.predict([ac.flatten()])
            res[index] = tmp
            print(tmp)

    return res.resize(9, 9)


def imgCuts(img):
    countores, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2:]
    t = []
    imges = []
    for c in countores:
        area = cv.contourArea(c)
        if area > 700 and area < 1000:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                t.append(approx)
                pts = reorder(approx)
                pts1 = [[0, 0], [100, 0], [0, 100], [100, 100]]
                pts = np.float32(pts)
                pts1 = np.float32(pts1)
                m = cv.getPerspectiveTransform(pts, pts1)
                imges.append([cv.warpPerspective(img, m, (100, 100))])
    print("Hi ", len(t))
    d = cv.drawContours(img, t, -1, color=(255, 0, 0), thickness=1)
    cv.imshow("d", d)
    return imges


def onMouse(event, x, y, a, b):
    if event == cv.EVENT_LBUTTONDOWN:
        print("x=", x, " y=", y)

def Overlay(grid,img,unsolved):

    imgs=np.zeros((len(img),len(img[1]),3))
    Empty_Grid=cv.imread(r"G:\ITE-FIFTH\Computer Vision\Pictures\grid2.jpg")
    Empty_Grid=cv.resize(Empty_Grid,(300,300),interpolation=cv.INTER_LANCZOS4)
    # kernal=np.ones_like(img)
    Empty_Grid=cv.bitwise_not(Empty_Grid)
    # img=img-Empty_Grid
    imgs[:,:,0]=np.zeros_like(img)
    imgs[:,:,1]=np.zeros_like(img)
    imgs[:,:,2]=np.zeros_like(img)

    img_prev=imgs
    img_next=np.zeros_like(imgs)
    for i in range(9):
        for j in range(9):
            if grid[j][i]!=0:
                img_next=cv.putText(img_prev,str(int(grid[j][i])), (int((i*33)+12), int((j*33)+32)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,15,155), 1, cv.LINE_AA)
                img_prev=img_next
    for i in range(9):
        for j in range(9):
            if unsolved[j][i]!=0:
                img_next=cv.putText(img_prev,str(int(unsolved[j][i])), (int((i*33)+12), int((j*33)+32)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                img_prev=img_next
    img_next=img_next+Empty_Grid
    return img_next


def video():
    PicPath = "G:\\ITE-FIFTH\\Computer Vision\\Video\\"
    vid = cv.VideoCapture(1)
    cnt = 0
    Good_Catch = []
    font = cv.FONT_HERSHEY_SIMPLEX
    new_frame_time = 0
    prev_frame_time = 0
    flag = False
    while (True):
        ret, frame = vid.read()
        cnt += 1
        new_frame_time = time.time()
        tt = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if cnt % 10:
            flag, z = Clean(tt)
        if flag and not np.all(z == 0) and not np.all(z == 255) and not np.count_nonzero(z == 255) > 10000:
            Good_Catch.append(z)
        elif len(Good_Catch) != 0:
            Good_Catch = []
        if len(Good_Catch) == 3:
            zs = img2digitsimg(Good_Catch[1])
            grid = predictNumbersSift(zs)
            unsolved=grid.copy()
            sudukoSolver.solve(unsolved)
            cv.imshow("Solved",Overlay(abs(grid-unsolved),Good_Catch[1]))
            # print(abs(grid-unsolved))
            Good_Catch = []
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv.putText(frame, fps, (7, 50), font, 2, (100, 255, 255), 2, cv.LINE_AA)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()


video()
# debug=cv.imread(r"G:\ITE-FIFTH\Computer Vision\Video\5.jpg",0)
# r=debug
# cv.imshow("debug",debug)
# debug=img2digitsimg(debug)
# debug=predictNumbersSift(debug)
# s=debug.copy()
# sudukoSolver.solve(s)
# print(debug)
# print(s)
# cv.imshow("solved",Overlay(abs(s-debug),r,debug))
cv.waitKey(0)
