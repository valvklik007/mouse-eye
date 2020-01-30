import cv2, pytesseract, random
import cv2 as cv
from PIL import ImageGrab as ig
import numpy as np


cord = ''
#680,100,1250,300
#400,900,1600,1080
cordinats = 65,100,1200,300
while(True):
    screen = ig.grab(bbox=(cordinats))
    #cv2.imshow("test", np.array(screen))
    #last_time = time.time()
    img1 = cv2.flip(np.array(screen),2)
    img2 = cv2.flip(img1,2)
    image = img2
    cv2.imwrite("test12.png", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #229, 200, 1
    ret, thresh = cv2.threshold(gray, 150, 200, 0, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = ""
    cv2.imshow("1233", img_erode)
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        output = (x)
    if cord == np.array(output):
        print("уже были кардинаты", random.random())
    else:
        cord = output
        a1 = cv.drawContours(image, contours, -1, (0, 0, 0), -1, cv.LINE_AA, hierarchy, 3)
        gray2 = cv2.cvtColor(a1, cv2.COLOR_BGR2RGB)
        cv2.imwrite("2(1).jpg", a1)

        image = cv2.imread('2(1).jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 200, 1, cv2.THRESH_BINARY)
        #4,11
        img_erode = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        a2 = cv.drawContours(image, contours, -1, (0, 0, 0), -10, cv.LINE_AA, hierarchy, 10)
        cv2.imwrite("123.png", a2)

        image = cv2.imread('123.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 180, 200, 1, cv2.THRESH_BINARY)
        img_erode = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        a2 = cv.drawContours(image, contours, -1, (0, 0, 0), -10, cv.LINE_AA, hierarchy, 10)
        cv2.imwrite("123(1).png", a2)

        img = cv2.imread("123(1).png")
        def foto(img, lower=np.array([190,190,190]),upper=np.array([255,255,255])):
            return cv2.inRange(img, lower, upper)
        a = foto(img)
        cv2.imwrite("test1.png", a)

    	#====Черный фон удаляем и альфа канал
        src = cv2.imread("test1.png", 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        b, g, r = cv2.split(src)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        cv2.imwrite("test3.png", dst)

        scale_percent = 120 # Процент от изначального размера
        width = int(dst.shape[1] * scale_percent / 100)
        height = int(dst.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
        gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        ret, threshold_image = cv2.threshold(gray_image, 50, 255, 1)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(threshold_image, lang='rus', config='--oem 1, --psm 6')
        print(text)
        cv2.imshow("123", threshold_image)
        #============
        # img = cv2.imread("123.png")
        # #180,180,180
        # def foto(img, lower=np.array([180,180,180]),upper=np.array([255,255,255])):
    	#     return cv2.inRange(img, lower, upper)
        # a = foto(img)
        # cv2.imwrite("test1.png", a)
    	# #====Черный фон удаляем
        # src = cv2.imread("test1.png", 1)
        # tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # b, g, r = cv2.split(src)
        # rgba = [b,g,r, alpha]
        # dst = cv2.merge(rgba,4)
    	# #====Сервый цывет=======
        # scale_percent = 100 # Процент от изначального размера
        # width = int(dst.shape[1] * scale_percent / 100)
        # height = int(dst.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # resized = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
        # gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2XYZ)
        # #115, 146
        # ret, threshold_image = cv2.threshold(gray_image, 1, 255, 1)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        # text = pytesseract.image_to_string(threshold_image, lang='rus', config='--oem 1, --psm 6')
        # print(text)
        # cv2.imwrite("test1.png", threshold_image)
        #=============
        #cv2.imshow("123", threshold_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
