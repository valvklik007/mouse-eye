import cv2, pytesseract, random
import cv2 as cv
from PIL import ImageGrab as ig
import numpy as np

cord = ''
#680,100,1250,300
#400,900,1600,1080
cordinats = 680,250,1250,450
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
    ret, thresh = cv2.threshold(gray, 150, 200, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = ""
    output1 = image.copy()
    x1 = []
    y1 = []
    w1 = []
    h1 = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        output = (x)
        if hierarchy[0][idx][3] == 0:
            x1.append(x)
            y1.append(y)
            w1.append(w)
            h1.append(h)
            cv2.rectangle(output1, (x, y), (x + w, y + h), (0, 255, 0), 0)
    cv2.imshow("1233", output1)
    if cord == np.array(output):
        print("уже были кардинаты", random.random())
    else:
        cord = output
        formula = ""
        formula1 = []
        for m in range(len(x1)):
            if h1[m] > 10:
                x2 = x1[m]-10
                y2 = y1[m]-10
                fy = y2 + h1[m]+10
                fx1 = x2 + w1[m]+10
                fy1 = fy - h1[m]-20
                formula = [(x2, y2),(x2, fy),(fx1, fy), (fx1, y2)]
                formula1.append(formula)
        mask = np.zeros(image.shape, dtype=np.uint8)
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret1, thresh1 = cv2.threshold(gray1, 200, 0, cv2.THRESH_BINARY)
        cv2.imwrite("2(1).png", thresh1)
        img1 = cv2.imread('2(1).png')
        #=================
        output1 = image.copy()
        for m in range(len(formula1)):
            mask = np.zeros(image.shape, dtype=np.uint8)
            roi_corners = np.array([formula1[m]],dtype=np.int32)
            channel_count = image.shape[2]
            ignore_mask_color = (255,)*channel_count
            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            masked_image = cv2.bitwise_and(image, mask)
            img2 = masked_image
            # Я хочу разместить логотип в левом верхнем углу, поэтому я создаю ROI
            rows,cols,channels = img2.shape
            roi = img1[0:rows, 0:cols ]
            # Теперь создайте маску логотипа и создайте ее обратную маску.
            img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, -1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask_inv = cv2.bitwise_not(mask)
            # Теперь затемните область логотипа в ROI
            img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            # Взять только область логотипа из логотипа изображения.
            img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
            # # Put logo in ROI and modify the main image
            dst = cv2.add(img1_bg,img2_fg)
            output1 = dst
            img1[0:rows, 0:cols ] = dst
        cv2.imwrite("test1.png", img1)
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
        ret, threshold_image = cv2.threshold(gray_image, 229, 255, 0)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(threshold_image, lang='rus+eng', config='--oem 1, --psm 6')
        print(text)
        cv2.imshow("123", threshold_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
