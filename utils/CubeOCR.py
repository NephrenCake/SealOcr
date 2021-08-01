import base64
import cv2
import numpy as np
import requests
import re
import json


def filter_non_red(img):
    img_ = np.copy(img)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # opencv 的H范围是0~180，红色的H范围大概是(0~8)∪(160,180)
    # V是亮度，太小了是黑色，太大了是白色1880
    h, w, _ = img_HSV.shape
    for i in range(h):
        for j in range(w):
            h = img_HSV[i, j, 0]
            not_RED_h = not (0 <= h <= 10 or 156 <= h <= 180)
            s = img_HSV[i, j, 1]
            not_RED_s = s < 20
            v = img_HSV[i, j, 2]
            is_BLACK = v <= 128
            if not_RED_h or not_RED_s or is_BLACK:
                img_[i, j, :] = 255
    return img_


def Process(img):
    h, w, _ = img.shape
    if h >= 500:
        h = h // 2
        w = w // 2
        img = cv2.resize(img, (w, h))

    oimg = img
    img = filter_non_red(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # canny = cv2.Canny(gray, 50, 150, apertureSize=3)
    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 3)
    binary = 255 - binary

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((10, 10), np.uint8)
    dialated = cv2.dilate(opening, kernel, iterations=5)

    gray = dialated
    # find contours of all the components and holes
    gray_temp = gray.copy()  # copy the gray image because function
    # findContours will change the imput image into another
    contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # find the max area of all the contours and fill it with 0
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    for i in range(len(contours)):
        if i != max_idx:
            cv2.fillConvexPoly(gray, contours[i], 0)

    # show image without max connect components

    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1

    rotrect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)

    o_box = np.float32([box[0], box[1], box[2], box[3]])
    t_box = np.float32([[250, 250], [0, 250], [0, 0], [250, 0]])
    M = cv2.getPerspectiveTransform(o_box, t_box)
    dst = cv2.warpPerspective(oimg, M, (250, 250))
    #cv2.imshow("Output Image", dst)
    #cv2.waitKey()
    return dst


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tobytes()).decode('utf8')


def ocr_request(dest):
    # 发送HTTP请求
    data = {'images': [cv2_to_base64(dest)]}
    headers = {"Content-type": "application/json"}
    url = "http://47.101.136.120:8866/predict/ocr_system"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    res = (r.json())["results"][0]  # [{}*n]
    results = []
    for r in res:
        results.append({
            "angle": r['angle'],
            "angle_conf": r['angle_conf'],
            "confidence": r["confidence"],
            "text": r["text"],
            "text_region": r["text_region"]
        })
    #print(results)
    return results


def get_deg(results):
    flag = 0
    for r in results:
        angles = r['angle']
        if int(angles) == 180:
            flag = flag - 1
        else:
            flag = flag + 1
    if flag >= 0:
        return 0
    else:
        return 180


def clear_deg(results, deg):
    nresult = []
    for r in results:
        angles = r["angle"]
        if int(angles) == deg:
            nresult.append(r)
    return nresult


def get_num_length(results):
    pattern = re.compile(r'\d+')
    max_length = 0
    for r in results:
        numtext = pattern.findall(r["text"])
        if len(numtext) > 0:
            length = len(numtext[0])
            if length > max_length:
                max_length = length
    return max_length


def GetNumArea(results, deg):
    max_length = 0
    for r in results:
        pattern_num = re.compile(r'\d+')
        numtext = pattern_num.findall(r["text"])
        if len(numtext) > 0:
            length = len(numtext[0])
            if length > max_length:
                max_length = length
                maxn = r
#-------------------------------

    try :
        r = maxn
    except UnboundLocalError:
        r = results[0]
    wordtext = pattern_num.findall(r["text"])

    if len(wordtext) > 0:
        wordtext = wordtext[0]
        cube = r["text_region"]
        UL = cube[0]
        UR = cube[1]
        l = UL[0]
        r = UR[0]
        h = UL[1]
        hl = cube[2][1]
        m = int((l + r) / 2)
        if deg == 180:
            Box1 = [[l, h], [m, h], [m, 250], [l, 250]]
            #print(Box0)
            Box0 = [[m, h], [r, h], [r, 250], [m, 250]]
            #print(Box1)
        else:
            Box0 = [[l, 0], [m, 0], [m, hl], [l, hl]]
            #print(Box0)
            Box1 = [[m, 0], [r, 0], [r, hl], [m, hl]]
            #print(Box1)
    return Box0, Box1, wordtext


def OCR(dst):
    result = ocr_request(dst)
    '''
    result= ["angle": r['angle'],
            "angle_conf": r['angle_conf'],
            "confidence": r["confidence"],
            "text": r["text"],
            "text_region": r["text_region"]]
    '''
    num_length = get_num_length(result)
    img = dst

    (h, w) = dst.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated = cv2.warpAffine(dst, M, (w, h))
    # cv2.imshow("Rotated", rotated)
    # cv2.waitKey()
    n_result = ocr_request(rotated)
    n_num_length = get_num_length(n_result)
    if n_num_length > num_length:
        result = n_result
        num_length = n_num_length
        img = rotated

    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey()
    n_result = ocr_request(rotated)
    n_num_length = get_num_length(n_result)
    if n_num_length >= num_length:
        result = n_result
        num_length = n_num_length
        img = rotated
    #cv2.imshow("0", img)
    angle = get_deg(result)

    B_1, B_0, NumWord = GetNumArea(result, angle)

    Xs = [i[0] for i in B_0]
    Ys = [i[1] for i in B_0]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg1 = img[y1:y1 + hight, x1:x1 + width]

    Xs = [i[0] for i in B_1]
    Ys = [i[1] for i in B_1]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg2 = img[y1:y1 + hight, x1:x1 + width]

    if angle == 180:
        (h, w) = cropImg1.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        cropImg1 = cv2.warpAffine(cropImg1, M, (w, h))
        (h, w) = cropImg2.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        cropImg2 = cv2.warpAffine(cropImg2, M, (w, h))

    #cv2.imshow("1", cropImg1)
    #cv2.imshow("2", cropImg2)
    #cv2.waitKey()
    res1 = ocr_request(cropImg1)
    res2 = ocr_request(cropImg2)
    res1 = clear_deg(res1, 0)
    res2 = clear_deg(res2, 0)
    op = ""

    for r in res1:
        pattern_word = re.compile(r'[\u4E00-\u9FA5]+')
        wordtext = pattern_word.findall(r["text"])
        if len(wordtext) > 0:
            wordtext = wordtext[0]
            op = op + wordtext

    for r in res2:
        pattern_word = re.compile(r'[\u4E00-\u9FA5]+')
        wordtext = pattern_word.findall(r["text"])
        if len(wordtext) > 0:
            wordtext = wordtext[0]
            op = op + wordtext

    return op, NumWord



def CubeProcess(img):
    try:
        image = Process(img)
        resd, resw = OCR(image)
        return resd, resw
    except :
        print("Error")
        return -1


