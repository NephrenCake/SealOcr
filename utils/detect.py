# -- coding: utf-8 --
import os
import math
import numpy as np
import cv2
import json


def enlarge_img(img):
    # 边界+50%，防止小尺寸图像中目标溢出边界，特别是旋转椭圆
    h = img.shape[0] // 2
    w = img.shape[1] // 2
    img_ = cv2.copyMakeBorder(img, h, h, w, w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img_


# extract_red() filter_non_red() 选一用就行了
def extract_red(img, cfg):
    """
    返回: 前景为红色，背景为黑色
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 印章一般浅红较多，而深红较少。因此为了保留更多的红色，应该更加偏向于保留白色，尽可能杀掉黑色。
    # 尽可能抓更多的红色，反正也没其他颜色
    # 区间1
    lower_red = np.array([0, 0, 177])
    upper_red = np.array([30, 0, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 区间2
    lower_red = np.array([150, 0, 177])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 在空白画布上拼接两个区间
    mask = mask0 + mask1
    mask = cv2.add(img, np.full(np.shape(img), 0, dtype=np.uint8), mask=mask)
    # 底色为黑色，把黑色转白。效率和 filter_non_red() 差不多，就先搁着了
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if all(mask[i, j] == [0, 0, 0]):
    #             mask[i, j] = [255, 255, 255]

    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], "1_red.jpg"), mask)
    return mask


def filter_non_red(img, cfg):
    """
    返回: 前景红色，背景白色
    """
    img_ = np.copy(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = img_hsv.shape
    for i in range(h):
        for j in range(w):
            h = img_hsv[i, j, 0]
            not_red = not (0 <= h <= 30 or 150 <= h <= 180)  # 尽可能抓更多的红色，反正也没其他颜色
            v = int(img_hsv[i, j, 2])
            s = int(img_hsv[i, j, 1])
            is_black = False
            if v + s <= 255 or v < 127:  # 黑色要尽量杀干净，不然k-means要出问题
                is_black = True
            if not_red or is_black:
                # print(i, j)
                img_[i, j, :] = 255

    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], "1_red.jpg"), img_)
    return img_


def k_means(img, cfg):
    """
    输入为红色前景，白色背景
    """
    # convert to np.float32
    data = np.float32(img.reshape((-1, 3)))
    k = 2  # 二分类，红/白
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)  # 返回两种分类群的中心数值
    mean = int(np.mean(center))  # 计算不同图片的不同分界阈值，即最能区分两个分类群的灰度值
    result = center[label.flatten()]
    result = result.reshape(img.shape)  # 重构图像

    # 灰度
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # 二值
    ret, thresh = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)  # 63-255保留深红
    # 翻转颜色为: 前景为白，背景为黑
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            thresh[i, j] = 255 - thresh[i, j]

    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], "2_k-means.jpg"), result)
        cv2.imwrite(os.path.join(cfg["to_path"], "3_thresh.jpg"), thresh)
    return thresh


def erode_dilate(img, category, cfg):
    """
    仅接受二值
    对于椭圆不需要膨胀太多，对于圆需要
    """
    # todo 调参啊
    if category=="椭圆":
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)  # 保留边缘
        img = cv2.erode(img, kernel, iterations=1)  # 去噪
        # kernel = np.ones((6, 6), np.uint8)  # 774
        img = cv2.dilate(img, kernel, iterations=3)  # 填充边缘
    else:
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)  # 保留边缘
        img = cv2.erode(img, kernel, iterations=1)  # 去噪
        kernel = np.ones((7, 7), np.uint8)  # 774
        img = cv2.dilate(img, kernel, iterations=5)  # 填充边缘

    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], "4_open.jpg"), img)
    return img


def find_max(opening, cfg):
    # 查找轮廓             opencv-contrib-python-3.4.1.15
    try:
        _, contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 1, 2
    except:
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    return contours, max_idx


def fit_shape(img, contours, max_idx, cfg):
    """
    |- 拟合最小圆、最小矩形，比较面积
      -> 圆小，那就是圆
      -> 矩形小，那可能是矩形章或者椭圆章
        |- 通过long_side short_side判断是长方形还是正方形
          -> 正方形则是矩形章
          -> 长方形为椭圆章
    """
    cnt = contours[max_idx]
    # 拟合圆
    c = cv2.minEnclosingCircle(cnt)  # (中心(x,y), 半径)
    c_x, c_y, c_r = int(c[0][0]), int(c[0][1]), int(c[1])
    # 拟合矩形
    r = cv2.minAreaRect(cnt)  # (中心(x,y), (宽,高), 旋转角度)
    r_x, r_y, r_w, r_h, r_t = int(r[0][0]), int(r[0][1]), int(r[1][0]), int(r[1][1]), int(r[2])
    r_box = cv2.boxPoints(r)  # cv2.cv.BoxPoints -> cv2.boxPoints() 注意是小写b，不是大写
    r_box = np.int0(r_box)  # [[]*4]
    # 拟合椭圆
    if len(cnt) > 5:
        ellipse = cv2.fitEllipse(cnt)
        e_x, e_y, e_a, e_b, e_t = int(ellipse[0][0]), int(ellipse[0][1]), int(ellipse[1][0] / 2), int(
            ellipse[1][1] / 2), int(ellipse[2])
        e_w, e_h = 2 * e_a, 2 * e_b
        e_box = cv2.boxPoints(((e_x, e_y), (e_w, e_h), e_t))
        e_box = np.int0(e_box)
    else:
        e_x, e_y, e_a, e_b, e_t = c_x, c_y, c_r, c_r, 0
        e_w, e_h = 2 * e_a, 2 * e_b
        e_box = cv2.boxPoints(((e_x, e_y), (e_w, e_h), e_t))
        e_box = np.int0(e_box)
    # ==========判断形状
    c_size = int(math.pi * c_r * c_r)
    r_size = r_w * r_h
    e_size = e_w * e_h
    # 用拟合的椭圆来确定 bounding box 的形状
    e_err = ((e_w - e_h) * (e_w - e_h)) / (e_w + e_h)
    r_err = ((r_w - r_h) * (r_w - r_h)) / (r_w + r_h)

    if c_size > r_size:
        # todo 调参
        if e_err > 2 and r_err > 0.3 and r_size > e_size * 3 / 4 \
                or 40 < (e_t + 180) % 90 < 50 and e_err > 3 and r_size > e_size * 9 / 10:
            # 椭圆
            cls = "ellipse"
            (b_x, b_y, b_w, b_h, b_t) = (e_x, e_y, e_w, e_h, e_t)
            b_box = e_box
        else:
            # 矩形
            cls = "rectangle"
            (b_x, b_y, b_w, b_h, b_t) = (r_x, r_y, r_w, r_h, r_t)
            b_box = r_box
    else:
        # 圆
        cls = "circle"
        (b_x, b_y, b_w, b_h, b_t) = (c_x, c_y, 2 * c_r, 2 * c_r, 0)
        b_box = r_box
    distance = ((c_x - r_x) * (c_x - r_x) + (c_y - r_y) * (c_y - r_y) + (c_x - e_x) * (c_x - e_x) + (c_y - e_y) * (
            c_y - e_y)) / 2
    if e_err < 0.3 and r_err < 0.3 and distance > 100:  # 处理特例边缘明显噪声
        # 圆
        cls = "circle"
        (b_x, b_y, b_w, b_h, b_t) = (e_x, e_y, e_w, e_h, e_t)
        b_box = r_box

    result = {
        "class": cls,
        "box": {
            "b_x": b_x,
            "b_y": b_y,
            "b_w": b_w,
            "b_h": b_h,
            "b_t": b_t
        }
    }

    if cfg["debug"]:
        # 绘制掩膜
        mask = np.zeros(np.shape(img), dtype=np.uint8)
        mask = cv2.drawContours(mask, contours, max_idx, (138, 43, 226), -1)  # 填充最大轮廓
        img = cv2.addWeighted(img, 0.5, mask, 0.7, 0)
        # 绘制最小圆
        # print("最小圆: ", c)
        cv2.circle(img, (c_x, c_y), 5, (0, 255, 0), -1)
        cv2.circle(img, (c_x, c_y), c_r, (0, 255, 0), 5)
        # 绘制最小矩形
        # print("最小矩形: ", r)
        # cv2.circle(img, (r_x, r_y), 5, (0, 0, 255), -1)
        img = cv2.drawContours(img, [r_box], 0, (0, 0, 255), 5)
        # 绘制最小椭圆
        cv2.ellipse(img, (e_x, e_y), (e_a, e_b), e_t, 0, 360, (255, 0, 0), 5)
        # 绘制box
        cv2.circle(img, (b_x, b_y), 5, (0, 0, 255), -1)
        img = cv2.drawContours(img, [b_box], 0, (255, 0, 255), 5)

        cv2.imwrite(os.path.join(cfg["to_path"], "5_interest.jpg"), img)
        with open(os.path.join(cfg["to_path"], "detect_result.json"), "w") as f:
            json.dump(result, f)
        print(f"c_size:{c_size} r_size:{r_size} e_size:{e_size} e_err:{e_err} r_err:{r_err} distance:{distance}")
    return result


def get_area(img, contours, max_idx, category):
    cnt = contours[max_idx]

    # 拟合圆
    c = cv2.minEnclosingCircle(cnt)  # (中心(x,y), 半径)
    c_x, c_y, c_r = int(c[0][0]), int(c[0][1]), int(c[1])
    cls = "circle"
    (b_x, b_y, b_w, b_h, b_t) = (c_x, c_y, 2 * c_r, 2 * c_r, 0)
    # 拟合椭圆
    if category == "椭圆":
        if len(cnt) > 5:  # 需要5个点以上才能拟合椭圆
            ellipse = cv2.fitEllipse(cnt)
            e_x, e_y, e_a, e_b, e_t = int(ellipse[0][0]), int(ellipse[0][1]), int(ellipse[1][0] / 2), int(
                ellipse[1][1] / 2), int(ellipse[2])
            e_w, e_h = 2 * e_a, 2 * e_b
        else:
            e_x, e_y, e_a, e_b, e_t = c_x, c_y, c_r, c_r, 0
            e_w, e_h = 2 * e_a, 2 * e_b
        cls = "eclipse"
        (b_x, b_y, b_w, b_h, b_t) = (e_x, e_y, e_w, e_h, e_t)

    return {
        "class": cls,
        "box": {
            "b_x": b_x,
            "b_y": b_y,
            "b_w": b_w,
            "b_h": b_h,
            "b_t": b_t
        }
    }


def rotate_cut(img, det, cfg):
    height = img.shape[0]
    width = img.shape[1]

    # getRotationMatrix2D 函数可获取旋转的仿射矩阵
    # 参数依次为（旋转中心，旋转角度，缩放比例）
    mat_rotate = cv2.getRotationMatrix2D((det["box"]["b_x"], det["box"]["b_y"]), det["box"]["b_t"], 1)
    img_ = cv2.warpAffine(img, mat_rotate, (width, height))
    x, y, w, h = det["box"]["b_x"], det["box"]["b_y"], det["box"]["b_w"], det["box"]["b_h"]
    p1_x, p1_y = x - w // 2, y - h // 2
    p2_x, p2_y = x + w // 2, y + h // 2
    # cv2.line(img_, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0), 5)
    # cv2.imshow("im", img_)
    # cv2.waitKey(0)
    img_ = img_[p1_y:p2_y, p1_x:p2_x]
    img_ = np.rot90(img_)

    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], "6_cut.jpg"), img_)
    return img_
