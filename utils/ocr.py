# -- coding: utf-8 --
import base64
import json
import os
import re

import numpy as np
import math
import cv2
import requests


def circle_ocr(img_, cfg):
    # 转正圆形内容(五角星？
    # 检测无偏转文字，确定印章种类
    cls = ["财务专用章", "发票专用章", "合同专用章", ]
    # cv2.imshow("img_", img_)
    # 上下翻转，分别检测文字和数字
    img_ = circle_to_rectangle(img_, 1)
    res = ocr_request(img_, cfg=cfg)
    return res


def ellipse_ocr(img_, cfg):
    res = []
    # 检测中间字符
    img1 = img_.copy() if img_.shape[0] < img_.shape[1] else np.rot90(img_.copy())
    res = res + ocr_request(img1, cfg=cfg)
    # 展平
    img2 = circle_to_rectangle(img1, 0)
    res = res + ocr_request(img2, cfg=cfg)
    # 展平
    img3 = circle_to_rectangle(img1, 1)
    res = res + ocr_request(img3, cfg=cfg)
    # 去重
    res = rm_words(res, cfg=cfg)

    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_1.jpg"), img1)
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_2.jpg"), img2)
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_3.jpg"), img3)
        with open(os.path.join(cfg["to_path"], "text.json"), mode="w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
    return res


def rectangle_ocr(img_, cfg):
    pass


def rm_words(res, cfg):
    letters_numbers = re.compile(r"[a-zA-Z0-9]", re.I)  # [a-z]|\d
    not_letters_numbers = re.compile(r"[^a-zA-Z0-9]", re.I)

    # 1. 过滤短文本
    for t in res:
        if len(t["text"]) < 5:
            res.remove(t)

    # 2. 去除中文文本中的数字和英文，以及去除英文数字文本中的中文
    for idx in range(len(res)):
        l_n = letters_numbers.findall(res[idx]["text"])
        n_l_n = not_letters_numbers.findall(res[idx]["text"])
        if len(l_n) != 0 and len(n_l_n) != 0:
            if len(l_n) > len(n_l_n):
                for l in n_l_n:
                    res[idx]["text"] = res[idx]["text"].replace(l, "")
            else:
                for l in l_n:
                    res[idx]["text"] = res[idx]["text"].replace(l, "")

    # 3. 过滤重复局部文本
    # 按照 文本长度*置信度 升序排序
    res.sort(key=lambda s: len(s["text"]) * s["confidence"], reverse=False)
    rm_ls = []
    for idx in range(len(res)):
        for j in range(idx + 1, len(res)):
            count = 0
            for t in res[idx]["text"]:  # 统计短文本的字符在长文本中出现的次数
                if t in res[j]["text"]:
                    count += 1
            rate = count / len(res[idx]["text"])
            if rate > 0.5:  # 如果短文本中超过一半都被包含在长文本中，则删去
                rm_ls.append(idx)
                print(f'{res[idx]["text"]} -> {res[j]["text"]} rate={rate} i={idx}')
                break
    # 降序删去重复字段
    rm_ls.sort(reverse=True)
    for idx in rm_ls:
        res.pop(idx)

    # debug
    if cfg["debug"]:
        for t in res:
            print(t)
    return res


def circle_to_rectangle(cir_img, flip):
    """
    对圆和椭圆的旋转文字进行极坐标到直角坐标转换
    :param flip: 上半圈用 1 , 下半圈用 0
    :type flip: int
    :param cir_img: 经过剪裁的图片，要边界精准、角度转正
    :type cir_img:
    :return: 坐标变换后的矩形
    :rtype:
    """
    x0, y0 = cir_img.shape[0] // 2, cir_img.shape[1] // 2
    # init
    radius = (x0 + y0) // 2
    rect_height = radius
    rect_width = int(2 * math.pi * radius)
    rect_img = np.zeros((rect_height, rect_width, 3), dtype="u1")

    except_count = 0
    for j in range(rect_width):
        theta = 2 * math.pi * (j / rect_width) + math.pi * (1 - flip)  # start position such as "+ math.pi"
        for i in range(rect_height):
            # 适应椭圆
            x = (x0 - i) * math.cos(theta) + x0  # "sin" is clockwise but "cos" is anticlockwise
            y = (y0 - i) * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                rect_img[i, j, :] = cir_img[x, y, :]
            except Exception:
                except_count = except_count + 1
    # print(f"{except_count} lost in circle_to_rectangle")
    rect_img = cv2.flip(rect_img, flip)

    return rect_img


def ocr_request(img, cfg):
    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')

    # 发送HTTP请求
    data = {'images': [cv2_to_base64(img)]}
    headers = {"Content-type": "application/json"}
    url = "http://47.101.136.120:8866/predict/ocr_system"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    # 打印预测结果
    res = (r.json())["results"][0]  # [{}*n]
    result = []
    for r in res:
        result.append({
            "confidence": r["confidence"],
            "text": r["text"],
        })
        if cfg["debug"]:
            # print(r)
            pass
    return result


def plot_ocr(img):
    pass
