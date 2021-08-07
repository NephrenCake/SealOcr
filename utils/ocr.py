# -- coding: utf-8 --
import base64
import json
import os
import re

import numpy as np
import math
import cv2
import requests

# img2 = np.rot90(np.rot90(img2))
# cv2.imshow("img3", img1)
# cv2.waitKey(0)
from utils.CubeOCR import OCR

en_or_num = re.compile(r"[a-zA-Z0-9]", re.I)  # [a-z]|\d
not_en_nor_num = re.compile(r"[^a-zA-Z0-9]", re.I)  # 非英文数字


def circle_ocr(img_, img_bw, cfg):

    # 1. 识别旋转汉字、定位=================================================
    img2 = circle_to_rectangle(img_)  # 拉直
    img2 = np.concatenate((img2, img2), axis=1)  # 横向拼接截断文本
    img2 = img2[:img2.shape[0] // 3 * 2, :]  # 只检测边缘2/3
    img_bw2 = circle_to_rectangle(img_bw)  # 拉直
    img_bw2 = np.concatenate((img_bw2, img_bw2), axis=1)  # 横向拼接截断文本
    img_bw2 = img_bw2[:img_bw2.shape[0] // 3 * 2, :]  # 只检测边缘2/3
    img2 = np.concatenate((img2, img_bw2), axis=0)  # 同时预测彩色和黑白
    res2 = rm_words(ocr_request(img2, cfg=cfg),
                    cfg=cfg)

    angle = cal_angle(res2, img2, cfg=cfg)
    # 2. 识别印章大类=================================================
    center = img_.shape[0] // 2
    mat_rotate = cv2.getRotationMatrix2D((center, center), angle, 1)  # 仿射矩阵
    img1 = cv2.warpAffine(img_, mat_rotate, (img_.shape[0], img_.shape[0]))
    res1 = rm_words(ocr_request(img1, cfg=cfg),
                    cfg=cfg)

    if len(res1) != 0:  # 消去五大章字样
        for item in res1:
            if item["text"].endswith("章"):
                img1 = cv2.fillPoly(img1, np.int32([item["text_region"]]), (255, 255, 255))
                break
    # 3. 识别下方数字=================================================
    img3 = circle_to_rectangle(img1, start=60)  # 拉直
    img3 = img3[:img3.shape[0] // 2, :img3.shape[1] // 3]  # 只检测边缘1/2和下1/3
    img3 = np.rot90(np.rot90(img3))  # 转正数字
    res3 = rm_words(ocr_request(img3, cfg=cfg),
                    cfg=cfg)
    # 4. 再次校准旋转汉字=================================================
    img4 = circle_to_rectangle(img1)  # 拉直
    img4 = img4[:img4.shape[0] // 3 * 2, :]  # 只检测边缘2/3
    res4 = rm_words(ocr_request(img4, cfg=cfg),
                    cfg=cfg)

    if not cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.jpg"), img1)
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.jpg"), img2)
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_3.jpg"), img3)
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_4.jpg"), img4)
        with open(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res1, f, ensure_ascii=False)
        with open(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res2, f, ensure_ascii=False)
        with open(os.path.join(cfg["to_path"], cfg["file_name"] + "_3.json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res3, f, ensure_ascii=False)
        with open(os.path.join(cfg["to_path"], cfg["file_name"] + "_3.json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res4, f, ensure_ascii=False)
    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_0.jpg"), img_)
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_1.jpg"), img1)
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_2.jpg"), img2)
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_3.jpg"), img3)
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_4.jpg"), img4)
    return rm_words(res1 + res2 + res3 + res4, cfg=cfg)


def cal_angle(res, img2, cfg):
    if len(res) == 0:
        print(cfg["file_name"] + " no word")
        return 0  # 无检测文字

    long_side = img2.shape[1]
    pos = (res[-1]["text_region"][0][0] + res[-1]["text_region"][1][0]) // 2  # 已经按照文本长度升序排列
    rate = (pos - long_side / 2) / (long_side / 2)
    angle = 360 * rate if len(not_en_nor_num.findall(res[-1]["text"])) < len(en_or_num.findall(res[-1]["text"])) else \
        360 * rate + 180  # 如果最长的是汉字文本则+180

    if cfg["debug"]:
        print(f"pos:{pos} rate:{rate} angle:{angle} long_side:{long_side}\n")
    return angle


def ellipse_ocr(img_, img_bw, cfg):
    # 1. 转正，检测中间字符
    img1 = img_.copy() if img_.shape[0] < img_.shape[1] else np.rot90(img_.copy())

    res1 = rm_words(ocr_request(img1, cfg=cfg),
                    cfg=cfg)

    if len(res1) != 0:
        for item in res1:
            if item["text"].endswith("章"):
                try:
                    img1 = cv2.fillPoly(img1, np.int32([item["text_region"]]), (255, 255, 255))
                except:
                    img1 = img1.copy()
                    img1 = cv2.fillPoly(img1, np.int32([item["text_region"]]), (255, 255, 255))
                break
    # 2. 展平 + 拼接
    img2 = circle_to_rectangle(img1)
    img2 = np.concatenate((img2, img2), axis=1)  # 横向拼接截断文本
    img2 = img2[:img2.shape[0] // 3 * 2, :]  # 只检测边缘2/3

    res2 = rm_words(ocr_request(img2, cfg=cfg),
                    cfg=cfg)

    if not cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.jpg"), img1)
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.jpg"), img2)
        with open(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res1, f, ensure_ascii=False)
        with open(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res2, f, ensure_ascii=False)
    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_1.jpg"), img1)
        cv2.imwrite(os.path.join(cfg["to_path"], "trans_2.jpg"), img2)
    return rm_words(res1 + res2, cfg=cfg)


def rectangle_ocr(img, img_bw, cfg):
    # 1. 确定90°旋转次数
    # img_r0 = img.copy()
    # img_r1 = np.rot90(img_r0)
    # img_r2 = np.rot90(img_r1)
    # img_r3 = np.rot90(img_r2)
    #
    # img2 = img2[:img2.shape[0] // 4, :]
    # img_1 = np.concatenate((
    #     img_r0[:img_r0.shape[0] // 3 * 2, :]
    # ), axis=0)

    word, num, = OCR(img)
    return [word, num]


def rm_words(res, cfg):
    # debug
    if cfg["debug"]:
        print("消去前===================")
        for item in res:
            print(item["text"])

    # 1. 过滤短文本
    save_list = []
    for item in res:
        if len(item["text"]) >= 5:
            save_list.append(item)
    res = save_list

    # 2. 去除中文文本中的数字和英文，以及去除英文数字文本中的中文
    # 这是有必要的，因为有可能会出现"郑91410182MA9FNWR87J回", 甚至"郑""回"是横着的
    for idx in range(len(res)):
        txt = res[idx]["text"]
        l_n = en_or_num.findall(txt)
        n_l_n = not_en_nor_num.findall(txt)
        if len(l_n) != 0 and len(n_l_n) != 0:
            if len(l_n) > len(n_l_n):
                for l in n_l_n:
                    res[idx]["text"] = res[idx]["text"].replace(l, "")
            else:
                for l in l_n:
                    res[idx]["text"] = res[idx]["text"].replace(l, "")

    # 2. 过滤重复局部文本
    rm_idx_list = []
    res.sort(key=lambda s: (len(s["text"]), s["confidence"]), reverse=False)  # 按照 文本长度、置信度 升序排序
    for idx in range(len(res)):
        for j in range(idx + 1, len(res)):
            count = 0
            for item in res[idx]["text"]:  # 统计短文本的字符在长文本中出现的次数
                if item in res[j]["text"]:
                    count += 1
            rate = count / len(res[idx]["text"])
            if rate > 0.5:  # 如果短文本中超过一半都被包含在长文本中，则删去
                rm_idx_list.append(idx)
                if cfg["debug"]:
                    print(f'{res[idx]["text"]} -> {res[j]["text"]} rate={rate} i={idx}')
                break
    save_list = []
    for idx, item in enumerate(res):
        if idx not in rm_idx_list:
            save_list.append(item)
    res = save_list

    # debug
    if cfg["debug"]:
        print("消去后===================")
        for item in res:
            print(item["text"])
    return res


def circle_to_rectangle(cir_img, start=0):
    """
    对圆和椭圆的旋转文字进行极坐标到直角坐标转换
    默认从正下方开始切，可以通过start设定偏移角度，单位为度
    """
    x0, y0 = cir_img.shape[0] // 2, cir_img.shape[1] // 2
    # init
    radius = (x0 + y0) // 2
    rect_height = radius
    rect_width = int(2 * math.pi * radius)
    rect_img = np.zeros((rect_height, rect_width, 3), dtype="u1")

    except_count = 0
    for j in range(rect_width):
        theta = 2 * math.pi * (j / rect_width) + 2 * math.pi * (start / 360)  # start position such as "+ math.pi"
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
    rect_img = cv2.flip(rect_img, 1)  # 纠正水平镜像

    return rect_img


def ocr_request(img, cfg):
    """
    return [{
                "angle": 0 / 180,
                "angle_conf": 0 - 1,
                "confidence": 0 - 1,
                "text": str,
                "text_region": [[x, y] * 4]
            } * n ]
    """

    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')

    # 发送HTTP请求
    r = requests.post(url="http://47.101.136.120:8866/predict/ocr_system",
                      data=json.dumps({'images': [cv2_to_base64(img)]}),
                      headers={"Content-type": "application/json"})
    result = (r.json())["results"][0]

    if cfg["debug"]:
        print(result)
    return result


def plot_ocr(img):
    pass
