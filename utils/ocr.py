# -- coding: utf-8 --
import base64
import json
import logging
import os
import re

import numpy as np
import math
import cv2
import requests

en_or_num = re.compile(r"[a-zA-Z0-9]", re.I)  # [a-z]|\d
not_en_nor_num = re.compile(r"[^a-zA-Z0-9]", re.I)  # 非英文数字
RESIZE_LENGTH = 1280
MAIN_PORT = 8866
RECT_PORT = 8867
LOCAL_HOST = True


def padding_and_resize(img_, resize=RESIZE_LENGTH):
    """
    将狭长图片统一处理成1280*1280，以适应ocr
    在服务端已设置min=1280预处理，此处可以忽略
    """
    long_side = max(img_.shape[0], img_.shape[1])
    img_ = cv2.copyMakeBorder(img_, 0, long_side - img_.shape[0], 0, long_side - img_.shape[1],
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])  # padding成正方形
    if resize != 0:
        img_ = cv2.resize(img_, (resize, resize))  # resize，因为padding过后目标太小
    return img_


def circle_ocr(img_, cfg):
    # 1. 识别旋转汉字、定位=================================================
    img2 = circle_to_rectangle(img_)  # 拉直
    img2 = np.concatenate((img2, img2), axis=1)  # 横向拼接截断文本
    img2 = img2[:img2.shape[0] // 3 * 2, :]  # 只检测边缘2/3
    img2 = padding_and_resize(img2)
    res2 = rm_words(ocr_request(img2, cfg=cfg), cfg=cfg)
    for item in res2:
        item["from"] = "side"

    angle = cal_angle(res2, img2, cfg=cfg)
    # 2. 识别印章大类=================================================
    center = img_.shape[0] // 2
    mat_rotate = cv2.getRotationMatrix2D((center, center), angle, 1)  # 仿射矩阵
    img1 = cv2.warpAffine(img_, mat_rotate, (img_.shape[0], img_.shape[0]))
    res1 = rm_words(ocr_request(img1, cfg=cfg), cfg=cfg)
    for item in res1:
        item["from"] = "center"

    if not cfg["debug"]:
        # cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.jpg"), img1)
        # cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.jpg"), img2)
        pass
    if cfg["debug"]:
        # cv2.imwrite(os.path.join(cfg["to_path"], "trans_0.jpg"), img_)
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.jpg"), img1)
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.jpg"), img2)
        # cv2.imwrite(os.path.join(cfg["to_path"], "trans_3.jpg"), img3)
    return rm_words(res1 + res2, cfg=cfg)


def cal_angle(res, img_, cfg):
    if len(res) == 0:
        logging.warning(cfg["file_name"] + " no word!")
        return 0  # 无检测文字

    long_side = img_.shape[1]
    pos = (res[-1]["text_region"][0][0] + res[-1]["text_region"][1][0]) // 2  # 已经按照文本长度升序排列
    rate = (pos - long_side / 2) / (long_side / 2)
    angle = 360 * rate if len(not_en_nor_num.findall(res[-1]["text"])) < len(en_or_num.findall(res[-1]["text"])) else \
        360 * rate + 180  # 如果最长的是汉字文本则+180

    if cfg["debug"]:
        print(f"pos:{pos} rate:{rate} angle:{angle} long_side:{long_side}\n")
    return angle


def ellipse_ocr(img_, cfg):
    img1 = img_.copy() if img_.shape[0] < img_.shape[1] else np.rot90(img_.copy())

    # 1. 转正，检测中间字符
    res1 = rm_words(ocr_request(img1, cfg=cfg), cfg=cfg)
    for item in res1:
        item["from"] = "center"

    # 2. 展平 + 拼接
    img2 = circle_to_rectangle(img1)
    img2 = np.concatenate((img2, img2), axis=1)  # 横向拼接截断文本
    img2 = img2[:img2.shape[0] // 3 * 2, :]  # 只检测边缘2/3
    img2 = padding_and_resize(img2)
    res2 = rm_words(ocr_request(img2, cfg=cfg), cfg=cfg)
    for item in res2:
        item["from"] = "side"

    if not cfg["debug"]:
        # cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.jpg"), img1)
        # cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.jpg"), img2)
        pass
    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.jpg"), img1)
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.jpg"), img2)
    return rm_words(res1 + res2, cfg=cfg)


def rectangle_ocr(img, cfg):
    # 1. 识别数字，并确定90°旋转次数。注意rot90是逆时针旋转
    img_set = {0: img.copy()}
    img_set[1] = np.rot90(img_set[0])
    img_set[2] = np.rot90(img_set[1])
    img_set[3] = np.rot90(img_set[2])

    # 取四个方向的底部1/3，纵向拼接
    img_1 = np.concatenate((img_set[0][img_set[0].shape[0] // 3 * 2:, :], img_set[2][img_set[2].shape[0] // 3 * 2:, :]),
                           axis=0)
    img_2 = np.concatenate((img_set[1][img_set[1].shape[0] // 3 * 2:, :], img_set[3][img_set[3].shape[0] // 3 * 2:, :]),
                           axis=0)
    if img_1.shape[1] > img_2.shape[1]:
        img_2 = cv2.copyMakeBorder(img_2, 0, 0, 0, img_1.shape[1] - img_2.shape[1],
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif img_1.shape[1] < img_2.shape[1]:
        img_1 = cv2.copyMakeBorder(img_1, 0, 0, 0, img_2.shape[1] - img_1.shape[1],
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img1 = np.concatenate((img_1, img_2), axis=0)  # 保证width相同后才能拼接
    resize_rate = RESIZE_LENGTH / img1.shape[0]  # 要在预处理之前获取缩放比例
    img1 = padding_and_resize(img1)  # ocr预处理
    res1 = rm_words(ocr_request(img1, cfg=cfg), cfg=cfg)

    # 获取第二次ocr图
    if len(res1) != 0:
        # 纠正角度并倒推缩放前的数字框
        p1, p2, p3, p4 = (np.array(res1[-1]["text_region"]) / resize_rate).tolist()
        height = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
        if height > img.shape[0] // 3 * 2 + img.shape[1] // 3:
            pos, bias = 3, (img.shape[1] // 3 * 2) - (img.shape[0] // 3 * 2 + img.shape[1] // 3)
        elif height > img.shape[0] // 3 * 2:
            pos, bias = 1, (img.shape[1] // 3 * 2) - (img.shape[0] // 3 * 2)
        elif height > img.shape[0] // 3:
            pos, bias = 2, (img.shape[0] // 3 * 2) - (img.shape[0] // 3)
        else:
            pos, bias = 0, (img.shape[0] // 3 * 2)
        p1[1], p2[1], p3[1], p4[1] = p1[1] + bias, p2[1] + bias, p3[1] + bias, p4[1] + bias
        # 1. error: (-215:Assertion failed) p.checkVector(2, 4) >= 0 in function 'cv::fillPoly'
        # 2. TypeError: Layout of the output array img is incompatible with cv::Mat
        #    (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)
        img2 = cv2.fillPoly(img_set[pos].copy().astype(np.uint8),
                            np.array([[p1, p2, p3, p4]]).astype(np.int32), (255, 255, 255))
    else:
        logging.warning(cfg["file_name"] + " no num!")
        res1 = [{"text": "", "confidence": 0}]
        img2 = img_set[0]

    # 2. 识别方章单字
    res2 = ocr_request(img2, cfg=cfg, port=RECT_PORT)  # 使用方章专属检测模型接口

    # 顺序拼接文本
    pic_center_x, pic_center_y = img2.shape[0] / 2, img2.shape[1] / 2
    cn_list = ["", "", "", ""]
    score = 0
    length = 0
    for item in res2:
        x, y = get_center(item)
        if len(en_or_num.findall(item["text"])) != 0:  # 识别汉字的时候丢弃所有数字，除非是完整的数字
            if item["confidence"] > res1[0]["confidence"] and len(item["text"]) == 13:
                res1 = [item]
            continue
        if x > pic_center_x and y < pic_center_y:
            cn_list[0] = item["text"]
        elif x > pic_center_x and y > pic_center_y:
            cn_list[1] = item["text"]
        elif x < pic_center_x and y < pic_center_y:
            cn_list[2] = item["text"]
        else:
            cn_list[3] = item["text"]
        score += item["confidence"]
        length += 1
    score = score / length
    cn_text = ""  # 拼接文本
    for word in cn_list:
        cn_text = cn_text + word

    if cn_list[0] == "" or cn_list[1] == "" or not cn_text.endswith("印"):
        score = 0  # 缺失文字，直接返回0置信度

    if not cfg["debug"]:
        # cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.jpg"), img1)
        # cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.jpg"), img2)
        pass
    if cfg["debug"]:
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_1.jpg"), img1)
        cv2.imwrite(os.path.join(cfg["to_path"], cfg["file_name"] + "_2.jpg"), img2)
        pass

    return [res1[-1]] + [{"text": cn_text, "confidence": score}]


def get_center(item):
    p1, p2, p3, p4 = item["text_region"]
    item_center_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
    item_center_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
    return item_center_x, item_center_y


def rm_words(res, cfg):
    # debug
    if cfg["debug"]:
        print("消去前===================")
        for item in res:
            print(item["text"])

    # 1. 过滤短文本
    save_list = []
    for item in res:
        num_len = len(en_or_num.findall(item["text"]))
        ch_len = len(not_en_nor_num.findall(item["text"]))
        if num_len == 13 or num_len == 18 or ch_len >= 5:
            save_list.append(item)
    res = save_list

    # 2. 去除中文文本中的数字和英文，以及去除英文数字文本中的中文
    # 这是有必要的，因为有可能会出现"郑91410182MA9FNWR87J回", 甚至"郑""回"是横着的
    # 当前版本没有上述情况，但为了防止意外。提升速度时可以删去
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


def ocr_request(img, cfg,
                url="127.0.0.1",
                port=MAIN_PORT,
                mode="ocr_system"):
    """
    return [{
        "confidence": 0 - 1,
        "text": str,
        "text_region": [[x, y] * 4]
    } * n ]
    """

    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')

    if LOCAL_HOST:
        url = "127.0.0.1"  # 使用相同容器下的网络
    # 发送HTTP请求
    r = requests.post(url=f"http://{url}:{port}/predict/{mode}",
                      data=json.dumps({'images': [cv2_to_base64(img)]}),
                      headers={"Content-type": "application/json"})
    result = (r.json())["results"][0]

    if cfg["debug"]:
        # print(result)
        pass
    return result


def plot_ocr(img, result):
    pass
