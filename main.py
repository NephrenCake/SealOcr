# -- coding: utf-8 --
import os
import time
import json
import cv2
import numpy as np
import re

from utils.ocr import circle_to_rectangle, ocr_request, ellipse_ocr, circle_ocr, rectangle_ocr
from utils.detect import erode_dilate, find_max, fit_shape, filter_non_red, k_means, rotate_cut, enlarge_img, get_area
from utils.Model import Model
from utils.CubeOCR import CubeProcess
import logging

model = Model(0)


def work(cfg):
    category, img = model.predict(cfg['img_path'])
    if category == '正方形':
        word, num, = CubeProcess(img)
        return img, category, {'文字：': word, '数字：': num}
    else:
        # 将原画布扩大，防止小尺寸图像在开运算时使边界溢出，那样会在 rotate_cut() 部分出错
        img = enlarge_img(img)
        img_ = img.copy()
        # ===========目标检测与分类
        t0 = time.time()
        # 提取红色部分
        img_ = filter_non_red(img_, cfg=cfg)
        # k-meas聚类
        img_ = k_means(img_, cfg=cfg)
        # 开运算去噪填充
        img_ = erode_dilate(img_, cfg=cfg)
        # 查找最大轮廓
        contours, max_idx = find_max(img_, cfg=cfg)
        # 检测并分类目标
        # det = fit_shape(img, contours, max_idx, cfg=cfg)
        det = get_area(img, contours, max_idx, category)  # 不进行分类
        # 截取目标区域
        img_ = rotate_cut(img, det, cfg=cfg)
        t1 = time.time()

        # print(f"{cfg['img_path']} done in {t1 - t0}s. class={det['class']}")
        # ============分类处理目标区域
        if category == '圆形':
            res = circle_ocr(img_, cfg=cfg)
        else:
            res = ellipse_ocr(img_, cfg=cfg)
        t2 = time.time()
        print(f"{cfg['img_path']} done in {round(t1 - t0, 2)}+{round(t2 - t1, 2)}s. class={det['class']}")
        logging.info(f"{cfg['img_path']} done in {round(t1 - t0, 2)}+{round(t2 - t1, 2)}s. class={det['class']}")

        num = ''
        words = ''
        print(res)
        for i in res:
            if bool(re.search(r'\d', i['text'])):
                num += i['text']
            else:
                words = i['text'] + words
        res = {'文字:': words, '数字:': num}
        return img_, category, res


def one_pic(file_, to_path_, opt_):
    fn = file_.split("/")[-1]
    if opt_["debug"]:
        to_path_ = os.path.join("debug", fn.split(".")[0])
        if not os.path.exists(to_path_):
            os.mkdir(to_path_)
    assert os.path.exists(to_path_)
    cfg = {
        "img_path": file_,
        "to_path": to_path_,
        "debug": opt_["debug"]
    }

    result, cls, res = work(cfg)
    if not opt_["debug"]:
        # print(os.path.join(cfg["to_path"], cls + "_" + fn.split("\\")[-1]))
        # input("wait")
        # print(cfg["to_path"], cls + "_" + fn.split("/")[-1])
        with open(os.path.join(cfg["to_path"], cls + "_" + fn.split("/")[-1]).replace(".jpg", ".json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        cv2.imwrite(os.path.join(cfg["to_path"], cls + "_" + fn.split("/")[-1]), result)
    return res


def main(opt):
    # 可以识别文件or文件夹，统一为list
    file_list = []
    file_name = opt["source"].split("/")[-2] if opt["source"].endswith("/") else opt["source"].split("/")[-1]
    if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".bmp"):
        file_list.append(opt["source"])
    elif "." not in file_name:
        t_l = os.listdir(opt["source"])
        for t in t_l:
            if t.endswith(".jpg") or t.endswith(".png") or t.endswith(".bmp"):
                file_list.append(os.path.join(opt["source"], t))
    print(f"{len(file_list)} images found.")

    # 检查输出文件夹
    to_path = None
    if opt["debug"]:
        if not os.path.exists("debug"):
            os.mkdir("debug")
    else:
        if not os.path.exists("run"):
            os.mkdir("run")
        to_path = os.path.join("run", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        if not os.path.exists(to_path):
            os.mkdir(to_path)

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.log", level=logging.INFO,
                        format=LOG_FORMAT)

    text = None
    for file in file_list:
        text = one_pic(file_=file, to_path_=to_path, opt_=opt)
        # try:
        #    res = one_pic(file_=file, to_path_=to_path, opt_=opt)
        # except Exception:
        #    logging.error(f"error occurred in {file}")

    # 线程池方法
    # with ThreadPoolExecutor(50) as t:
    #     for file in file_list:
    #         print("submit!")
    #         t.submit(one_pic, file_=file, to_path_=to_path, opt_=opt)

    logging.info(f"all completed {time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
    return text


if __name__ == '__main__':
    # 参数设置
    opt = {
        "source": r'D:\ProjectFiles\Seal_Text_Detection\Seal_Text_Detection\seal_source/4101035073055.jpg',
        "debug": True,  # debug模式将可视化各环节，否则只输出结果
    }
    main(opt)
