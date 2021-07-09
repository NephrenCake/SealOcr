# -- coding: utf-8 --
import os
import time
import json
import cv2
import numpy as np

from utils.ocr import circle_to_rectangle, ocr_request, ellipse_ocr, circle_ocr, rectangle_ocr
from utils.detect import erode_dilate, find_max, fit_shape, filter_non_red, k_means, rotate_cut, enlarge_img
import logging


def main(cfg):
    img = cv2.imread(cfg["img_path"])
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
    det = fit_shape(img, contours, max_idx, cfg=cfg)
    # 截取目标区域
    img_ = rotate_cut(img, det, cfg=cfg)
    t1 = time.time()
    # print(f"{cfg['img_path']} done in {t1 - t0}s. class={det['class']}")

    # ============分类处理目标区域
    res = []
    if det["class"] == "circle":
        res = circle_ocr(img_, cfg=cfg)
    elif det["class"] == "ellipse":
        res = ellipse_ocr(img_, cfg=cfg)
    elif det["class"] == "rectangle":
        res = rectangle_ocr(img_, cfg=cfg)

    t2 = time.time()
    print(f"{cfg['img_path']} done in {round(t1 - t0, 2)}+{round(t2 - t1, 2)}s. class={det['class']}")
    logging.info(f"{cfg['img_path']} done in {round(t1 - t0, 2)}+{round(t2 - t1, 2)}s. class={det['class']}")

    return img_, det["class"], res


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

    result, cls, res = main(cfg)
    if not opt_["debug"]:
        # print(os.path.join(cfg["to_path"], cls + "_" + fn.split("\\")[-1]))
        # input("wait")
        # print(cfg["to_path"], cls + "_" + fn.split("/")[-1])
        with open(os.path.join(cfg["to_path"], cls + "_" + fn.split("/")[-1]).replace(".jpg", ".json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        cv2.imwrite(os.path.join(cfg["to_path"], cls + "_" + fn.split("/")[-1]), result)


if __name__ == '__main__':
    # 参数设置
    opt = {
        "source": r'seal_source/4101035073055.jpg',
        "debug": True,  # debug模式将可视化各环节，否则只输出结果
    }

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

    # 处理图片
    # with ThreadPoolExecutor(10) as t:
    #     for file in file_list:
    #         print("submit!")
    #         t.submit(one_pic, file_=file, to_path_=to_path, opt_=opt)

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.log", level=logging.INFO,
                        format=LOG_FORMAT)

    for file in file_list:
        # one_pic(file_=file, to_path_=to_path, opt_=opt)
        try:
            one_pic(file_=file, to_path_=to_path, opt_=opt)
        except Exception:
            logging.error(f"error occurred in {file}")

    logging.info(f"all completed {time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
