# -- coding: utf-8 --
import os
import time
import json
import cv2
from utils.ocr import ellipse_ocr, circle_ocr, rectangle_ocr
from utils.detect import erode_dilate, find_max, filter_non_red, k_means, rotate_cut, enlarge_img, get_area, Differ_kmeans
from utils.Model import Model
from utils.CubeOCR import CubeProcess
import logging
model = Model(0)

def Precisefilter(img,cfg):
    img_ = filter_non_red(img, cfg=cfg)
    t1 = time.time()
    # k-meas聚类
    img_ = k_means(img_, cfg=cfg)
    img_bw = cv2.merge([img_, img_, img_])
    return img_,img_bw

def Lazyfilter(img,cfg):
    img_ = Differ_kmeans(img,cfg)
    img_bw = cv2.merge([img_,img_,img_])
    return img_,img_bw

def work(cfg):
    category, img = model.predict(cfg['img_path'])
    img = enlarge_img(img)
    img_ = img.copy()
    t0 = time.time()
    img_,img_bw = Lazyfilter(img_,cfg)
    t2 = time.time()  # kmeans = t2 - t1
    # 开运算去噪填充
    img_ = erode_dilate(img_, category, cfg=cfg)
    t3 = time.time()
    # 查找最大轮廓
    contours, max_idx = find_max(img_)
    t4 = time.time()
    # 检测并分类目标
    # det = fit_shape(img, contours, max_idx, cfg=cfg)
    det = get_area(img, contours, max_idx, category, cfg=cfg)  # 不进行分类
    t5 = time.time()
    # 截取目标区域  img:原图 img_bw:二值
    img = rotate_cut(img, det, cfg=cfg)
    img_bw = rotate_cut(img_bw, det, cfg=cfg)
    t6 = time.time()
    # ============分类处理目标区域
    if category == '圆形':
        res = circle_ocr(img, img_bw, cfg=cfg)  # 使用二值图预测
    elif category == "正方形":
        res = rectangle_ocr(img, img_bw, cfg=cfg)
    else:
        res = ellipse_ocr(img, img_bw, cfg=cfg)
    t7 = time.time()
    detail = f"{cfg['id']} done " \
             f"kmeans+redfilter:{round(t2 - t0, 2)}+" \
             f"fit:{round(t5 - t4, 2)}+" \
             f"ocr:{round(t7 - t6, 2)}=" \
             f"total{round(t7 - t0, 2)}s. " \
             f"class={det['class']}"
    print(detail)
    logging.info(detail)
    return category, res

def one_pic(file_, to_path_, opt_):
    fn = file_.split("/")[-1]
    if opt_["debug"]:
        to_path_ = os.path.join("debug", fn.split(".")[0])
        if not os.path.exists(to_path_):
            os.mkdir(to_path_)
    assert os.path.exists(to_path_)
    cfg = {
        "id": fn.split(".")[0],
        "img_path": file_,
        "to_path": to_path_,
        "debug": opt_["debug"]
    }

    cls, res = work(cfg)
    if not opt_["debug"]:
        with open(os.path.join(cfg["to_path"], fn.split("/")[-1]).replace(".jpg", ".json"), mode="w",
                  encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
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
    logging.basicConfig(filename=f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.log",
                        level=logging.INFO,
                        format=LOG_FORMAT)

    text = None
    for file in file_list:
        text = one_pic(file_=file, to_path_=to_path, opt_=opt)
        '''
        try:
           text = one_pic(file_=file, to_path_=to_path, opt_=opt)
        except Exception:
           logging.error(f"error occurred in {file}")
        '''
    logging.info(f"all completed {time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
    return json.dumps(text, ensure_ascii=False)


if __name__ == '__main__':
    # 参数设置
    opt = {
        "source": r'C:/Users/a8275/Desktop/project/rd/4101035073190.jpg' ,
        "debug": False  # debug模式将可视化各环节，否则只输出结果
    }
    print(main(opt))