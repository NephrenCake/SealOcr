# -- coding: utf-8 --
import time
from utils.ocr import ellipse_ocr, circle_ocr, rectangle_ocr
from utils.detect import *
from utils.Model import Model
import logging

model = Model(0)


def Precisefilter(img, cfg):
    img_ = filter_non_red(img, cfg=cfg)
    img_ = k_means(img_, cfg=cfg)
    img_bw = cv2.merge([img_, img_, img_])
    return img_, img_bw


def Lazyfilter(img, cfg):
    img_ = Differ_kmeans(img, cfg)
    img_bw = cv2.merge([img_, img_, img_])
    return img_, img_bw


def work(cfg):
    t0 = time.time()
    # 分类
    category, img = model.predict(cfg['img_path'])
    img = enlarge_img(img)
    img_ = img.copy()
    # 提取红色
    t1 = time.time()
    img_, img_bw = Lazyfilter(img_, cfg)
    # 开运算去噪填充
    t2 = time.time()
    img_ = erode_dilate(img_, category, cfg=cfg)
    # 查找最大轮廓
    t3 = time.time()
    contours, max_idx = find_max(img_)
    # 完成检测框
    t4 = time.time()
    det = get_area(img, contours, max_idx, category, cfg=cfg)  # 不进行分类
    # 截取目标区域  img:原图 img_bw:二值
    t5 = time.time()
    img = rotate_cut(img, det, cfg=cfg)
    # ============分类处理目标区域
    t6 = time.time()
    result = []
    if category == '圆形':
        res = circle_ocr(img, cfg=cfg)  # 使用二值图预测
    elif category == "正方形":
        res = rectangle_ocr(img, cfg=cfg)
    else:
        res = ellipse_ocr(img, cfg=cfg)
    # 提取文本信息
    for item in res:
        result.append(item["text"])
    result_string = json.dumps(result, ensure_ascii=False)
    if not cfg["debug"]:
        with open(os.path.join(cfg["to_path"], cfg["file_name"] + ".json"), mode="w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
    t7 = time.time()

    detail = f"{cfg['file_name']} done in {round(t7 - t0, 2)}s. " \
             f"class={det['class']}, text={result_string}"
    logging.info(detail)
    return result_string


def one_pic(file_, to_path_, opt_):
    fn = file_.split("/")[-1]
    if opt_["debug"]:
        to_path_ = os.path.join("debug", fn.split(".")[0])
        if not os.path.exists(to_path_):
            os.mkdir(to_path_)
    assert os.path.exists(to_path_)
    cfg = {
        "file_name": fn.split(".")[0],
        "img_path": file_,
        "to_path": to_path_,
        "debug": opt_["debug"]
    }
    res = work(cfg)
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
    logging.info(f"{len(file_list)} images found.")

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

    res = json.dumps([], ensure_ascii=False)
    for file in file_list:
        # res = one_pic(file_=file, to_path_=to_path, opt_=opt)
        try:
           res = one_pic(file_=file, to_path_=to_path, opt_=opt)
        except Exception:
           logging.error(f"error occurred in {file}")
    return res


if __name__ == '__main__':
    # 参数设置
    opt = {
        "source": r'validation/origin/rectangle/4101035072803.jpg',
        "debug": False  # debug模式将可视化各环节，否则只输出结果
    }
    print(main(opt))
