# -- coding: utf-8 --
import re
import time
import traceback

from utils.ocr import ellipse_ocr, circle_ocr, rectangle_ocr
from utils.detect import *
from utils.Model import Model
import logging

model = Model(0)

CODE_SCORE = 0  # 0.98 0.97
NAME_SCORE = 0  # 0.9


def get_result(res, category):
    en_or_num = re.compile(r"[a-zA-Z0-9]", re.I)  # [a-z]|\d
    not_en_nor_num = re.compile(r"[^a-zA-Z0-9]", re.I)  # 非英文数字

    null_text = {"text": "", "confidence": 0}
    result = {
        "code": null_text,  # 印章编号
        "name": null_text,  # 印章名称
        "SealType": "99",  # 印章类型代码
        "strSealType": "其他类型印章"  # 印章类型名称
    }
    seal_type = {
        # "法定名称章": "01",
        "财务专用章": "02",
        # "发票专用章": "03",
        "合同专用章": "04",
        # "法定代表人名章": "05",
        "其他类型印章": "99"
    }

    logging.info(f"<{[str(item['text']) + '|' + str(item['confidence']) for item in res]}>")

    if category == "正方形":
        result["SealType"] = "05"
        result["strSealType"] = "法定代表人名章"
        result["code"] = {
            "text": res[0]["text"], "confidence": res[0]["confidence"]
        } if res[0]["confidence"] >= CODE_SCORE else null_text  # 设定编号
        result["name"] = {
            "text": res[1]["text"], "confidence": res[1]["confidence"]
        } if res[1]["confidence"] >= NAME_SCORE else null_text  # 设定name
    elif category == "椭圆":  # todo 调参
        result["SealType"] = "03"
        result["strSealType"] = "发票专用章"
        for item in res:
            text = item["text"]
            score = item["confidence"]
            from_ = item["from"]
            if len(en_or_num.findall(text)) > len(not_en_nor_num.findall(text)):
                result["code"] = {
                    "text": text, "confidence": score
                } if score >= CODE_SCORE and from_ == "center" else null_text  # 设定编号
            else:
                result["name"] = {
                    "text": text, "confidence": score
                } if score >= NAME_SCORE and from_ == "side" else null_text  # 设定name
    elif len(res) == 2:
        result["SealType"] = "01"
        result["strSealType"] = "法定名称章"
        for item in res:
            text = item["text"]
            score = item["confidence"]
            from_ = item["from"]
            if len(en_or_num.findall(text)) > len(not_en_nor_num.findall(text)):
                result["code"] = {
                    "text": text, "confidence": score
                } if score >= CODE_SCORE and from_ == "side" else null_text  # 设定编号
            else:
                result["name"] = {
                    "text": text, "confidence": score
                } if score >= NAME_SCORE and from_ == "side" else null_text  # 设定name
    else:
        set_type = True
        for item in res:
            text = item["text"]
            score = item["confidence"]
            from_ = item["from"]
            if len(en_or_num.findall(text)) > len(not_en_nor_num.findall(text)):
                result["code"] = {
                    "text": text, "confidence": score
                } if score >= CODE_SCORE and from_ == "side" else null_text  # 设定编号
            else:
                if set_type and from_ == "center":
                    # 设定type
                    if text not in seal_type.keys():
                        result["SealType"] = "99"
                        result["strSealType"] = "其他类型印章"  # "其他类型印章"
                    else:
                        result["SealType"] = seal_type[text]
                        result["strSealType"] = text
                    set_type = False
                else:
                    # 设定name
                    result["name"] = {
                        "text": text, "confidence": score
                    } if score >= NAME_SCORE and from_ == "side" else null_text  # 设定name
    return result


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
    if category == '圆形':
        res = circle_ocr(img, cfg=cfg)  # 使用二值图预测
    elif category == "正方形":
        res = rectangle_ocr(img, cfg=cfg)
    else:
        res = ellipse_ocr(img, cfg=cfg)

    # 将文本列表处理成标准json格式
    result = get_result(res, category)

    result_string = json.dumps(result, ensure_ascii=False)
    if not cfg["debug"]:
        # with open(os.path.join(cfg["to_path"], cfg["file_name"] + ".json"), mode="w", encoding="utf-8") as f:
        #     json.dump(result, f, ensure_ascii=False)
        pass
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
        # to_path = os.path.join("run", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        # if not os.path.exists(to_path):
        #     os.mkdir(to_path)

    res = json.dumps([], ensure_ascii=False)
    for file in file_list:
        # res = one_pic(file_=file, to_path_=to_path, opt_=opt)
        try:
            res = one_pic(file_=file, to_path_=to_path, opt_=opt)
        except Exception as e:
            logging.error(f"error occurred in {file}: {e}\n"
                          f"{traceback.format_exc()}")
            traceback.print_exc()
    return res


if __name__ == '__main__':
    # 参数设置
    opt = {
        "source": r'validation/origin/rectangle/4101035072803.jpg',
        "debug": False  # debug模式将可视化各环节，否则只输出结果
    }
    print(main(opt))
