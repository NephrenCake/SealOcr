# -- coding: utf-8 --
import json


def cal_pre_rec():
    cls = "rectangle"
    # pre: dict = read_json(f"validation/rotate_cut/{cls}/text_result.json")
    pre: dict = read_json(f"text_result.json")
    lab: dict = read_json(f"validation/rotate_cut/{cls}/text_label.json")

    total_lab = 0
    right_in_label = 0
    total_pre = 0
    right_in_predict = 0
    right_pic = 0

    errors = []
    for key in lab.keys():
        total_lab += len(lab[key])
        total_pre += len(pre[key])

        correct_texts = []  # 应该但没有被检测到的文本
        wrong_texts = []  # 不应该被却被检测到的文本
        for text in pre[key]:
            if text in lab[key]:
                right_in_label += 1
            else:
                wrong_texts.append(text)
        for text in lab[key]:
            if text in pre[key]:
                right_in_predict += 1
            else:
                correct_texts.append(text)
        if len(correct_texts) == 0 and len(wrong_texts) == 0:
            right_pic += 1
        else:
            errors.append({
                key: {
                    "correct_texts": correct_texts,
                    "wrong_texts": wrong_texts
                }
            })

    recall = right_in_predict / total_lab
    precision = right_in_label / total_pre
    print("文本级标注:=============")
    print(f"recall: {recall}")
    print(f"precision: {precision}")
    print("图片级标注:=============")
    print(f"precision: {right_pic / 50}")
    print("未检对文本:=============")
    for item in errors:
        print(item)


def read_json(path):
    with open(path, mode="r", encoding="utf-8") as f:
        res = json.load(f)
    return res


def write_json(content, path):
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False)


if __name__ == '__main__':
    cal_pre_rec()
    pass
