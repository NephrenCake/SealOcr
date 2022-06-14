# 印章识别

1. 传统cv定位
2. deeplearning分类
3. 传统cv纠正印章偏转角度
4. paddleocr识别印章文字（需要本地部署服务和模型）

> 以后闲下来会打算用 rotate-yolo5 重构

```mermaid
graph TB
	印章分类-->圆形印章检测
	印章分类-->方形印章检测
	印章分类-->椭圆形印章检测
```

```bash
# window
set FLASK_APP=API.py
# linux
export FLASK_APP=API.py

flask run -h 0.0.0.0 -p 8000
```
