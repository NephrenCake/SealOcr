# 印章识别

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
