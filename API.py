# -- coding: utf-8 --
import logging
import time

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from main import main

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])


def get_logger():
    if not os.path.exists("log"):
        os.mkdir("log")
    logger = logging.getLogger()
    fh = logging.FileHandler(filename=f"log/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.log",
                             encoding="utf-8", mode="a")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s : %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        # print(request.files)
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(path)
        text = main({"source": path, "debug": False})
        if os.path.isfile(path):
            os.remove(path)
        return text

    return render_template('fileupload.html')


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)
