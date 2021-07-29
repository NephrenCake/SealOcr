from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from utils import Model
from main import main

# model = Model.Model(2)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
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
