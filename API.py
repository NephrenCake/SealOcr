from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from utils import Model
from main import main


#model = Model.Model(2)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
@app.route('/upload')
def upload_file():
    return render_template('fileupload.html')

@app.route('/uploader',methods=['GET','POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
        text = main({"source": path, "debug": False})
        if (os.path.isfile(path)):
            os.remove(path)
        else:
            pass
        return text
    else:
        return render_template('upload.html')

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)