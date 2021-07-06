from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import Model
from main import SealRecog
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
model = Model.Model(2)
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
        text,uid = SealRecog(path,model)
        if (os.path.isfile(path)):
            os.remove(path)
        else:
            pass
        return "文字："+text+"\n编号"+uid
    else:
        return render_template('upload.html')

if __name__ == '__main__':
   app.run(debug=True)