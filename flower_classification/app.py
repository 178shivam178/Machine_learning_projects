from flask import Flask
from flask import render_template,request
import os

app=Flask(__name__)
UPLOAD_FOLDER=r"E:\personal_project\flower_classification\static"
@app.route('/',methods=['GET','POST'])
def upload_predict():
    if request.method=="POST":
        image_file=request.files.get("image",None)
        if image_file:
            image_location=os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
                )
            image_file.save(image_location)
            return render_template("index.html",prediction=1,image_loc=image_file.filename)
    return render_template("index.html",prediction=0,image_loc=None)

if __name__=='__main__':
    app.run(port=12000,debug=True)