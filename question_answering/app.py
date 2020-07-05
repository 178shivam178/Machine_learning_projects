from flask import request,render_template
import flask
import os
from answering import predict

app = flask.Flask(__name__,template_folder='tamplates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def get_data():
    context=request.form['Paragraph']
    question=request.form['Question1']
    answer=predict(context,question)
    return render_template('index.html',answer=answer)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=int(os.environ.get('PORT', 5000)))