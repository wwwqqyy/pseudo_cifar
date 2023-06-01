from flask import Flask, render_template, request
from label import get_label

app = Flask(__name__)


@app.route('/')
def main():  # put application's code here
    return render_template('label.html')


@app.route("/imageLabel", methods=["POST"])
def img_label():
    image = request.files["file"]
    img_bytes = image.read()
    img_stream = get_label(image_bytes=img_bytes)
    return img_stream


if __name__ == '__main__':
    app.run()
