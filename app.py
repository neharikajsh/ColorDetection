from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
cap = cv2.VideoCapture(0)

colors = {
    "red": [0, 0, 255],
    "green": [0, 255, 0],
    "blue": [255, 0, 0],
    "yellow": [0, 255, 255],
    "cyan": [255, 255, 0],
    "magenta": [255, 0, 255],
    "orange": [0, 128, 255],
    "pink": [255, 192, 203],
    "purple": [128, 0, 128],
    "brown": [42, 42, 165],
    "gray": [128, 128, 128]
}

selected_color = [0, 255, 255]  # Default to yellow

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lowerLimit, upperLimit = get_limits(color=selected_color)
            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', colors=colors)

@app.route('/select_color/<color>')
def select_color(color):
    global selected_color
    if color in colors:
        selected_color = colors[color]
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
