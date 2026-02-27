import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from time import time

# Load dataset
data = pd.read_csv("colors_super_final.csv")
X = data[['R', 'G', 'B']]
y = data['color_name']

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X, y)

# Precompute for fast closest match
rgb_values = data[['R', 'G', 'B']].values
names = data['color_name'].values

def get_closest_color(r, g, b):
    target = np.array([r, g, b])
    dists = np.sqrt(((rgb_values - target) ** 2).sum(axis=1))
    return names[np.argmin(dists)]

# Variables
clicked = False
r = g = b = xpos = ypos = 0
color_name = ""
last_click_time = 0
frame = None

# Mouse callback
def draw_function(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked, color_name, last_click_time, frame

    if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
        clicked = True
        xpos, ypos = x, y

        # Take average color around clicked point (reduces noise)
        h, w = frame.shape[:2]
        x1, x2 = max(0, x-3), min(w, x+4)
        y1, y2 = max(0, y-3), min(h, y+4)

        roi = frame[y1:y2, x1:x2]
        b_avg, g_avg, r_avg = roi.mean(axis=(0, 1))
        b, g, r = int(b_avg), int(g_avg), int(r_avg)

        # Use closest color method (more stable for webcam)
        color_name = get_closest_color(r, g, b)

        last_click_time = time()

# Start webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Color Recognition')
cv2.setMouseCallback('Color Recognition', draw_function)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    if clicked and (time() - last_click_time) <= 5:
        cv2.rectangle(frame, (20, 20), (600, 60), (b, g, r), -1)
        text = f"{color_name}  RGB=({r},{g},{b})"
        text_color = (255, 255, 255) if (r + g + b) < 400 else (0, 0, 0)
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    elif (time() - last_click_time) > 5:
        clicked = False

    cv2.imshow("Color Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
