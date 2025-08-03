---

## 📚 Summary: Seaborn + AI/ML Hands-on Practice

---

### 🔷 Seaborn — Data Visualization with Python

**What is Seaborn?**
Seaborn is a Python library based on Matplotlib. It makes beautiful, statistical visualizations simple and elegant.(Power BI in 144p)

#### 🔹 Sample Dataset

We used the built-in `tips` dataset:

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
print(tips.head())
```

#### 🔹 Visualizations

```python
# Scatter Plot
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
plt.title("Total Bill vs Tip by Gender")
plt.show()

# Histogram
sns.histplot(data=tips, x="total_bill", bins=20, kde=True)
plt.title("Total Bill Distribution")
plt.show()

# Box Plot
sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")
plt.title("Box Plot of Bills by Day and Gender")
plt.show()

# Bar Plot
sns.barplot(data=tips, x="day", y="tip", estimator=sum)
plt.title("Total Tips by Day")
plt.show()

# Heatmap
sns.heatmap(tips.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

#### ✅ What we learned:

* `load_dataset()` to use built-in datasets.
* How to visualize categories, relationships, and distributions.
* Heatmaps show correlation between numeric fields.

---

### 🤖 AI / ML Hands-on: Virtual Mouse with Mediapipe + PyAutoGUI

#### ✅ What we built:

A hand-tracking **virtual mouse** using:

* `mediapipe` (for hand tracking)
* `pyautogui` (to move/click the cursor)
* `opencv-python` (for webcam and drawing)

#### 📦 Requirements:

```bash
pip install mediapipe opencv-python pyautogui
```

#### 💻 Final Code:

```python
import cv2
import mediapipe as mp
import pyautogui

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
clicking = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_img)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8]
            thumb_finger = landmarks[4]

            x = int(index_finger.x * img.shape[1])
            y = int(index_finger.y * img.shape[0])
            screen_x = screen_width * index_finger.x
            screen_y = screen_height * index_finger.y

            pyautogui.moveTo(screen_x, screen_y)

            distance = ((index_finger.x - thumb_finger.x) ** 2 + (index_finger.y - thumb_finger.y) ** 2) ** 0.5

            if distance < 0.02:
                if not clicking:
                    clicking = True
                    pyautogui.click()
            else:
                clicking = False

            cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 🧠 Concepts Learned

* `mediapipe` hand landmarks: index finger tip (8), thumb tip (4)
* `pyautogui.moveTo()` to control mouse
* Using distance between fingers to detect a **pinch-click**
* OpenCV: real-time webcam feed and drawing overlays

---



