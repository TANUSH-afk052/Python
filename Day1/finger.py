import cv2
import mediapipe as mp

# Setup Mediapipe with hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# camera
cap = cv2.VideoCapture(0)

# Finger tip landmarks (thumb to pinky finger indexing)
finger_tips_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks in list
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            if lm_list:
                # Thumb logic (different axis)
                if lm_list[4][1] > lm_list[3][1]:
                    finger_count += 1

                # Other fingers
                for tip_id in finger_tips_ids[1:]:
                    if lm_list[tip_id][2] < lm_list[tip_id - 2][2]:
                        finger_count += 1

    # Display finger count
    cv2.rectangle(img, (20, 300), (170, 425), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f'Fingers: {finger_count}', (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imshow("Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
