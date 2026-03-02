import cv2
import mediapipe as mp
import pygame
import numpy as np

# ---------------- AUDIO ----------------
pygame.mixer.pre_init(44100, -16, 2, 256)
pygame.init()

notes = {
    "C": pygame.mixer.Sound("sounds/c.wav"),
    "D": pygame.mixer.Sound("sounds/d.wav"),
    "E": pygame.mixer.Sound("sounds/e.wav"),
    "F": pygame.mixer.Sound("sounds/f.wav"),
    "G": pygame.mixer.Sound("sounds/g.wav"),
}

# --------------- MEDIAPIPE --------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]
finger_pips = [3, 6, 10, 14, 18]

finger_note_map = {
    0: "C",
    1: "D",
    2: "E",
    3: "F",
    4: "G"
}

previous_state = [0, 0, 0, 0, 0]

# --------------- MAIN LOOP --------------
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_state = [0, 0, 0, 0, 0]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --------- THUMB FIX ----------
            thumb_tip_x = hand_landmarks.landmark[4].x
            thumb_ip_x = hand_landmarks.landmark[3].x

            if thumb_tip_x > thumb_ip_x:
                current_state[0] = 1

            # -------- OTHER FINGERS -------
            for i in range(1, 5):
                tip_y = hand_landmarks.landmark[finger_tips[i]].y
                pip_y = hand_landmarks.landmark[finger_pips[i]].y

                if tip_y < pip_y:
                    current_state[i] = 1

            # -------- PLAY NOTES ----------
            for i in range(5):
                if current_state[i] == 1 and previous_state[i] == 0:
                    note = finger_note_map[i]
                    notes[note].play()

            previous_state = current_state

    # --------------- DRAW PIANO UI ---------------
    key_width = w // 5
    key_height = 120
    start_y = h - key_height

    note_names = ["C", "D", "E", "F", "G"]

    for i in range(5):
        x1 = i * key_width
        x2 = (i + 1) * key_width

        if current_state[i] == 1:
            color = (0, 255, 0)  # Green when pressed
        else:
            color = (255, 255, 255)

        cv2.rectangle(img, (x1, start_y), (x2, h), color, -1)
        cv2.rectangle(img, (x1, start_y), (x2, h), (0, 0, 0), 2)

        cv2.putText(img, note_names[i],
                    (x1 + key_width//3, start_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2)

    cv2.imshow("AI Air Piano - Pro Version", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()