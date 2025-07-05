import cv2
import mediapipe as mp
import util
import pyautogui
from pynput.mouse import Button, Controller
import random
import keyboard
import time
import sounddevice as sd
import numpy as np
from collections import deque

pyautogui.FAILSAFE = False
last_action_time = 0
COOLDOWN = 1.0 
prev_screenshot_state = None 
prev_pointer_x, prev_pointer_y = None, None
MIN_MOVEMENT_THRESHOLD = 0.01

gesture_enabled = False
last_toggle_time = 0
TOGGLE_COOLDOWN = 5
prev_fists_closed = False
   
dragging = False

mouse = Controller()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode = False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)
draw = mp.solutions.drawing_utils

def both_hands_fist(multi_landmarks):
    if len(multi_landmarks) != 2:
        return False
    hands = []
    for handLms in multi_landmarks:
        coords = [(lm.x, lm.y) for lm in handLms.landmark]
        hands.append(coords)
    return is_hand_closed(hands[0]) and is_hand_closed(hands[1])

def update_last_action():
    global last_action_time
    last_action_time = time.time()

def normalize(val, min_in, max_in):
    val = max(0.0, min(val, 1.0))
    if max_in - min_in == 0:
        return 0.5
    val = max(min(val, max_in), min_in) 
    return (val - min_in) / (max_in - min_in)

def is_pinch(landmark_lst):
    if len(landmark_lst) < 9:
        return False

    thumb_tip = landmark_lst[4]
    index_tip = landmark_lst[8]
    dist = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

    return dist < 0.03  

def is_thumb_up(landmark_lst):
    if len(landmark_lst) < 21:
        return False

    thumb_tip_y = landmark_lst[4][1]
    thumb_ip_y = landmark_lst[3][1]
    thumb_mcp_y = landmark_lst[2][1]
    wrist_y = landmark_lst[0][1]
    fingers_folded = all(
        landmark_lst[tip][1] > landmark_lst[base][1]
        for tip, base in [(8, 5), (12, 9), (16, 13), (20, 17)]
    )
    thumb_extended = thumb_tip_y < thumb_ip_y < thumb_mcp_y < wrist_y

    return fingers_folded and thumb_extended

def is_thumb_down(landmark_lst):
    if len(landmark_lst) < 21:
        return False

    thumb_tip_y = landmark_lst[4][1]
    thumb_ip_y = landmark_lst[3][1]
    thumb_mcp_y = landmark_lst[2][1]
    wrist_y = landmark_lst[0][1]
    fingers_folded = all(
        abs(landmark_lst[tip][1] - wrist_y) < 0.1
        for tip in [8, 12, 16, 20]
    )
    thumb_down = thumb_tip_y > thumb_ip_y > thumb_mcp_y > wrist_y

    return fingers_folded and thumb_down

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(finger_tip, frame):
    global prev_pointer_x, prev_pointer_y
    if finger_tip is None:
        return
    screen_width, screen_height = pyautogui.size()
    x = finger_tip.x
    y = finger_tip.y

    h, w, _ = frame.shape
    cv2.circle(frame, (int(x * w), int(y * h)), 15, (0, 255, 255), 3)

    if prev_pointer_x is not None and prev_pointer_y is not None:
        dx = abs(x - prev_pointer_x)
        dy = abs(y - prev_pointer_y)
        if dx < MIN_MOVEMENT_THRESHOLD and dy < MIN_MOVEMENT_THRESHOLD:
            return  
        
    pyautogui.moveTo(int(x * screen_width), int(y * screen_height))
    prev_pointer_x, prev_pointer_y = x, y

def is_hand_open(landmark_lst):
    return all(util.get_angle(landmark_lst[i], landmark_lst[i+1], landmark_lst[i+3]) > 150
               for i in [5, 9, 13, 17])

def is_hand_closed(landmark_lst):
    return all(util.get_angle(landmark_lst[i], landmark_lst[i+1], landmark_lst[i+3]) < 90
               for i in [5, 9, 13, 17])

def is_left_click(landmark_lst, thumb_index_dist):
    return (util.get_angle(landmark_lst[5], landmark_lst[6], landmark_lst[8])<90 and 
            thumb_index_dist>80 and
            util.get_angle(landmark_lst[9], landmark_lst[10], landmark_lst[12])>90)

def is_right_click(landmark_lst, thumb_index_dist):
    return (util.get_angle(landmark_lst[5], landmark_lst[6], landmark_lst[8])>90 and
            thumb_index_dist>80 and
            util.get_angle(landmark_lst[9], landmark_lst[10], landmark_lst[12])<90)

def is_double_click(landmark_lst, thumb_index_dist):
    return (util.get_angle(landmark_lst[5], landmark_lst[6], landmark_lst[8])<90 and
            thumb_index_dist>80 and
            util.get_angle(landmark_lst[9], landmark_lst[10], landmark_lst[12])<90)

def is_screenshot(landmark_lst):
    global prev_screenshot_state
    if len(landmark_lst) >= 21 and is_hand_open(landmark_lst):
        prev_screenshot_state = 'open'
    elif is_hand_closed(landmark_lst):
        if prev_screenshot_state == 'open':
            label = random.randint(1,1000)
            pyautogui.screenshot(f'my_screenshot_{label}.png')
            update_last_action()
        prev_screenshot_state = 'closed'

def detect_gestures(frame, landmark_lst, processed):
    global dragging

    index_finger_tip = find_finger_tip(processed)
    if index_finger_tip is not None:
        try:
            if len(landmark_lst) >= 9:
                if all(i < len(landmark_lst) for i in [4, 5, 6, 8]):
                    thumb_index_dist = util.get_distance([landmark_lst[4], landmark_lst[5]])
                    angle = util.get_angle(landmark_lst[5], landmark_lst[6], landmark_lst[8])
                    if thumb_index_dist < 50 and angle > 90:
                        move_mouse(index_finger_tip, frame)
                else:
                    move_mouse(index_finger_tip, frame)
            else:
                move_mouse(index_finger_tip, frame)
        except:
            move_mouse(index_finger_tip, frame)

    if (len(landmark_lst)<21):
        return

    #Drag and Drop
    if is_pinch(landmark_lst):
        if not dragging:
            mouse.press(Button.left)
            dragging = True
        cv2.putText(frame, "Pinching", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,200), 2)
    else:
        if dragging:
            cv2.putText(frame, "Dragging", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,200), 2)
            mouse.release(Button.left)
            dragging = False
    
    #Volume control
    if is_thumb_up(landmark_lst):
        pyautogui.press("volumeup")
        cv2.putText(frame, "Volume up", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,200), 2)
        return
    elif is_thumb_down(landmark_lst):
        pyautogui.press("volumedown")
        cv2.putText(frame, "Volume Down", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,200), 2)
        return

    thumb_index_dist = util.get_distance([landmark_lst[4], landmark_lst[5]])

    if keyboard.is_pressed('esc'):
        return 

    #left click
    if is_left_click(landmark_lst, thumb_index_dist):
        mouse.press(Button.left)
        mouse.release(Button.left)
        cv2.putText(frame, "Left Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        update_last_action()

    #right click
    elif is_right_click(landmark_lst, thumb_index_dist):
        mouse.press(Button.right)
        mouse.release(Button.right)
        cv2.putText(frame, "Right Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        update_last_action()

    #double click
    elif is_double_click(landmark_lst, thumb_index_dist):
        pyautogui.doubleClick()
        cv2.putText(frame, "Double Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        update_last_action()

    #screenshot
    elif is_screenshot(landmark_lst):
        cv2.putText(frame, "Screenshot", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        update_last_action()

def capture():
    cap = cv2.VideoCapture(0)
    global gesture_enabled, last_toggle_time, prev_fists_closed
    while (True):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        landmarks_lst = []
        hands_lms = processed.multi_hand_landmarks

        if gesture_enabled and hands_lms:
            hand_landmarks = hands_lms[0]
            draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            landmarks_lst = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            detect_gestures(frame, landmarks_lst, processed)
        else:
            cv2.putText(frame, "Detection OFF", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #Gesture enable/disable
        if not hands_lms or len(hands_lms) != 2:
            prev_fists_closed = False
        else:
            now = time.time()
            fists_closed = both_hands_fist(hands_lms)

            if fists_closed and not prev_fists_closed and (now - last_toggle_time) > TOGGLE_COOLDOWN:
                gesture_enabled = not gesture_enabled
                last_toggle_time = now
                print("✊✊ Both fists! Gesture:", "ENABLED" if gesture_enabled else "DISABLED")
                prev_fists_closed = fists_closed
            else:
                prev_fists_closed = False

        #Scroll
        if hands_lms and len(hands_lms)>0:
            lm = hands_lms[0].landmark
            coords = [(pt.x, pt.y) for pt in lm]
            if is_hand_open(coords):
                wrist_y = lm[mpHands.HandLandmark.WRIST].y
                screen_h = pyautogui.size().height
                if wrist_y < 0.15:
                    pyautogui.scroll(100)
                    cv2.putText(frame, "Scroll Up", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,200), 2)
                elif wrist_y>0.85:
                    pyautogui.scroll(-100)
                    cv2.putText(frame, "Scroll Down", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,200), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()