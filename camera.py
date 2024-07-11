'''
0 - Unrecognized gesture, label: Unknown
1 - Closed fist, label: Closed_Fist -> right click
2 - Open palm, label: Open_Palm -> left click
3 - Pointing up, label: Pointing_Up -> move the mouse
4 - Thumbs down, label: Thumb_Down -> scroll down
5 - Thumbs up, label: Thumb_Up -> scroll up
6 - Victory, label: Victory
7 - Love, label: ILoveYou
'''
import cv2
import mediapipe as mp
import pyautogui

model_path = './gesture_recognizer.task'

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Create the task
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

x_mouse, y_mouse = 0, 0

# Callback function to print the result
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        for gesture in result.gestures[0]:  # Accessing the first set of gestures
            print(f'Gesture: {gesture.category_name}, Confidence: {gesture.score}')
            # Do something with the mouse right here
            if gesture.category_name == 'Pointing_Up':
                print(f'Move the mouse {x_mouse}, {y_mouse}')
                pyautogui.moveTo(x_mouse, y_mouse)
            elif gesture.category_name == 'Closed_Fist':
                pyautogui.rightClick()
            elif gesture.category_name == 'Open_Palm':
                pyautogui.leftClick()
            elif gesture.category_name == 'Thumb_Down':
                pyautogui.scroll(-1)
            elif gesture.category_name == 'Thumb_Up':
                pyautogui.scroll(1)
            else:
                print('No mouse control.')
    else:
        print('No gesture recognized.')

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Initialize the gesture recognizer
recognizer = GestureRecognizer.create_from_options(options)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Define the camera
camera = cv2.VideoCapture(0)

# Set camera frame size to match screen dimensions
camera.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

timestamp_ms = 0  # Initialize timestamp

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    # Flip the frame to avoid mirror image
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare the data by converting the frame from OpenCV to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

    # Process the image with MediaPipe Hands
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the bounding box of the hand
            x_min = int(hand_landmarks.landmark[0].x * frame.shape[1])
            y_min = int(hand_landmarks.landmark[0].y * frame.shape[0])
            x_max, y_max = x_min, y_min
            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 8:
                    print(id, lm)
                    x_mouse, y_mouse = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])

            # Draw the bounding box around the detected hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Increment timestamp for each frame
            timestamp_ms += 1

            # Predict the gesture
            recognizer.recognize_async(mp_image, timestamp_ms)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Type 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
