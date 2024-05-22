import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt



# different mediapipe objects for displaying/drawing handlandmarks
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles


# model for detecting hand landmarks
hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

DATA_DIR='./data'
# DATA_DIR is the main data directory, dir_ refers to the subdirectories/classes


for dir_ in os.listdir(DATA_DIR):
    print('Displaying for class:',dir_)
    img_name=os.listdir(os.path.join(DATA_DIR, dir_))[0]
    img_path=os.path.join(DATA_DIR,dir_,img_name)
    #print(img_path)
    img=cv2.imread(img_path)
    
    # cv2 reads image in BGR so convert to RGB
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # results stores all the image hand landmarks
    results=hands.process(img_rgb)
        
    # now iterating through results, if multiple landmarks found then...
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
        
        #Drawing hand landmarks on the image
            mp_drawing.draw_landmarks(
                    img_rgb, #image to draw
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS, #hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                    )
                
    plt.figure()
    plt.axis('off')
    plt.imshow(img_rgb)