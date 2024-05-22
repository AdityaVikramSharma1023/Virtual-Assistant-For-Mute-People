import os
import cv2

# first step is to collect images of hand
#and store it in folder say "data"

# if such directory does not exist then create one
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Handsigns- A,B,C,D,E,F,.....Z
number_of_classes = 26
dataset_size = 300 #for each class

# select camera index. By default main camera index is 0, but may vary
cap = cv2.VideoCapture(0)
# cap is a videocapture object

#coordinates for creating rectangle
start=[20,80]
end=[300,380]
color=(255, 0, 127) #in BGR format

# creating subdirectory for each class
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        # waiting frame
        ret, frame = cap.read()
        # mirror the frame vertically
        frame=cv2.flip(frame,1)
        
        # show text on the frame
        cv2.putText(frame,
                    'Press "Q" to Start!!',
                    (100, 50), #text pos
                    cv2.FONT_HERSHEY_DUPLEX, #font
                    1.3, #fontscale
                    color, #rgb color
                    2, #thickness
                    cv2.LINE_AA)
        
        # Display rectangle
        frame = cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color, 5)
      
        # Naming window
        cv2.namedWindow('Window', cv2.WINDOW_NORMAL) 
      
        # Resizing window
        cv2.resizeWindow('Window', 500, 500)
        
        # Moving window
        cv2.moveWindow('Window', 20, 20) 
      
        # Displaying the frame 
        cv2.imshow('Window', frame)
        
        if cv2.waitKey(25) == ord('q'):
            break
        
    counter = 0
    #stop if 300 imgs captured for each class
    while counter < dataset_size:
        # reading frame
        ret, frame = cap.read()
        # mirror the frame vertically
        frame=cv2.flip(frame,1)
        
        #cropping frame as preview
        cut_frame=frame[start[1]:end[1],start[0]:end[0]]
        
        # Naming window
        cv2.namedWindow('Preview', cv2.WINDOW_NORMAL) 
      
        # Moving window
        cv2.moveWindow('Preview', 750, 20) 
        
        cv2.imshow('Preview', cut_frame)
        
        # Displaying rectangle
        frame = cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color, 5)   
        
        cv2.imshow('Window', frame)
        
        # img captured every 100 msecs
        if cv2.waitKey(100) == ord('e'):
            break
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), cut_frame)

        counter += 1
    
    cv2.destroyWindow('Preview')
    
cap.release()
cv2.destroyAllWindows()