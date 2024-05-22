import os
import cv2

DIR='./data'

newDIR='./flipped_data'

if not os.path.exists(newDIR):
    os.makedirs(newDIR)
    

for subdir in os.listdir(DIR):
    count=0
    new_subdir=os.path.join(newDIR,subdir)
    if not os.path.exists(new_subdir):
        os.makedirs(new_subdir)
    print('Entering subdir:',subdir)

    
    for img_name in os.listdir(os.path.join(DIR,subdir)):
        img=cv2.imread(os.path.join(DIR,subdir,img_name))
        img=cv2.flip(img, 1)
        out_name=str(count)+'.jpg'
        cv2.imwrite(os.path.join(new_subdir,out_name),img)
        count+=1
    
print('done')
    
