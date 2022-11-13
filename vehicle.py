from json import detect_encoding
import cv2
import numpy as np

#Initializing substructor   

algo=cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy
    
detect = []
offset=6
counter=0

#Line implementation
count_line_position= 550

#Web Camera
cap = cv2.VideoCapture('video.mp4')

#width and height of Box
min_width_react=80
min_height_react=80

while (cap.isOpened()):
    ret,frame= cap.read()
    grey= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(3,3),5)
#Applying on this on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada= cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    countershape,h= cv2.findContours(dilatada,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0, 0, 255),1)

    for (i,c) in enumerate(countershape):
        (x,y,w,h)= cv2.boundingRect(c)
        validate_counter = (w>= min_width_react) and (h>= min_height_react)
        if not validate_counter:    
            continue

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
 
        center= center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,4, (0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
            cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0, 0, 255),1)
            detect.remove((x,y))
            print("Vehicle Counter:"+str(counter))

    cv2.putText(frame,"Vehicle Counter:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)                

        


    frame=cv2.resize(frame,(1200,700))
    cv2.imshow('video.mp4',frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
    
        break


  
cap.release()
cv2.destroyAllWindows()

    