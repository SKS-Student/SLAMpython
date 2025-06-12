import cv2
import numpy as np


#Use first camera on device (if its a laptop its the laptop camera not any external one)
camera = cv2.VideoCapture(0)


print("Press 'q' to quit")

while True:
    
    #Capture Frame by Frame
    ret,frame = camera.read()
     
     
    #Grayscale the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Apply Canny through grayscale lower threshold is 100 meaning any intensity/gradient change below 100 is not considered and anything above 200 is considered strong edge
    edge = cv2.Canny(gray, 100, 200)


    #Converts the grayscaled image that edge is providing into RGB
    edgesWhite = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    #Every edge detected will be colored white
    edgesWhite[edge > 0] = [255, 255, 255]


    orb = cv2.ORB_create(2000)
    ret, prev_frame = camera.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Detect and compute keypoints/descriptors
    kp, des = orb.detectAndCompute(gray, None)

    if des is not None and prev_des is not None:
        # Match features between current and previous frame
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches normally
        match_img = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches[:50], None, flags=2)

        # Create a blank image (black) of the same size as the side-by-side match image
        h1, w1 = prev_frame.shape[:2]
        h2, w2 = frame.shape[:2]
        blank = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        cv2.imshow("Mith", blank)

        # Draw matches on the blank image
        match_img_black = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches[:100], blank, flags=2)
        cv2.imshow("Feature Matches", match_img_black)

    else:
        print("Couldnt show one")

    # Update previous frame and keypoints
    prev_frame = frame.copy()
    prev_gray = gray.copy()
    prev_kp = kp
    prev_des = des



    #If frame not read correctly break
    if not ret:
        print("Can't receive frame (stream ended?). Exiting...")
        break

    # Display the resulting frame
    cv2.imshow("Camera", frame)
    cv2.imshow("CameraGray", gray)
    cv2.imshow('Gray Edges Live Feed', edge)
    cv2.imshow('White Edges Live Feed', edgesWhite)
    
    #Waits for a event every 1 millisecond and takes only the last 8 bits of the key code for compatibility (different devices might have different lengths of key code bits)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit
    if (key == ord("q")):
        print("Exiting")
        break


#Stops using the camera
camera.release() 
#Close all windows of the camera shown
cv2.destroyAllWindows()    