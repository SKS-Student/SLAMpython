import cv2
import numpy as np


#Use first camera on device (if its a laptop its the laptop camera not any external one)
camera = cv2.VideoCapture(0)


print("Press 'q' to quit")

while True:
    
    #Capture Frame by Frame
    ret,frame = camera.read()
    h2, w2 = frame.shape[:2]

    white_frame = np.ones((h2, w2, 3), dtype=np.uint8) * 255
    


    #Grayscale the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    



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

        # Draw matches
        match_img = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches[:200], None, flags=2)


        cv2.imshow("Feature Matches", match_img)

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

    
    # Draw keypoints on the white image
    keypoints_img = cv2.drawKeypoints(
        white_frame, kp, None,
        color=(0, 0, 255),  # red keypoints
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    

    # Display the resulting frame
    cv2.imshow('White Feed', white_frame)
    cv2.imshow("Camera", frame)
    cv2.imshow("Keypoints", keypoints_img)
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