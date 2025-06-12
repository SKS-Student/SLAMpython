import cv2

#Use first camera on device (if its a laptop its the laptop camera not any external one)
camera = cv2.VideoCapture(0) 


print("Press 'q' to quit")

while True:
    
    #Capture Frame by Frame
    ret,frame = camera.read()

    #If frame not read correctly break
    if not ret:
        print("Can't receive frame (stream ended?). Exiting...")
        break

    # Display the resulting frame
    cv2.imshow("Camera", frame)

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