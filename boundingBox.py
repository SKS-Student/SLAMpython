import cv2

# Use first camera on device
camera = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Can't receive frame (stream ended?). Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.blur(gray, (3,3))

    # Apply Canny edge detection
    edge = cv2.Canny(gray, 100, 150)

    # Find contours from the edge image
    contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding rectangles around all contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1500:
            if w > 400:  # Optional: Filter out very small boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert edges to white for better display
    edgesWhite = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    edgesWhite[edge > 0] = [255, 255, 255]

    # Display the results
    cv2.imshow("Camera", frame)
    cv2.imshow("CameraGray", gray)
    cv2.imshow("Gray Edges Live Feed", edge)
    cv2.imshow("White Edges Live Feed", edgesWhite)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting")
        break

camera.release()
cv2.destroyAllWindows()
