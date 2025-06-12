import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use first camera
camera = cv2.VideoCapture(0)

# Camera intrinsics (adjust to your camera if needed)
K = np.array([[700, 0, 320],
              [0, 700, 240],
              [0,   0,   1]])

# ORB and matcher setup
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Read first frame
ret, prev_frame = camera.read()
if not ret:
    print("Cannot read from camera.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

# For 3D rendering
all_points_3d = []

print("Press 'q' to quit")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    h2, w2 = frame.shape[:2]
    white_frame = np.ones((h2, w2, 3), dtype=np.uint8) * 255

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    if des is not None and prev_des is not None:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) >= 8:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate Essential Matrix and pose
            E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

                # Triangulate and store 3D points
                proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                proj2 = np.hstack((R, t))
                pts1_norm = cv2.undistortPoints(pts1, K, None)
                pts2_norm = cv2.undistortPoints(pts2, K, None)
                points_4d = cv2.triangulatePoints(proj1, proj2, pts1_norm, pts2_norm)
                points_3d = points_4d[:3] / points_4d[3]
                all_points_3d.append(points_3d.T)

        # Draw matches
        match_img = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches[:200], None, flags=2)
        cv2.imshow("Feature Matches", match_img)

    else:
        print("Descriptors not available")

    # Update previous data
    prev_frame = frame.copy()
    prev_gray = gray.copy()
    prev_kp = kp
    prev_des = des

    # Draw keypoints on white screen
    keypoints_img = cv2.drawKeypoints(
        white_frame, kp, None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    cv2.imshow('White Feed', white_frame)
    cv2.imshow("Camera", frame)
    cv2.imshow("Keypoints", keypoints_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting")
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()

# === Render 3D map after exiting ===
if all_points_3d:
    points = np.vstack(all_points_3d)

    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=1)
    ax.set_title("3D Map of Space")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.view_init(elev=135, azim=-90)
    plt.show()
