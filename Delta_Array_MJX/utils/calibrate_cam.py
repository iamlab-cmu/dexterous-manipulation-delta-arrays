import cv2
import numpy as np
import glob

# Set your chessboard dimensions (number of inside corners)
# For a 9x6 board, you have 9 corners horizontally and 6 corners vertically
# But check how you printed it. The "inside corners" is usually 1 less
# than the number of squares in each dimension.
CHECKERBOARD = (7, 4)
square_size = 0.011
# Prepare arrays to store 3D points and 2D points for all images
objpoints = []  # 3D points (real world space)
imgpoints = []  # 2D points (image plane)

# We assume each square in the chessboard is 1 unit in "world space"
# (For absolute units, you can specify each square edge = e.g. 0.02m)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = (
    np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
    .T.reshape(-1, 2)
    * square_size
)

# Load images (assume all images in a folder named "calib_imgs/*.jpg")
images = glob.glob("./calibration_data/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (1920, 1080))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret == True:
        # Refine corner locations (optional but recommended)
        corners_sub = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners_sub)
        objpoints.append(objp)
        
        # Visualization (draw corners)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_sub, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Now calibrate the camera
# This returns camera matrix, distortion coeffs, rotation and translation vectors
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# Save to a file so you can use later
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

# Calculate and print the reprojection error to gauge calibration quality
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("Total error: ", mean_error/len(objpoints))
