import cv2
import numpy as np

def create_transformation_matrix(rvec, tvec):
    """Creates a 4x4 homogeneous transformation matrix from rvec and tvec."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    T[0, 3] -= 1
    return T

def calibrate_world_frame(camera_matrix, dist_coeffs,
                          ref_marker_id=0,
                          marker_size=0.05):
    """
    Detects the reference marker and saves its pose to a file, with the marker
    origin at the marker's center.

    Args:
        camera_matrix: Camera matrix from calibration.
        dist_coeffs: Distortion coefficients from calibration.
        ref_marker_id: ID of the reference ArUco marker.
        marker_size: Size of the ArUco marker in meters.
    """

    # Use an ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ref_marker_pose = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # Check if we see at least one marker
        if ids is not None:
            for i in range(len(ids)):
                if ids[i][0] == ref_marker_id:
                    # 2D corners shape: (1,4,2) => reshape to (4,2)
                    marker_corners_2d = corners[i].reshape((4, 2))

                    # Make sure the corner order matches your 3D definition
                    # Typically ArUco detects corners in: top-left, top-right,
                    # bottom-right, bottom-left. Let's assume that's the order.
                    # If it's reversed or out of order, reorder accordingly.

                    # 3D corners with origin at the center:
                    # (top-left, top-right, bottom-right, bottom-left)
                    marker_points_3d = np.array([
                        [-marker_size/2, -marker_size/2, 0],  # top-left
                        [ marker_size/2, -marker_size/2, 0],  # top-right
                        [ marker_size/2,  marker_size/2, 0],  # bottom-right
                        [-marker_size/2,  marker_size/2, 0],  # bottom-left
                    ], dtype=np.float32)

                    # Solve for rvec, tvec
                    ret_pnp, rvec, tvec = cv2.solvePnP(
                        marker_points_3d,
                        marker_corners_2d,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if not ret_pnp:
                        print("solvePnP failed for marker", ref_marker_id)
                        continue

                    ref_marker_pose = (rvec, tvec)

                    # Optionally draw the marker outline for debugging
                    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    # Draw the axes (they should now appear at the marker center)
                    frame = cv2.drawFrameAxes(
                        frame, camera_matrix, dist_coeffs,
                        rvec, tvec, 0.03
                    )

                    print("Reference marker detected (centered). Press 's' to save and exit.")
                    break  # Found the reference marker

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1)
        if key == ord('s') and ref_marker_pose is not None:
            # Save the reference marker pose
            T_camera_to_world = create_transformation_matrix(ref_marker_pose[0], ref_marker_pose[1])
            np.save("./calibration_data/world_frame_transform.npy", T_camera_to_world)
            print("World frame transformation matrix saved to world_frame_transform.npy")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_matrix = np.load("./calibration_data/camera_matrix.npy")
    dist_coeffs = np.load("./calibration_data/dist_coeffs.npy")
    calibrate_world_frame(camera_matrix, dist_coeffs)
