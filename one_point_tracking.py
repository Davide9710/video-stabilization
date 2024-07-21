import cv2
import numpy as np

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(2):  # Smooth x and y
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=30)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read first frame
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Find a good point to track
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1, qualityLevel=0.01, minDistance=30, blockSize=3)
    
    if prev_pts is None or len(prev_pts) == 0:
        raise Exception("No good features found to track")

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 2), np.float32)

    for i in range(n_frames-1):
        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Select good points
        good_new = curr_pts[status==1]
        good_old = prev_pts[status==1]

        # If no good points, find new points
        if len(good_new) == 0:
            new_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=1, qualityLevel=0.01, minDistance=30, blockSize=3)
            if new_pts is not None and len(new_pts) > 0:
                prev_pts = new_pts
            else:
                # If no points found, use the previous transformation
                transforms[i] = transforms[i-1] if i > 0 else [0, 0]
                prev_gray = curr_gray
                continue
        else:
            # Calculate movement
            dx = good_new[0][0] - good_old[0][0]
            dy = good_new[0][1] - good_old[0][1]

            # Store transformation
            transforms[i] = [dx, dy]

            # Update previous points
            prev_pts = good_new.reshape(-1, 1, 2)

        # Move to next frame
        prev_gray = curr_gray

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Smooth trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames-1):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]

        # Reconstruct transformation matrix accordingly to new values
        m = np.array([[1, 0, -dx], [0, 1, -dy]])

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width,height))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        out.write(frame_stabilized)

    # Release video
    cap.release()
    out.release()

# Usage
input_video = "input.mp4"
output_video = "stabilized_video_single_point.mp4"
stabilize_video(input_video, output_video)