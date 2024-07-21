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
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=50)
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

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in range(n_frames-1):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        
        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

        # Extract translation
        dx = m[0,2]
        dy = m[1,2]

        # Extract rotation angle
        da = np.arctan2(m[1,0], m[0,0])

        # Store transformation
        transforms[i] = [dx,dy,da]

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
        da = transforms_smooth[i,2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

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
input_video = "input2.mp4"
output_video = "stabilized_video.mp4"
stabilize_video(input_video, output_video)