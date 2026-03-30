import cv2
import numpy as np
import pybullet as p
import math
import time
from simulation_setup import setup_simulation
from lk import manual_pyr_lk
def get_agent_camera_frame(car_id, width=320, height=240):
    # Get the car's current position and orientation in the world
    pos, orn = p.getBasePositionAndOrientation(car_id)
    #Convert the quaternion orientation to a 3x3 rotation matrix
    rot_matrix = p.getMatrixFromQuaternion(orn)
    forward_vector = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
    #Mount the camera slightly above and forward of the car's center
    #pos is a tuple, convert to numpy array
    car_pos = np.array(pos)
    camera_eye = car_pos + np.array([0.0, 0.0, 0.3]) + (forward_vector * 0.2)
    #Set the target point a few meters directly in front of the camera
    camera_target = camera_eye + (forward_vector * 5.0)
    camera_up = [0, 0, 1]
    #Compute the View and Projection matrices
    view_matrix = p.computeViewMatrix(camera_eye.tolist(), camera_target.tolist(), camera_up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60.0, aspect=width / float(height), nearVal=0.1, farVal=100.0)
    img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    #Extract the RGB channels
    rgba = np.reshape(img_arr[2], (height, width, 4)).astype(np.uint8) #img_arr is a list, its third item is the rgba pixel values
    rgb = rgba[:, :, :3].copy() #opacity values aren't required for our purpose
    #the .copy() ensures rgb is a contiguous ndarray, this is important because cv2 doesn't like working with non-contiguous ndarrays
    return rgb
def calculate_foe_ransac(prev_pts, curr_pts, iters=50, distance_threshold=3.0):
    """
    Calculates the FOE using RANSAC to reject outlier flow vectors.
    
    """
    if prev_pts is None or curr_pts is None:
        return None
    if len(prev_pts) < 2 or len(prev_pts) != len(curr_pts):
        return None

    prev_pts = prev_pts.reshape(-1, 2)
    curr_pts = curr_pts.reshape(-1, 2)

    flow_vectors = curr_pts - prev_pts
    u = flow_vectors[:, 0]
    v = flow_vectors[:, 1]

    # Filter out stationary points
    magnitude_sq = u**2 + v**2
    moving_mask = magnitude_sq > 0.1 
    
    u = u[moving_mask]
    v = v[moving_mask]
    x = prev_pts[moving_mask, 0]
    y = prev_pts[moving_mask, 1]

    num_points = len(x)
    if num_points < 2:
        return None

    # Line equation: -v*X + u*Y = -v*x + u*y  ->  A * FOE = B
    A = np.column_stack((-v, u))
    B = -v * x + u * y

    # Pre-calculate the vector norms so we can find true perpendicular distances later
    norms = np.sqrt(v**2 + u**2)
    # Avoid division by zero just in case
    norms[norms == 0] = 1e-6 
    
    A_norm = A / norms[:, np.newaxis]
    B_norm = B / norms

    best_foe = None
    max_inliers = -1

    # --- The RANSAC Loop ---
    for _ in range(iters):
        # 1. Randomly sample 2 lines
        idx = np.random.choice(num_points, 2, replace=False)
        A_samp = A[idx]
        B_samp = B[idx]

        try:
            # 2. Find the exact intersection of these 2 lines
            foe_guess = np.linalg.solve(A_samp, B_samp)
        except np.linalg.LinAlgError:
            continue # Skip if the two lines are perfectly parallel

        # 3. Calculate the perpendicular distance of ALL lines to this guessed FOE
        distances = np.abs(np.dot(A_norm, foe_guess) - B_norm)

        # 4. Count how many lines are "inliers" (pass close to the guess)
        inliers_mask = distances < distance_threshold
        num_inliers = np.sum(inliers_mask)

        # 5. Keep the guess with the most votes
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            
            # Optional but recommended: run standard least squares ONLY on the winning inliers 
            # to get the most accurate sub-pixel center
            best_foe, _, _, _ = np.linalg.lstsq(A[inliers_mask], B[inliers_mask], rcond=None)

    if best_foe is not None:
        return int(best_foe[0]), int(best_foe[1])
        
    return None
def calculate_gradient_forces(prev_pts, curr_pts, foe_point, img_width=320, img_height=240):
    """
    Calculates 2D attractive and repulsive vectors for gradient-based control.
    """
    #Define the car's position in the image (bottom center)
    car_pos = np.array([img_width / 2.0, img_height], dtype=np.float32)
    #Define the target (The vanishing point / horizon center)
    target_pos = np.array([img_width / 2.0, img_height / 2.0], dtype=np.float32)
    
    #Calculate the Attractive Vector
    #Attractive vector pointing from car to the target
    v_att = target_pos - car_pos 
    #Normalize it and apply an attractive gain (alpha)
    norm_att = np.linalg.norm(v_att)
    if norm_att > 0:
        v_att = (v_att / norm_att) * 37.0 #multiplying by alpha
    else:
        v_att = np.array([0.0, 0.0])

    # If no tracking points, just return the attractive vector pulling us forward
    if prev_pts is None or curr_pts is None or foe_point is None or len(prev_pts) == 0:
        return v_att
        
    foe_x, foe_y = foe_point
    prev_pts = prev_pts.reshape(-1, 2)
    curr_pts = curr_pts.reshape(-1, 2)
    
    #Calculate the Repulsive Vectors (based on TTC)
    #Flow vectors and TTC math
    flow_vectors = curr_pts - prev_pts
    dist_to_foe = np.linalg.norm(prev_pts - np.array([foe_x, foe_y]), axis=1)
    speed = np.linalg.norm(flow_vectors, axis=1)
    speed = np.maximum(speed, 1e-5)

    ttc = dist_to_foe / speed
    ttc = np.maximum(ttc, 1e-5)
    
    # Repulsive vectors push FROM the obstacle TO the car
    # v_rep_dir = car_pos - obstacle_pos
    v_rep_dirs = car_pos - prev_pts
    
    # Normalize the direction vectors
    norms = np.linalg.norm(v_rep_dirs, axis=1)
    norms = np.maximum(norms, 1e-5)
    v_rep_dirs_normalized = v_rep_dirs / norms[:, np.newaxis]
    
    # Scale by inverse TTC and a repulsive gain (gamma)
    # Closer objects (small TTC) create massive vectors
    gamma = 40.0
    magnitudes = gamma / ttc

    magnitudes = np.where(ttc < 150, magnitudes, 0.0) #remove points which are too far away
    left_wall_mask = prev_pts[:, 0] < (img_width * 0.20) #too prevent driving off the walls
    right_wall_mask = prev_pts[:, 0] > (img_width * 0.80)
    magnitudes[left_wall_mask] *= 4.0
    magnitudes[right_wall_mask] *= 4.0
    # Multiply normalized directions by magnitudes
    repulsive_vectors = v_rep_dirs_normalized * magnitudes[:, np.newaxis]
    
    # Sum all repulsive vectors into one net repulsive force
    net_v_rep = np.sum(repulsive_vectors, axis=0)
    
    #Sum for Final Gradient Vector
    final_gradient_vector = v_att + net_v_rep
    
    return final_gradient_vector
if __name__ == "__main__":
    car_id, steer_j, motor_j = setup_simulation()
    
    #preprocessing: get the first frame and the corner detection initialized
    first_frame = get_agent_camera_frame(car_id, width=320, height=240)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feature_params = dict(maxCorners=300, qualityLevel=0.005, minDistance=10, blockSize=7)
    #Create a mask for removing tracking on the part of the feed containing the car
    feature_mask = 255 * np.ones_like(prev_gray, dtype=np.uint8)
    # Black out (0) the bottom portion of the image where the car hood is.
    feature_mask[135:240, 110:210] = 0
    feature_mask[180:240, 0:320] = 0
    # Pass this mask into the feature detector
    prev_pts = cv2.goodFeaturesToTrack(prev_gray.astype(np.uint8), mask=feature_mask, **feature_params)
    mask = np.zeros_like(first_frame)
    smoothed_foe = None
    alpha = 0.05  # Smoothing factor for foe
    smoothed_steering = 0.0
    steering_alpha = 0.1  # The "stiffness" of the steering column (0.05 to 0.15 is usually best)
    print("\nSimulation running. Close the PyBullet window or press Ctrl+C to exit.")
    dt = 1.0 / 60.0
    try:
        while True:
            #Capture the agent's view
            frame = get_agent_camera_frame(car_id, width=320, height=240)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            
            #feed the grayscale frame into our custom LK and get the status array & new coordinates
            next_pts, status_arr = manual_pyr_lk(prev_pts, gray, prev_gray)         
            #filter the output points to remove those not deemed worth tracking
            good_new = next_pts[status_arr == 1]
            good_old = prev_pts[status_arr == 1]
            
            #NumPy guarantees that this way of filtering gives the same number of points in good_old and good_new


            #Calculate and draw the FOE using origin tracking
            foe_point = calculate_foe_ransac(good_old, good_new)
            current_foe_tuple = None
            if foe_point is not None:
                raw_foe = np.array(foe_point, dtype=np.float32)
                if smoothed_foe is None:
                    #happens in the first frame
                    smoothed_foe = raw_foe
                else:
                    #take the exponential moving average of the current and previous foe's
                    smoothed_foe = (alpha * raw_foe) + ((1.0 - alpha) * smoothed_foe)
                
                #draw the smoothed foe
                if smoothed_foe is not None:
                    foe_x, foe_y = int(smoothed_foe[0]), int(smoothed_foe[1])
                    foe_x = max(0, min(foe_x, 320)) 
                    foe_y = max(0, min(foe_y, 240))
                    current_foe_tuple = (foe_x, foe_y)
                    cv2.drawMarker(frame, (foe_x, foe_y), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, 
                                   markerSize=20, thickness=2)
                    cv2.circle(frame, (foe_x, foe_y), 5, (255, 0, 0), -1)


            #find the heading vector using visual potential gradients
            final_vector = calculate_gradient_forces(good_old, good_new, current_foe_tuple)
    
            # Extract X and Y components of the force
            fx = final_vector[0]
            fy = final_vector[1] # Note: In images, Y goes down. Up is negative Y.
    
            # Calculate desired steering angle using the arctangent of the gradient
            # Since forward is -Y in image coordinates, we invert fy for standard math
            raw_steering_angle = math.atan2(fx, -fy) 
    
            # Clamp to physical limits of the PyBullet vehicle (e.g., ~0.5 radians)
            raw_steering_command = max(-0.5, min(raw_steering_angle, 0.5))
            smoothed_steering = (steering_alpha * raw_steering_command) + ((1.0 - steering_alpha) * smoothed_steering)

            #Integrate with the PyBullet API, and finalize the steering instructions
            for joint in steer_j:
                p.setJointMotorControl2(bodyIndex=car_id, jointIndex=joint, controlMode=p.POSITION_CONTROL,
                                         targetPosition=smoothed_steering)
            speed = 15.0
            max_force = 10.0
            for joint in motor_j:
                p.setJointMotorControl2(bodyIndex= car_id, jointIndex=joint, controlMode=p.VELOCITY_CONTROL, 
                                         targetVelocity=speed, force=max_force)
                

            #draw the corners' vectors onto the mask
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            img = cv2.add(frame, mask)
            prev_gray = gray.copy()
            if len(good_new) > 0:
                prev_pts = good_new.reshape(-1, 1, 2) 
                #cv2 requires a particular format of points for input, this ensures that
                
            else:
                prev_pts = None
            #Check if number of tracked points has become too low
            if prev_pts is None or len(prev_pts) < 20:
                feature_mask = 255 * np.ones_like(gray, dtype=np.uint8)
                # Black out (0) the bottom portion of the image where the car hood is.
                feature_mask[135:240, 110:210] = 0
                feature_mask[180:240, 0:320] = 0
                # Pass this mask into the feature detector
                prev_pts = cv2.goodFeaturesToTrack(gray.astype(np.uint8), mask=feature_mask, **feature_params)
                mask = np.zeros_like(first_frame)
                
            #show the image, changing frame from RGB to BGR because imshow requires it
            cv2.imshow("Mounted Camera View", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            
            #Step the physics engine
            p.stepSimulation()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass #in case Ctrl+C is pressed, ends the while loop
    finally:
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()
        print("Simulation ended.")