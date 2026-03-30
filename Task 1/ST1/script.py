import numpy as np
import cv2 as cv

def manual_pyr_lk(pts_arr, curr_img, prev_img, win_size=15, max_level=2, max_iters=5, eps=0.03):
    #create pyramids for current and previous image
    curr_pyr = [curr_img]
    prev_pyr = [prev_img]
    for _ in range(max_level):
        #curr_pyr.append(cv.pyrDown(curr_pyr[-1])) 
        #prev_pyr.append(cv.pyrDown(prev_pyr[-1]))
        #reduce resolution of each level by half, using the pyrDown functions makes it easier
        blurred = cv.GaussianBlur(curr_pyr[-1], (5, 5), sigmaX=1.0, sigmaY=1.0)
        curr_pyr.append(blurred[::2, ::2])
        blurred = cv.GaussianBlur(prev_pyr[-1], (5, 5), sigmaX=1.0, sigmaY=1.0)
        prev_pyr.append(blurred[::2, ::2])
        
    #Array g stores best guesses of corner displacements, status tells us which points from pts_arr are still worth tracking
    g = np.zeros_like(pts_arr, dtype=np.float32)
    status = np.ones(len(pts_arr), dtype=np.uint8)
    
    #Iterate over each layer of the pyramid
    for level in range(max_level, -1, -1):
        p_img = prev_pyr[level]
        c_img = curr_pyr[level]
        
        #Calculate Sobel gradients for the previous image at this scale
        Ix = cv.Sobel(p_img, cv.CV_32F, 1, 0, ksize=3, scale=1.0/8.0)
        Iy = cv.Sobel(p_img, cv.CV_32F, 0, 1, ksize=3, scale=1.0/8.0)
        
        #Since the original image has been scaled down, the point coordinates must also be scaled down
        level_pts = pts_arr / (2 ** level)
        
        for i, pt in enumerate(level_pts):
            if status[i] == 0:
                continue
            
            x, y = pt.ravel()
            gx, gy = g[i].ravel() #Current displacement guess
            
            #Extract stationary patch and gradients from the previous image
            p_patch = cv.getRectSubPix(p_img, (win_size, win_size), (x, y))
            ix_patch = cv.getRectSubPix(Ix, (win_size, win_size), (x, y))
            iy_patch = cv.getRectSubPix(Iy, (win_size, win_size), (x, y))
            #We use the Inverse Composition method:
            #For each patch, the A and A.T @ A are calculated beforehand rather than separately for each iteration
            #We only update g at each level, and then use g to update the coordinates of the tracked points at the end
            A = np.column_stack((ix_patch.flatten(), iy_patch.flatten()))
            H = A.T @ A
            
            #If the det(H) is close to zero, then there usually isn't any corner in the patch anymore
            if abs(np.linalg.det(H)) < 1e-5:
                status[i] = 0
                continue
                
            H_inv = np.linalg.inv(H)
            #Iteratively update the guesses for flow using vector (v) as the current error
            vx, vy = 0.0, 0.0
            
            for _ in range(max_iters):
                #Extract moving patch from CURRENT image using our guess + residual
                c_patch = cv.getRectSubPix(c_img, (win_size, win_size), (x + gx + vx, y + gy + vy))
                
                #The temporal error(I_t)
                b = (p_patch - c_patch).flatten()
                
                #Solve for calculated optical flow velocities
                delta = H_inv @ (A.T @ b)
                vx += delta[0]
                vy += delta[1]
                
                #Stop iterating early if the shift is microscopically small
                if delta[0]**2 + delta[1]**2 < eps**2:
                    break
                    
            #Accumulate the found displacement into our total guess
            g[i][0][0] += vx
            g[i][0][1] += vy
            
            #Mark point as lost if it moves off-screen
            curr_x = x + g[i][0][0]
            curr_y = y + g[i][0][1]
            if (curr_x < 0 or curr_x >= p_img.shape[1] or curr_y < 0 or curr_y >= p_img.shape[0]):
                status[i] = 0
                
        #Scale the displacement guess up for the next, higher-resolution level
        if level > 0:
            g *= 2.0
            
    #Final coordinates are the original points + total accumulated displacement
    new_pts = pts_arr + g
    return new_pts, status


class Vid: 
    def __init__(self, filename):
        vid = cv.VideoCapture(filename)
        ret, first_frame = vid.read()
        if not ret: 
            print(f"Could not open {filename}")
            return
            
        mask = np.zeros_like(first_frame)
        prev_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY).astype(np.float32)
        corners = cv.goodFeaturesToTrack(prev_frame.astype(np.uint8), 15, 0.01, 10)
        
        color = np.random.randint(0, 255, (100, 3)) #colors are randomly chosen for each point
        frame_cnt = 1
        
        while True:
            ret, frame = vid.read()
            if not ret: break 
                
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)
            frame_cnt += 1
            
            if corners is not None and len(corners) > 0:
                new_corners, status = manual_pyr_lk(corners, gray_frame, prev_frame)            
                #Filter out points having status = 0
                good_new = new_corners[status == 1]
                good_old = corners[status == 1]
                #draw the points and their displacements
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    col = color[i % 100].tolist()
                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), col, 2)
                    frame = cv.circle(frame, (int(a), int(b)), 4, col, -1)
                
                corners = good_new.reshape(-1, 1, 2)
            else:
                corners = None
                
            img = cv.add(frame, mask)
            
            #Refresh features occasionally
            if frame_cnt % 30 == 0:
                new_features = cv.goodFeaturesToTrack(gray_frame.astype(np.uint8), 100, 0.01, 10)
                if new_features is not None:
                    corners = new_features
                    mask = np.zeros_like(first_frame)
                    
            prev_frame = gray_frame.copy()
            cv.imshow('Lucas-Kanade Optical Flow', img)
            
            if cv.waitKey(30) == ord('q'):
                break
            
        vid.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    Vid("clip.mp4")