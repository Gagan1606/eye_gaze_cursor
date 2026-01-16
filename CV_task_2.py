import cv2  
import pyautogui as pag
import numpy as np
import mediapipe as mp
import time

mp_face_mesh=mp.solutions.face_mesh
capture=cv2.VideoCapture(0)
#facemesh landmark indices, thresholds and other parameters
face_width=None
left_corners=[362, 263]
right_corners=[33, 133]
left_iris_idx=[468,469,470,471,472,473]
right_iris_idx = [474,475,476,477]
max_x, max_y=pag.size()
left_eye_landmarks=[362, 385, 387, 263, 373, 380]
right_eye_landmarks=[33, 160, 158, 133, 153, 144]
ear_threshold=0.2
blink_time_threshold=0.8
num_blinks=0
last_ear=1
blink_time_stamp=[]
hold_position_threshold=face_width*0.1 #15
hold_time_threshold=3
hold_time_stamp=[]

def calculate_ear(eye_landmarks, landmarks_arr):
    v1= np.linalg.norm(landmarks_arr[eye_landmarks[1]] - landmarks_arr[eye_landmarks[5]])
    v2= np.linalg.norm(landmarks_arr[eye_landmarks[2]] - landmarks_arr[eye_landmarks[4]])
    h=np.linalg.norm(landmarks_arr[eye_landmarks[0]]-landmarks_arr[eye_landmarks[3]])
    ear=(v1+v2)/(h*2)
    return ear

def reset_trackers(except_tracker=None):#function to clear the current tracking list when an action is executed 
    if except_tracker!="blink":
        blink_time_stamp.clear()
    if except_tracker!="hold":
        hold_time_stamp.clear()
    return None

 
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    neutral_avg=None
    prev_iris_centre=None
    while True:
        _, frame=capture.read()
        if len(frame): print("frame exists")
        results=face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #because facemesh expects rgb
        if (results.multi_face_landmarks) is not None: print(f'num_faces:{len(results.multi_face_landmarks)}')
        if results.multi_face_landmarks: 
            landmarks=results.multi_face_landmarks[0].landmark
            landmarks_arr=np.array([[lm.x, lm.y] for lm in landmarks])
            #calculating face width
            if face_width is None: face_width = np.linalg.norm(landmarks_arr[left_corners[0]] - landmarks_arr[right_corners[0]])
            #calculating the actual eye centre
            left_eye_centre=(np.mean(landmarks_arr[left_corners], axis=0))
            right_eye_centre=(np.mean(landmarks_arr[right_corners], axis=0))
            print(f"eye:{right_eye_centre*np.array([max_x, max_y])}")
            #calculating where the person is looking at
            left_iris_centre=np.mean(landmarks_arr[left_iris_idx], axis=0)
            right_iris_centre=np.mean(landmarks_arr[right_iris_idx], axis=0)
            print(f"iris:{right_iris_centre*np.array([max_x, max_y])}")
            #calculating the raw offset
            left_offset=(left_iris_centre-left_eye_centre)*np.array([max_x, max_y])
            right_offset=(right_iris_centre-right_eye_centre)*np.array([max_x, max_y])
            avg_offset=(left_offset+right_offset)/2
            print(f'avg_offset:{avg_offset}')
            
            if neutral_avg is None:
                neutral_avg=avg_offset #setting neutral gaze direction 
                print(f"calculated neutral: {neutral_avg}px")
            relative_x = avg_offset[0] - neutral_avg[0]
            relative_y=avg_offset[1]-neutral_avg[1]
            print(f"x - axis - real_offset:{avg_offset[0]:.1f} calc_neutral:{neutral_avg[0]:.1f} rel_offset:{relative_x:.1f}")
            print(f"y - axis - real_offset:{avg_offset[1]:.1f} calc_neutral:{neutral_avg[1]:.1f} rel_offset:{relative_y:.1f}")
            threshold_x=face_width*0.05 #7
            threshold_y=face_width*0.08 #12
            #right and left movement, if its greater than the threshold offset, move the cursor
            if abs(avg_offset[0]-neutral_avg[0]) > threshold_x:  
                print(f"movement is present - {avg_offset[0]}")
                print(f"relative:{avg_offset[0]-neutral_avg[0]}")
                if (avg_offset[0]-neutral_avg[0]) > threshold_x: #right
                    print(f"right {avg_offset[0]:.2f}")
                    cv2.putText(frame, f"right {avg_offset[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    pag.moveRel(15, 0)
                    reset_trackers("blink")
                elif avg_offset[0]-neutral_avg[0]<-threshold_x: #left
                    print(f"left {avg_offset[0]:.2f}")
                    cv2.putText(frame, f"left {avg_offset[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    pag.moveRel(-15, 0)
                    reset_trackers("blink")
            #up and down movement, we dont really a lot of neutral offset in case of y-axis, so we are using avg_offset directly instead of relative
            if (avg_offset[1] > threshold_y) or (avg_offset[1]<-threshold_y):
                print(f"up or down triggered - avg_offset_y: {avg_offset[1]}")
                if avg_offset[1] > threshold_y: 
                    print("moving down")
                    cv2.putText(frame, "moving down", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    pag.moveRel(0, 15)
                    reset_trackers("blink")
                elif avg_offset[1] < -threshold_y:
                    print("moving up")
                    cv2.putText(frame, "moving up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    pag.moveRel(0, -15)
                    reset_trackers("blink")
            
            #blink detection, we are using the eye aspect ratio, its a blink if the ear is less than the threshold ear
            left_ear=calculate_ear(left_eye_landmarks, landmarks_arr)
            right_ear=calculate_ear(right_eye_landmarks, landmarks_arr)
            avg_ear=(left_ear+right_ear)/2
            current_time=time.time()
            if avg_ear<ear_threshold: 
                print(f'blink detected')
                reset_trackers("blink")
                blink_time_stamp.append(current_time)
                blink_time_stamp=[x for x in blink_time_stamp if current_time-x<blink_time_threshold]
                if len(blink_time_stamp)==3 and blink_time_stamp[2]-blink_time_stamp[0]<blink_time_threshold: #right click
                    print(f'right click')
                    cv2.putText(frame, f'right click', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    pag.rightClick()
                    reset_trackers("blink")
                    prev_iris_centre=left_iris_centre
                elif len(blink_time_stamp)==2 and blink_time_stamp[1]-blink_time_stamp[0]<blink_time_threshold: #left click
                    print(f'left click')
                    cv2.putText(frame, f'left click', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    pag.click()
                    reset_trackers("blink")
                    prev_iris_centre=left_iris_centre
            last_ear=avg_ear #update the ear
            
            #gaze detection
            if prev_iris_centre is not None:
                if np.linalg.norm(prev_iris_centre-left_iris_centre)<hold_position_threshold: #if iris centre is almost at the same location, its a gaze
                    hold_time_stamp.append(current_time)
                    reset_trackers("blink")
                    if len(hold_time_stamp):
                        if hold_time_stamp[len(hold_time_stamp)-1]-hold_time_stamp[0]>=hold_time_threshold:
                            pag.click()
                            cv2.putText(frame, f'select', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                            reset_trackers("blink")
                prev_iris_centre=left_iris_centre
            else: prev_iris_centre=left_eye_centre
                
        cv2.imshow("result", frame)
        if cv2.waitKey(100) & 0xFF == ord('x'):break
        