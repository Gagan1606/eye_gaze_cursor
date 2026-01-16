import cv2  
import pyautogui as pag
import numpy as np
import mediapipe as mp

mp_face_mesh=mp.solutions.face_mesh
capture=cv2.VideoCapture(0)
left_corners=[362, 263]
right_corners=[33, 133]
left_iris_idx=[468,469,470,471,472,473]
right_iris_idx = [474,475,476,477]
threshold_offset=5
max_x, max_y=pag.size()

# def extent_move(offset, direction):
# def best_direction():
    
 
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while True:
        _, frame=capture.read()
        if len(frame): print("frame exists")
        results=face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if (results.multi_face_landmarks) is not None: print(f'num_faces:{len(results.multi_face_landmarks)}')
        if results.multi_face_landmarks: 
            landmarks=results.multi_face_landmarks[0].landmark
            landmarks_arr=np.array([[lm.x, lm.y] for lm in landmarks])
            left_eye_centre=(np.mean(landmarks_arr[left_corners], axis=0))
            if np.size(left_eye_centre): print(f'l_centre:{np.size(left_eye_centre)}')
            right_eye_centre=(np.mean(landmarks_arr[right_corners], axis=0))
            print(f"eye:{right_eye_centre*np.array([max_x, max_y])}")
            left_iris_centre=np.mean(landmarks_arr[left_iris_idx], axis=0)
            right_iris_centre=np.mean(landmarks_arr[right_iris_idx], axis=0)
            print(f"iris:{right_iris_centre*np.array([max_x, max_y])}")
        
            left_offset=(left_eye_centre-left_iris_centre)*np.array([max_x, max_y])
            print(f'left_offset:{left_offset}')
            right_offset=(right_iris_centre-right_eye_centre)*np.array([max_x, max_y])
            avg_offset=(left_offset+right_offset)/2
            
            if 'neutral_avg' not in globals():
                neutral_avg=avg_offset 
                print(f"CALIBRATED neutral: {neutral_avg}px")
        
            # if abs(avg_offset[0])>(avg_offset[1]):
            #     print("entered if 1") 
            #     if avg_offset[0]>0 : 
            #         print("entered if 2")
            #         cv2.putText(frame, "moving right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            #         # pag.moveTo(max_x-10, None)
            #         pag.moveRel(20, 0)
            # max(avg_offset[0],avg_offset[1])
            
            relative_offset = avg_offset[0] - neutral_avg[0]
            print(f"RAW:{avg_offset[0]:.1f} NEUT:{neutral_avg[0]:.1f} REL:{relative_offset:.1f}")

            if abs(avg_offset[0]-neutral_avg[0]) > 6:  
                print(f"TRIGGERED! - {avg_offset[0]}")
                print(f"relative:{avg_offset[0]-neutral_avg[0]}")
                if (avg_offset[0]-neutral_avg[0]) > 7:
                    print(f"RIGHT {avg_offset[0]:.2f}")
                    cv2.putText(frame, f"RIGHT {avg_offset[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    pag.moveRel(15, 0)
                elif avg_offset[0]-neutral_avg[0]<-7:
                    print(f"LEFT {avg_offset[0]:.2f}")
                    cv2.putText(frame, f"LEFT {avg_offset[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    pag.moveRel(-15, 0)

        cv2.imshow("result", frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):break
        
        # else:
        #     cv2.imshow("result", frame)  # Show empty frame
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #     continue
