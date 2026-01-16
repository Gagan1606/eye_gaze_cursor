AIM : detect horizontal, vertical movements and blinking, gazing and perform various cursor operations

Main Idea: face mesh landmarks return coordinates of various parts of the face, we use them to detect if the person is doing a 
particular activity and use pyautogui to perform cursor operations

We calculate neutral iris position assuming the first few frames is going to be a straight gaze and eye centre using the corners of the eyes.
Then we calculate the usual offset - iris doesn't exactly lie in the eye centre.  
1) for horizontal and vertical movements - 
        we calculate the offset of iris from the eye centre and compare with neutral offset and see if 