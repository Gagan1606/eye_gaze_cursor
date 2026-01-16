AIM : detect horizontal, vertical movements and blinking, gazing and perform various cursor operations

Main Idea: face mesh landmarks return coordinates of various parts of the face, we use them to detect if the person is doing a 
particular activity and use pyautogui to perform cursor operations

We calculate neutral iris position assuming the first few frames is going to be a straight gaze and eye centre using the corners of the eyes.
Then we calculate the usual offset - iris doesn't exactly lie in the eye centre.  
1) for horizontal and vertical movements - 
        we calculate the offset of iris from the eye centre and compare with neutral offset and see if it is more than a threshold and trigger the cursor operation.
2) blinking - we use eye aspect ratio thresholding, i.e. vertical to horizontal points on the eye, above a certain threshold is considered a blinking. We use time stamps to see if its a valid double or triple blink.
3) gazing - iris centre or the average offset staying almost the same is considered gazing and we use time stamps to see if its a valid gaze for continuos amount of time.
