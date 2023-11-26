import cv2
import numpy as np
import Cards
import os

# RANK AND SUIT WIDTH AND HEIGHT SETTINGS
RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# Card to check
card_check = "QH"
isolate = "SUIT"
folder_save = "temp"
framePerImages = 10

# Path of video file and destination of dataset images
video_path = "C:/Users/rayha/Pictures/Camera Roll/"
image_destination = "../dataset/" + isolate.lower() + "/"
os.chdir(image_destination + folder_save + "/")

# Open the video file
cap = cv2.VideoCapture(video_path + card_check + ".mp4")
videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Frame Counter
count = 0

# Loop through the frames of the video
while True:
    # Read a frame from the video file
    ret, frame = cap.read()

    # If the frame was not read successfully, break out of the loop
    if not ret:
        break

    if count % framePerImages == 0:
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        retval, thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)

        # Find contours and sort them by size
        cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea,reverse=True)

        # Assume largest contour is the card. If there are no contours, print an error
        flag = 0
        image2 = frame.copy()

        if len(cnts) == 0:
            continue

        card = cnts[0]

        # Approximate the corner points of the card
        peri = cv2.arcLength(card,True)
        approx = cv2.approxPolyDP(card,0.01*peri,True)
        pts = np.float32(approx)

        x,y,w,h = cv2.boundingRect(card)

        # Flatten the card and convert it to 200x300
        warp = Cards.flattener(frame,pts,w,h)

        # Grab corner of card image, zoom, and threshold
        corner = warp[0:84, 0:32]
        #corner_gray = cv2.cvtColor(corner,cv2.COLOR_BGR2GRAY)
        corner_zoom = cv2.resize(corner, (0,0), fx=4, fy=4)
        corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)
        retval, corner_thresh = cv2.threshold(corner_blur, 155, 255, cv2. THRESH_BINARY_INV)

        # Isolate suit or rank
        if isolate == "RANK": # Isolate rank
            rank = corner_thresh[20:185, 0:128] # Grabs portion of image that shows rank
            rank_cnts, hier = cv2.findContours(rank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if len(rank_cnts) < 1:
                continue
            rank_cnts = sorted(rank_cnts, key=cv2.contourArea,reverse=True)
            x,y,w,h = cv2.boundingRect(rank_cnts[0])
            rank_roi = rank[y:y+h, x:x+w]
            rank_sized = cv2.resize(rank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
            final_img = rank_sized

        if isolate == "SUIT": # Isolate suit
            suit = corner_thresh[186:336, 0:128] # Grabs portion of image that shows suit
            suit_cnts, hier = cv2.findContours(suit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if len(suit_cnts) < 1:
                continue
            suit_cnts = sorted(suit_cnts, key=cv2.contourArea,reverse=True)
            x,y,w,h = cv2.boundingRect(suit_cnts[0])
            suit_roi = suit[y:y+h, x:x+w]
            suit_sized = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
            final_img = suit_sized

        
        cv2.imwrite(str(count // framePerImages) + ".jpg", final_img)

    if count >= videoFrames:
        break
    count += 1

    # Wait for a key press and check if the 'q' key was pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()
