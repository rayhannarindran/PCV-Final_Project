import cv2
import numpy as np
import Cards
import os

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

image_path = "C:/Users/rayha/Pictures/Camera Roll/"

card_check = "QS"
isolate = "suit"

# Open the image file
frame = cv2.imread(image_path + card_check + ".jpg")

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
    print("No contours were found")
    quit()

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
if isolate == "rank": # Isolate rank
    rank = corner_thresh[20:185, 0:128] # Grabs portion of image that shows rank
    rank_cnts, hier = cv2.findContours(rank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rank_cnts = sorted(rank_cnts, key=cv2.contourArea,reverse=True)
    x,y,w,h = cv2.boundingRect(rank_cnts[0])
    rank_roi = rank[y:y+h, x:x+w]
    rank_sized = cv2.resize(rank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
    final_img = rank_sized
    cv2.imwrite(card_check + ".jpg", final_img)

if isolate == "suit": # Isolate suit
    suit = corner_thresh[186:336, 0:128] # Grabs portion of image that shows suit
    suit_cnts, hier = cv2.findContours(suit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    suit_cnts = sorted(suit_cnts, key=cv2.contourArea,reverse=True)
    x,y,w,h = cv2.boundingRect(suit_cnts[0])
    suit_roi = suit[y:y+h, x:x+w]
    suit_sized = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
    final_img = suit_sized
    cv2.imwrite(card_check + ".jpg", final_img)
