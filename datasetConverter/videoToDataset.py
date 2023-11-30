import cv2
import numpy as np
import deteksiKartu as dk
import os

# Lebar dan Tinggi Angka
LEBAR_ANGKA = 70
TINGGI_ANGKA = 125

# Set Angka Kartu yang akan diambil
card_check = "A"
folder_save = "A"
framePerImages = 10

# Set Direktori
video_path = "C:/Users/rayha/Pictures/Camera Roll/"
image_destination = "../dataset/"
os.chdir(image_destination + folder_save + "/")

# Membuka Video
cap = cv2.VideoCapture(video_path + card_check + ".mp4")    # Sesuaikan Video dengan Kartu
videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Frame Counter (tiap 10 frame akan diambil)
count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Gambar diproses dan disimpan tiap 10 frame
    if count % framePerImages == 0:
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Mengubah gambar ke grayscale
        blur = cv2.GaussianBlur(gray,(5,5),0) # Menghilangkan noise dengan blurring
        retval, thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY) # Mengubah gambar ke binary

        # Mencari contour di frame
        cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea,reverse=True)

        # Mengambil contour terbesar (dianggap sebagai kartu)
        image2 = frame.copy()

        # Jika tidak ada contour, lanjutkan
        if len(cnts) == 0:
            continue

        card = cnts[0]

        # Mencari sudut dari kartu
        peri = cv2.arcLength(card,True)
        approx = cv2.approxPolyDP(card,0.01*peri,True)
        pts = np.float32(approx)

        # Membuat bounding box dari kartu
        x,y,w,h = cv2.boundingRect(card)

        # Warp gambar kartu agar rata
        warp = dk.warpKartu(frame,pts,w,h)

        # Mengambil sudut kiri atas dari kartu (termasuk angka dan tingkat)
        corner = warp[0:84, 0:32]
        corner_zoom = cv2.resize(corner, (0,0), fx=4, fy=4)
        corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)
        retval, corner_thresh = cv2.threshold(corner_blur, 155, 255, cv2. THRESH_BINARY_INV)

        angka = corner_thresh[20:185, 0:128] # Mengambil angka dari sudut kiri atas saja
        angka_cnts, hier = cv2.findContours(angka, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(angka_cnts) < 1:
            continue
        angka_cnts = sorted(angka_cnts, key=cv2.contourArea,reverse=True)
        x,y,w,h = cv2.boundingRect(angka_cnts[0])

        # Crop angka dari sudut kiri atas
        angka_roi = angka[y:y+h, x:x+w]
        angka_sized = cv2.resize(angka_roi, (LEBAR_ANGKA, TINGGI_ANGKA), 0, 0) # Resize angka agar sama ukurannya
        final_img = angka_sized

        # Menyimpan gambar sesuai direktori
        cv2.imwrite(str(count // framePerImages) + ".jpg", final_img)

    # Increment Counter Frame
    if count >= videoFrames:
        break
    count += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
