import cv2
import time
import deteksiKartu as dk

# ! Settings Kamera Disini
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

# Source Video; 0 = Kamera ; path ke video
source = 0

# Set Video Settings
video = cv2.VideoCapture(source)
video.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
video.set(cv2.CAP_PROP_FPS, FRAME_RATE)

time_start = time.time()
bufferKartu = []

while(True):
    ret, image = video.read()

    # * PROSES GAMBAR
    thresh, edge = dk.imageBinaryEdge(image)
    contours, hierarchy = dk.contoursImage(edge)
    approx = dk.cornersImage(contours)

    # * LOOP DETEKSI KARTU
    kartu = []
    if len(contours) != 0:
        for i in range(len(contours)):
            kartu.append(dk.prosesKartu(image, contours[i], approx[i]))

        # ! Program hanya mendeteksi setiap 2 detik (agar tidak lagging)
        if time.time() - time_start > 3:
            bufferKartu = kartu
            for i in range(len(bufferKartu)):
                kartu[i].prediksi_angka = dk.prediksiKartu(kartu[i])
            time_start = time.time()

        for i in range(len(bufferKartu)):
            image = dk.drawKartu(image, bufferKartu[i])

    cv2.imshow("Video", image)    
    
    # * Program Keluar
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
video.release()

