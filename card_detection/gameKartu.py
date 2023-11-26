import cv2
import numpy as np
import time
import pygame
from pygame import mixer
import deteksiKartu as dk

# * SETTING KAMERA
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

# Source Video; 0 = Kamera ; path ke video
sourceVideo = 0
sourceAudio = "Casino.wav"

# Set Video
video = cv2.VideoCapture(sourceVideo)
video.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
video.set(cv2.CAP_PROP_FPS, FRAME_RATE)

# Set Pygame
pygame.init()
mixer.music.load(sourceAudio)
mixer.music.play(-1)

# Timer dan Buffer
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
            kartu.append(dk.prosesKartu(image, contours[i], approx[i], IM_WIDTH, IM_HEIGHT))
        
    # ! Program hanya mendeteksi setiap 3 detik (agar tidak lagging)
    if time.time() - time_start > 3:
        bufferKartu = kartu
        for i in range(len(bufferKartu)):
            kartu[i].prediksi_angka = dk.prediksiKartu(kartu[i])
        time_start = time.time()

    # * GAMBAR CONTOUR DAN ANGKA KARTU
    for i in range(len(bufferKartu)):
        image = dk.drawKartu(image, bufferKartu[i])

    # * LOOP PERMAINAN
    game = dk.game(bufferKartu)

    # * STATE PEMAIN
    if len(game.player_state) > 1:
        textSize_P = cv2.getTextSize(game.player_state, cv2.FONT_HERSHEY_PLAIN, 3, 4)[0]
        textX_P = ((IM_WIDTH//2 + 200) - (textSize_P[0]))//2
        textY_P = (IM_HEIGHT//4) - (textSize_P[1]//2)
        image = cv2.putText(image, game.player_state, (textX_P, textY_P), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
        image = cv2.putText(image, game.player_state, (textX_P, textY_P), cv2.FONT_HERSHEY_PLAIN, 3, (100,255,100), 4)

    if len(game.computer_state) > 1:
        textSize_C = cv2.getTextSize(game.computer_state, cv2.FONT_HERSHEY_PLAIN, 3, 4)[0]
        textX_C = ((IM_WIDTH//2 + 200) - textSize_C[0])//2
        textY_C = ((IM_HEIGHT*3)//4) - (textSize_C[1]//2)
        image = cv2.putText(image, game.computer_state, (textX_C, textY_C), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
        image = cv2.putText(image, game.computer_state, (textX_C, textY_C), cv2.FONT_HERSHEY_PLAIN, 3, (100,255,100), 4)

    if len(game.dealer_state) > 1:
        textSize_D = cv2.getTextSize(game.dealer_state, cv2.FONT_HERSHEY_PLAIN, 3, 4)[0]
        textX_D = (IM_WIDTH//2 + 200)+((IM_WIDTH - (IM_WIDTH//2 + 200))//2) - textSize_D[0]//2
        textY_D = (IM_HEIGHT - textSize_D[1])//2
        image = cv2.putText(image, game.dealer_state, (textX_D, textY_D), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
        image = cv2.putText(image, game.dealer_state, (textX_D, textY_D), cv2.FONT_HERSHEY_PLAIN, 3, (100,255,100), 4)

    # * UI DAN POIN PEMAIN
    image = cv2.line(image, (0, IM_HEIGHT//2), (IM_WIDTH//2 + 200, IM_HEIGHT//2), (0, 255, 0), 3)
    image = cv2.line(image, (IM_WIDTH//2 + 200, 0), (IM_WIDTH//2 + 200, IM_HEIGHT), (0, 255, 0), 3)

    image = cv2.putText(image, "Player", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
    image = cv2.putText(image, "Player", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 4)
    image = cv2.putText(image, str(game.point_player), (110, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
    image = cv2.putText(image, str(game.point_player), (110, 110), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 4)

    image = cv2.putText(image, "Computer", (50, IM_HEIGHT - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
    image = cv2.putText(image, "Computer", (50, IM_HEIGHT - 20), cv2.FONT_HERSHEY_PLAIN, 3, (100,100,255), 4)
    image = cv2.putText(image, str(game.point_computer), (150, IM_HEIGHT - 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
    image = cv2.putText(image, str(game.point_computer), (150, IM_HEIGHT - 70), cv2.FONT_HERSHEY_PLAIN, 3, (100,100,255), 4)

    image = cv2.putText(image, "Dealer", (IM_WIDTH - 170, IM_HEIGHT - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
    image = cv2.putText(image, "Dealer", (IM_WIDTH - 170, IM_HEIGHT - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,100), 4)
    image = cv2.putText(image, str(game.point_dealer), (IM_WIDTH - 110, IM_HEIGHT - 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 12)
    image = cv2.putText(image, str(game.point_dealer), (IM_WIDTH - 110, IM_HEIGHT - 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,100), 4)

    # Show Video
    cv2.imshow("Video", image)

    # Uncomment untuk lihat deteksi kartu
    # if (len(bufferKartu) > 0):
    #     if (isinstance(bufferKartu[0].angka, np.ndarray) and len(bufferKartu[0].angka.shape) == 3):
    #         cv2.imshow("Threshold", bufferKartu[0].angka)

    # Program Keluar
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
video.release()
