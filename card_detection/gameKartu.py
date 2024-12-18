import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import time
import pygame
from pygame import mixer
import copy
import deteksiKartu as dk

# * SETTING KAMERA
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

# Source Video; 0 = Kamera ; path ke video
sourceVideo = 0
sourceAudio = "Casino.mp3"
winAudio = "Win.mp3"
loseAudio = "Lose.mp3"

# Set Text
fontpath_casino = "Casino.ttf"
fontpath_cards = "CardsFont.ttf"
font_casino = ImageFont.truetype(fontpath_casino, 40)
font_casino_state = ImageFont.truetype(fontpath_casino, 70)
font_cards = ImageFont.truetype(fontpath_cards, 40)

black = (0,0,0,0) # Preset warna
red = (0,0,200,0)
white = (255,255,255,0)
purple = (200,0,200,0)
green = (0,170,0,0)

# Set Video
video = cv2.VideoCapture(sourceVideo)
video.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
video.set(cv2.CAP_PROP_FPS, FRAME_RATE)

# Set Pygame
pygame.init()
mixer.music.load(sourceAudio)
mixer.music.play(-1)
winSFX = mixer.Sound(winAudio)
loseSFX = mixer.Sound(loseAudio)

# Timer, buffer, and game
frameCounter = 0
detectionTimer = time.time()
winCheckCounter = 0 # Jika state Win atau Lose, maka akan menunggu 60 frame untuk mengecek input
bufferKartu = [] # Buffer kartu, agar tidak lagging
bufferKartuCheckWin = [] # Buffer kartu untuk mengecek win

# Wins
playerWins = 0
computerWins = 0

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
    
    # ! Program hanya mendeteksi setiap 1 detik (agar tidak lagging)
    if time.time() - detectionTimer > 1:
        bufferKartu = kartu
        for i in range(len(bufferKartu)):
            kartu[i].prediksi_angka = dk.prediksiKartu(bufferKartu[i])
        bufferKartu = dk.sortKartu(bufferKartu)
        detectionTimer = time.time()

    # * GAMBAR CONTOUR DAN ANGKA KARTU
    for i in range(len(bufferKartu)):
        image = dk.drawKartu(image, bufferKartu[i])

    # * LOOP PERMAINAN
    game = dk.game(bufferKartu)

    # ! Untuk Win/Lose/Draw Screen
    imagePilSave = Image.fromarray(copy.deepcopy(image))
    drawSave = ImageDraw.Draw(imagePilSave)

    # * UI DAN POIN PEMAIN
    image = cv2.line(image, (0, IM_HEIGHT//2), (IM_WIDTH//2 + 200, IM_HEIGHT//2), (190, 0, 0), 10)
    image = cv2.line(image, (IM_WIDTH//2 + 200, 0), (IM_WIDTH//2 + 200, IM_HEIGHT), (190, 0, 0), 10)

    image = cv2.line(image, (0, IM_HEIGHT//2), (IM_WIDTH//2 + 200, IM_HEIGHT//2), (255, 255, 255), 2)    
    image = cv2.line(image, (IM_WIDTH//2 + 200, 0), (IM_WIDTH//2 + 200, IM_HEIGHT), (255, 255, 255), 2)

    # * Image Conversion (Untuk text dengan PIL)
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    # * STATE PEMAIN
    if len(game.player_state) > 1:
        textSize_P = cv2.getTextSize(game.player_state, cv2.FONT_HERSHEY_PLAIN, 3, 4)[0]
        textX_P = ((IM_WIDTH//2 + 200) - (textSize_P[0]))//2
        textY_P = (IM_HEIGHT//4) - (textSize_P[1]//2)
        draw.text((textX_P, textY_P), game.player_state, font=font_casino_state, fill=red,
                  stroke_width=4, stroke_fill=white)

    if len(game.computer_state) > 1:
        textSize_C = cv2.getTextSize(game.computer_state, cv2.FONT_HERSHEY_PLAIN, 3, 4)[0]
        textX_C = ((IM_WIDTH//2 + 200) - textSize_C[0])//2
        textY_C = ((IM_HEIGHT*3)//4) - (textSize_C[1]//2)
        draw.text((textX_C, textY_C), game.computer_state, font=font_casino_state, fill=red,
                  stroke_width=4, stroke_fill=white)

    if len(game.dealer_state) > 1:
        textSize_D = cv2.getTextSize(game.dealer_state, cv2.FONT_HERSHEY_PLAIN, 3, 4)[0]
        textX_D = (IM_WIDTH//2 + 200)+((IM_WIDTH - (IM_WIDTH//2 + 200))//2) - textSize_D[0]//2
        textY_D = (IM_HEIGHT - textSize_D[1])//2
        draw.text((textX_D, textY_D), game.dealer_state, font=font_casino_state, fill=red,
                  stroke_width=4, stroke_fill=white)

    # * POIN PEMAIN
    draw.text((15, 15), "Player", font=font_casino, fill=white,
              stroke_width=3, stroke_fill=black)
    draw.text((15, 50), "pts: " + str(game.point_player), font=font_casino, fill=white,
              stroke_width=3, stroke_fill=black)

    draw.text((15, IM_HEIGHT - 40), "Computer", font=font_casino, fill=purple,
              stroke_width=3, stroke_fill=white)
    draw.text((15, IM_HEIGHT - 75), "pts: " + str(game.point_computer), font=font_casino, fill=purple,
              stroke_width=3, stroke_fill=white)

    draw.text((IM_WIDTH - 150, IM_HEIGHT - 40), "Dealer", font=font_casino, fill=green,
              stroke_width=3, stroke_fill=white)
    draw.text((IM_WIDTH - 150, IM_HEIGHT - 75), "pts: " + str(game.point_dealer), font=font_casino, fill=green,
              stroke_width=3, stroke_fill=white)

    # * KARTU PEMAIN
    temp_kartu_player = []
    temp_kartu_computer = []
    temp_kartu_dealer = []

    for i in range(len(game.player)):
        temp_kartu_player.append(game.player[i].angka)
        draw.text((IM_WIDTH//2 + 160 - (i*50), IM_HEIGHT//2 - 60), game.player[i].prediksi_angka, font=font_cards, fill=white,
                  stroke_width=2, stroke_fill=black)

    for i in range(len(game.computer)):
        temp_kartu_computer.append(game.computer[i].angka)
        draw.text((IM_WIDTH//2 + 160 - (i*50), IM_HEIGHT - 50), game.computer[i].prediksi_angka, font=font_cards, fill=purple, 
                  stroke_width=2, stroke_fill=white)

    for i in range(len(game.dealer)):
        temp_kartu_dealer.append(game.dealer[i].angka)
        draw.text((IM_WIDTH//2 + 220, 20 + (i*50)), game.dealer[i].prediksi_angka, font=font_cards, fill=green,
                  stroke_width=2, stroke_fill=white)
        
    # * WINS PEMAIN
    draw.text((IM_WIDTH//2 + 30, 10), "Wins: " + str(playerWins), font=font_casino, fill=white,
              stroke_width=3, stroke_fill=black)
    draw.text((IM_WIDTH//2 + 30, IM_HEIGHT//2 + 15), "Wins: " + str(computerWins), font=font_casino, fill=purple,
                stroke_width=3, stroke_fill=white)

    # * CEK WINS
    if frameCounter % 10 == 0:
        bufferKartuCheckWin = bufferKartu
        bufferKartuCheckWin = dk.sortKartu(bufferKartuCheckWin)

    if game.player_state == "Win" or game.player_state == "Lose" or game.computer_state == "Win" or game.computer_state == "Lose" or game.player_state == "Draw" or game.computer_state == "Draw":
        if bufferKartuCheckWin == bufferKartu:
            if winCheckCounter == 20:
                mixer.music.pause()
                if game.player_state == "Win" and game.computer_state == "Lose":
                    drawSave.text((IM_WIDTH//2 - 110, IM_HEIGHT//2 - 60), "You Win!", font=font_casino_state, fill=white,
                            stroke_width=3, stroke_fill=black)
                    mixer.Sound.play(winSFX)
                    
                elif game.player_state == "Lose" and game.computer_state == "Win":
                    drawSave.text((IM_WIDTH//2 - 110, IM_HEIGHT//2 - 60), "You Lose!", font=font_casino_state, fill=white,
                            stroke_width=3, stroke_fill=black)
                    mixer.Sound.play(loseSFX)
                else:
                    drawSave.text((IM_WIDTH//2-120, IM_HEIGHT//2 - 60), "Draw!", font=font_casino_state, fill=white,
                            stroke_width=3, stroke_fill=black)
                    mixer.Sound.play(winSFX)
                    
                drawSave.text((IM_WIDTH//2 - 200, IM_HEIGHT//2 + 20), "Press Y to save state!", font=font_casino, fill=white,
                            stroke_width=3, stroke_fill=black)
                drawSave.text((IM_WIDTH//2 - 220, IM_HEIGHT//2 + 60), "Press N if state is false!", font=font_casino, fill=white,
                            stroke_width=3, stroke_fill=black)
                drawSave.text((IM_WIDTH//2 - 185, IM_HEIGHT//2 + 100), "Press Q to quit game!", font=font_casino, fill=white,
                            stroke_width=3, stroke_fill=black)
                
                cv2.imshow("Video", np.array(imagePilSave))
                
                state = cv2.waitKey(0)
                if state == ord("y") or state == ord("Y"):
                    if game.player_state == "Win" and game.computer_state == "Lose":
                        playerWins += 1
                    elif game.player_state == "Lose" and game.computer_state == "Win":
                        computerWins += 1
                    mixer.music.unpause()
                elif state == ord("n") or state == ord("N"):
                    mixer.music.unpause()
                elif state == ord("q") or state == ord("Q"):
                    break
                else:
                    print("Invalid Input, continuing...")
                    pass
                winCheckCounter = 0
            else:
                winCheckCounter += 1
        else:
            winCheckCounter = 0
    
    # Show Video
    image = np.array(image_pil)
    cv2.imshow("Video", image)
    frameCounter += 1

    # Program Keluar
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
video.release()
