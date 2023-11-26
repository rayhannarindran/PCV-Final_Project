import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model CNN
MODEL_ANGKA = load_model("./models/modelDeteksiRankKartu")
MODEL_TINGKAT = load_model("./models/modelDeteksiSuitKartu")

KELAS_ANGKA = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q']
KELAS_TINGKAT = ['Clubs', 'Diamonds', 'Hearts', 'Spades']

# ! Pastikan AS terakhir dalam penyortiran, sisanya bebas
SORT = ['0', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'K', 'Q', 'A']

# Adaptive thresholding
BKG_THRESH = 70

# Dimensi Kartu
LEBAR_KARTU = 200
TINGGI_KARTU = 300

# Area Kartu
AREA_KARTU_MAKS = 120000
AREA_KARTU_MIN = 25000

# Lebar dan Tinggi Ujung Kartu (Lokasi Angka dan Tingkat)
LEBAR_UJUNG = 32
TINGGI_UJUNG = 84

# Lebar dan Tinggi Angka
LEBAR_ANGKA = 70
TINGGI_ANGKA = 125

# Lebar dan Tinggi Tingkat
LEBAR_TINGKAT = 70
TINGGI_TINGKAT = 100

# * Kelas Kartu
class Kartu:
    def __init__(self):
        self.contour = []
        self.corners = [] # Titik sudut kartu
        self.warp = []
        self.corner = [] # Angka dan Tingkat Kartu
        self.center = []
        self.angka = []
        self.tingkat = []
        self.prediksi_angka = "0"
        self.posisi_area = "Unknown"
        # self.prediksi_tingkat = "0"

# * Kelas Game
class gameState:
    def __init__(self):
        self.player = []
        self.computer = []
        self.dealer = []
        self.point_player = 0
        self.point_computer = 0
        self.point_dealer = 0
        self.player_state = ""
        self.computer_state = ""
        self.dealer_state = ""

# * Inisialisasi Kelas Kartu
def prosesKartu(image, contour, approx, IM_WIDTH, IM_HEIGHT):
    kartuQ = Kartu()
    kartuQ.contour = contour
    kartuQ.corners = approx

    if len(kartuQ.corners) < 3:
        return kartuQ

    # ! Set titik tengah kartu, untuk menentukan area kartu diletakkan
    M = cv2.moments(kartuQ.contour)
    kartuQ.center = [int(M['m10']/M['m00']), int(M['m01']/M['m00'])]

    if kartuQ.center[1] < IM_HEIGHT//2 and kartuQ.center[0] < IM_WIDTH//2 + 200:
        kartuQ.posisi_area = "Player"
    elif kartuQ.center[1] > IM_HEIGHT//2 and kartuQ.center[0] < IM_WIDTH//2 + 200:
        kartuQ.posisi_area = "Computer"
    else:
        kartuQ.posisi_area = "Dealer"

    x,y,w,h = cv2.boundingRect(kartuQ.contour)
    kartuQ.warp = warpKartu(image, kartuQ.corners, w, h)

    cornerKartu = kartuQ.warp[0:TINGGI_UJUNG, 0:LEBAR_UJUNG]
    cornerKartuZoomed = cv2.resize(cornerKartu, (0,0), fx=4, fy=4)
    kartuQ.corner = cornerKartuZoomed

    cornerKartuZoomed = cv2.cvtColor(cornerKartuZoomed, cv2.COLOR_BGR2GRAY)
    cornerKartuZoomed = cv2.adaptiveThreshold(cornerKartuZoomed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY_INV, 11, 2)

    angkaCropped = cornerKartuZoomed[20:170, 0:120]
    tingkatCropped = cornerKartuZoomed[171:300, 0:120]

    angkaCroppedCnts, _ = cv2.findContours(angkaCropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    angkaCroppedCnts = sorted(angkaCroppedCnts, key=cv2.contourArea, reverse=True)

    if len(angkaCroppedCnts) != 0:
        xa, ya, wa, ha = cv2.boundingRect(angkaCroppedCnts[0])
        angkaCropped = angkaCropped[ya:ya+ha, xa:xa+wa]
        angkaCropped = cv2.resize(angkaCropped, (LEBAR_ANGKA, TINGGI_ANGKA), fx=0, fy=0)
        angkaCropped = np.repeat(angkaCropped[:, :, np.newaxis], 3, axis=2)
        kartuQ.angka = angkaCropped

    tingkatCroppedCnts, _ = cv2.findContours(tingkatCropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tingkatCroppedCnts = sorted(tingkatCroppedCnts, key=cv2.contourArea, reverse=True)

    if len(tingkatCroppedCnts) != 0:
        xt, yt, wt, ht = cv2.boundingRect(tingkatCroppedCnts[0])
        tingkatCropped = tingkatCropped[yt:yt+ht, xt:xt+wt]
        tingkatCropped = cv2.resize(tingkatCropped, (LEBAR_TINGKAT, TINGGI_TINGKAT), fx=0, fy=0)
        tingkatCropped = np.repeat(tingkatCropped[:, :, np.newaxis], 3, axis=2)
        kartuQ.tingkat = tingkatCropped

    return kartuQ

# * Prediksi kartu dengan model CNN
def prediksiKartu(kartu):
    prediksi_angka = "0"
    # prediksi_tingkat = "0"

    if len(kartu.angka) != 0 and len(kartu.tingkat) != 0:
        angka = kartu.angka
        # tingkat = kartu.tingkat

        angkaArray = tf.keras.utils.img_to_array(angka)
        angkaArray = np.expand_dims(angkaArray, axis=0)
        prediksi_angka = MODEL_ANGKA.predict(angkaArray, verbose=0)
        prediksi_angka = KELAS_ANGKA[np.argmax(prediksi_angka)]

        # tingkatArray = tf.keras.utils.img_to_array(tingkat)
        # tingkatArray = np.expand_dims(tingkatArray, axis=0)
        # prediksi_tingkat = MODEL_TINGKAT.predict(tingkatArray, verbose=0)
        # prediksi_tingkat = KELAS_TINGKAT[np.argmax(prediksi_tingkat)]

    return prediksi_angka #, prediksi_tingkat

# * Menggambar contour dan angka kartu
def drawKartu(image, kartu):
    image = cv2.drawContours(image, [kartu.contour], 0, (255,0,0), 2)
    angkaKartu = kartu.prediksi_angka
    posisiTextAngka = (kartu.center[0], kartu.center[1] - 10)
    # tingkatKartu = kartu.prediksi_tingkat
    # posisiTextTingkat = (kartu.center[0] - 50, kartu.center[1] + 10)
    image = cv2.putText(image, angkaKartu, posisiTextAngka, cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 6)
    image = cv2.putText(image, angkaKartu, posisiTextAngka, cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)
    # image = cv2.putText(image, tingkatKartu, posisiTextTingkat, cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 6)
    # image = cv2.putText(image, tingkatKartu, posisiTextTingkat, cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)
    return image

# * Loop Game
def game(kartu):
    permainan = gameState()

    if len(kartu) == 0:
        return permainan

    for i in range(len(kartu)):
        if kartu[i].posisi_area == "Player":
            permainan.player.append(kartu[i])
        elif kartu[i].posisi_area == "Computer":
            permainan.computer.append(kartu[i])
        else:
            permainan.dealer.append(kartu[i])

    # ! Sortir kartu agar AS dihitung terakhir (untuk menentukan apakah AS bernilai 1 atau 11)
    permainan.player = sorted(permainan.player, key=lambda kartu: SORT.index(kartu.prediksi_angka))
    permainan.computer = sorted(permainan.computer, key=lambda kartu: SORT.index(kartu.prediksi_angka))
    permainan.dealer = sorted(permainan.dealer, key=lambda kartu: SORT.index(kartu.prediksi_angka))

    # * Melakukan penjumlahan poin pada tiap pemain
    for i in range(len(permainan.player)):
        if permainan.player[i].prediksi_angka == "A":
            if permainan.point_player + 11 > 21:
                permainan.point_player += 1
            else:
                permainan.point_player += 11
        elif permainan.player[i].prediksi_angka == "J" or permainan.player[i].prediksi_angka == "Q" or permainan.player[i].prediksi_angka == "K":
            permainan.point_player += 10
        else:
            permainan.point_player += int(permainan.player[i].prediksi_angka)

    for i in range(len(permainan.computer)):
        if permainan.computer[i].prediksi_angka == "A":
            if permainan.point_computer + 11 > 21:
                permainan.point_computer += 1
            else:
                permainan.point_computer += 11
        elif permainan.computer[i].prediksi_angka == "J" or permainan.computer[i].prediksi_angka == "Q" or permainan.computer[i].prediksi_angka == "K":
            permainan.point_computer += 10
        else:
            permainan.point_computer += int(permainan.computer[i].prediksi_angka)

    for i in range(len(permainan.dealer)):
        if permainan.dealer[i].prediksi_angka == "A":
            if permainan.point_dealer + 11 > 21:
                permainan.point_dealer += 1
            else:
                permainan.point_dealer += 11
        elif permainan.dealer[i].prediksi_angka == "J" or permainan.dealer[i].prediksi_angka == "Q" or permainan.dealer[i].prediksi_angka == "K":
            permainan.point_dealer += 10
        else:
            permainan.point_dealer += int(permainan.dealer[i].prediksi_angka)

    # * State menang dan kalah pemain
    if len(permainan.dealer) >= 2:
        if permainan.point_dealer > 21:
            permainan.dealer_state = "Dealer BUST"
            
            if permainan.point_player <= 21:
                permainan.player_state = "Player Won"
            else:
                permainan.player_state = "Player BUST"

            if permainan.point_computer <= 21:
                permainan.computer_state = "Computer Won"
            else:
                permainan.computer_state = "Computer BUST"

        else:
            if permainan.point_player > 21:
                permainan.player_state = "Player BUST"
            elif permainan.point_player > permainan.point_dealer:
                permainan.player_state = "Player Won"
            else:
                permainan.player_state = "Player Lost"
            
            if permainan.point_computer > 21:
                permainan.computer_state = "Computer BUST"
            elif permainan.point_computer > permainan.point_dealer:
                permainan.computer_state = "Computer Won"
            else:
                permainan.computer_state = "Computer Lost"

    elif permainan.point_player < 21 and permainan.point_computer < 21:
        if permainan.point_computer < 17:
            permainan.computer_state = "Computer Hit"
        elif permainan.point_computer < permainan.point_player and permainan.point_computer < 21:
            permainan.computer_state = "Computer Hit"
        else:
            permainan.computer_state = "Computer Stand"

    else:
        if permainan.point_player == 21:
            permainan.player_state = "Player Won"
        elif permainan.point_player > 21:
            permainan.player_state = "Player BUST"
        
        if permainan.point_computer == 21:
            permainan.computer_state = "Computer Won"
        elif permainan.point_computer > 21:
            permainan.computer_state = "Computer BUST"
    
    # * State antara Player dan Computer
    if len(permainan.dealer) >= 2 or permainan.point_player == 21 or permainan.point_computer == 21:
        if permainan.computer_state == "Computer Lost" or permainan.computer_state == "Computer BUST":
            if permainan.player_state == "Player Won":
                permainan.player_state = "Player Won to COM"
                permainan.computer_state = "COM Lost to Player"
        elif permainan.player_state == "Player Lost" or permainan.player_state == "Player BUST":
            if permainan.computer_state == "Computer Won":
                permainan.player_state = "Player Lost to COM"
                permainan.computer_state = "COM Won to Player" 
        else:
            if permainan.point_computer == permainan.point_player:
                permainan.player_state = "Player Draw to COM"
                permainan.computer_state = "COM Draw to Player"
            elif permainan.point_computer > permainan.point_player:
                permainan.player_state = "Player Lost to COM"
                permainan.computer_state = "COM Won to Player"
            else:
                permainan.player_state = "Player Won to COM"
                permainan.computer_state = "COM Lost to Player"

    return permainan

# * Ubah Image menjadi Binary dan Cari Edge
def imageBinaryEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 3, 3)

    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(image, 100, 200)

    return thresh, edge

# * Mencari Contour dari Kartu
def contoursImage(edges):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Hilangkan contour di luar ukuran kartu
    if len(contours) != 0:
        for i in range(len(contours)-1, -1, -1):
            if cv2.contourArea(contours[i]) > AREA_KARTU_MAKS or cv2.contourArea(contours[i]) < AREA_KARTU_MIN:
                contours.pop(i)

    return contours, hierarchy

# * Mencari titik sudut tiap kartu
def cornersImage(contours):
    approx = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx.append(cv2.approxPolyDP(contour, 0.02 * peri, True))
    return approx

# * Meratakan kartu agar dapat di prediksi angka
def warpKartu(image, pts, w, h):   
    pts_source = np.zeros((4, 2), dtype = "float32")
	
    s = np.sum(pts, axis = 2)
    kiri_atas = pts[np.argmin(s)]
    kanan_bawah = pts[np.argmax(s)]
	
    diff = np.diff(pts, axis = -1)
    kanan_atas = pts[np.argmin(diff)]
    kiri_bawah = pts[np.argmax(diff)]

    if w <= 0.6*h:
        pts_source[0] = kiri_atas
        pts_source[1] = kanan_atas
        pts_source[2] = kiri_bawah
        pts_source[3] = kanan_bawah

    if w >= 1.3*h:
        pts_source[0] = kiri_atas
        pts_source[1] = kiri_bawah
        pts_source[2] = kanan_atas
        pts_source[3] = kanan_bawah

    if w > 0.6*h and w < 1.3*h:
        if pts[1][0][1] <= pts[3][0][1]:
            pts_source[0] = pts[1][0]
            pts_source[1] = pts[0][0]
            pts_source[2] = pts[3][0]
            pts_source[3] = pts[2][0]

        if pts[1][0][1] > pts[3][0][1]:
            pts_source[0] = pts[0][0]
            pts_source[1] = pts[3][0]
            pts_source[2] = pts[2][0]
            pts_source[3] = pts[1][0]

    pts_dest = np.float32([[0, 0], [LEBAR_KARTU-1, 0],[LEBAR_KARTU-1, TINGGI_KARTU-1], [0, TINGGI_KARTU-1]])

    matrix = cv2.getPerspectiveTransform(pts_source, pts_dest)
    warp = cv2.warpPerspective(image, matrix, (LEBAR_KARTU, TINGGI_KARTU))

    return warp
