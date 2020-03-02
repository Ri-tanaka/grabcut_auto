#-*- cording:utf-8 -*-
#!/usr/bin/env python
'''
===============================================================================
Grubcutを使ってイラストの顔パーツを自動で分割します。
USAGE:
    python grabcut_auto.py <画像ファイル名> <パーツ>
    パーツには"right_eye", "left_eye", "mouth", "face", "bangs", "nose"
    のうちのどれかを入力してください。
README FIRST:
    1.入力ウィンドウと出力ウィンドウが開きます。
    2.入力ウィンドウ上で、前景抽出をしたい領域をマウスで四角に囲みます。
    3.'n'を数回押すことによって前景抽出を行います。
    4.以下のキーを入力し、'n'を押すことでより明確に前景抽出を行うことができます。

Key '0' - 明確な背景領域を選択
Key '1' - 明確な前景領域を選択
Key '2' - 曖昧な背景領域を選択
Key '3' - 曖昧な前景領域を選択
key '4' - 白目を前景抽出する
key '5' - "nose"を選択して鼻を前景抽出する
Key 'n' - 前景抽出をアップデートする
Key 'r' - リセット
Key 's' - 出力を保存
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import os

from landmark import Landmark


class App():
    BLUE = [255,0,0]        # rectangle color
    RED = [0,0,255]         # PR BG
    GREEN = [0,255,0]       # PR FG
    BLACK = [0,0,0]         # sure BG
    WHITE = [255,255,255]   # sure FG

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
    DRAW_PR_BG = {'color' : RED, 'val' : 2}

    # setting up flags
    rect = (0,0,1,1)
    drawing = False         # flag for drawing curves
    rectangle = False       # flag for drawing rect
    rect_over = False       # flag to check if rect drawn
    rect_or_mask = 100      # flag for selecting rect or mask mode
    value = DRAW_FG         # drawing initialized to FG
    thickness = 3           # brush thickness

    #マウス操作
    def onmouse(self, event, x, y, flags, param):
        # 四角形の範囲指定
        if event == cv.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x,y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img2.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

        # 前景・背景指定時のマウス操作
        if event == cv.EVENT_LBUTTONDOWN:
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                # 描写
                cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                # 描写
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                # 描写
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)


    def run(self):

        #　コマンド引数で画像ファイル名を指定した場合
        if len(sys.argv) >= 2:
            filename = sys.argv[1] # for drawing purposes
            workDir = filename[:filename.rfind(".")]
            if not os.path.exists(workDir):
                os.mkdir(workDir)

            #　画像ファイルからランドマーク取得
            landmark = Landmark(filename)

            if len(sys.argv) == 3: #　コマンド引数で認識するパーツを指定した場合
                key = sys.argv[2]
                lm, rect = landmark.get_landmark(key)
            
            else:#指定しなかった場合 デフォルトで右目認識
                key = "right_eye"
                lm, rect = landmark.get_landmark("right_eye")
             
        #　コマンド引数で画像ファイルを指定しなかった場合デフォルトの画像を使う
        else:
            print("No input image given, so loading default image, lena.jpg \n")
            print("Correct Usage: python grabcut.py <filename> \n")
            filename = 'syasyaki.png'

        #　画像読み込み
        self.img = cv.imread(cv.samples.findFile(filename))
        self.img2 = self.img.copy()                                # a copy of original image
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
        self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown

        self.img = self.img2.copy()
        cv.rectangle(self.img, (rect[0], rect[1]), (rect[2], rect[3]), self.BLUE, 2)
        self.rect = (min(rect[0], rect[2]), min(rect[1], rect[3]), abs(rect[0] - rect[2]), abs(rect[1] - rect[3]))
        
        self.rect_or_mask = 0

        self.rect_over = True
        print(" Now press the key 'n' a few times until no further change \n")

        # input and output windows
        cv.namedWindow('output', cv.WINDOW_NORMAL)
        cv.namedWindow('input', cv.WINDOW_NORMAL)
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.img.shape[1]+10,90)

        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        SkinLow = np.array([0,30,60])
        SkinHigh = np.array([40,150,255])
        hsv_mask = cv.inRange(hsv, SkinLow, SkinHigh)
        #img_color = cv.bitwise_and(self.img,self.img, hsv_mask)
 
       
        print(" Instructions: \n")
        print(" Draw a rectangle around the object using right mouse button \n")

        while(1):

            #入力、出力画像の出力
            cv.imshow('output', self.output)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)

            # key bindings
            if k == 27:         # esc to exit
                break
            elif k == ord('q'): #終了
                break
            elif k == ord('0'): # BG drawing
                print(" mark background regions with left mouse button \n")
                self.value = self.DRAW_BG
            elif k == ord('1'): # FG drawing
                print(" mark foreground regions with left mouse button \n")
             
                '''
                # 顔から白部分を認識
                for i in rect: 
                    dst = cv.inRange(self.img,(250,250,250),(255,255,255))
                
                # 白部分を前景指定
                for ex in range(dst.shape[0]):
                    for ey in range(dst.shape[1]):
                        if dst[ex,ey] == 255:
                            cv.circle(self.img, (ey,ex), self.thickness, self.value['color'], -1)
                            cv.circle(self.mask, (ey,ex), self.thickness, self.value['val'], -1)
                '''
                self.value = self.DRAW_FG
            elif k == ord('2'): # PR_BG drawing
                self.value = self.DRAW_PR_BG
            elif k == ord('3'): # PR_FG drawing
                self.value = self.DRAW_PR_FG
            elif k == ord('4') :#白目を楕円で抽出
                
                #楕円
                long = int(np.linalg.norm(lm[0] - lm[2]) / 2) #直径/2
                short = int(np.linalg.norm(lm[1] - lm[3]) / 2) #短径/2

                center = (int(abs(lm[0][0] + lm[2][0]) / 2), int(abs(lm[0][1] + lm[2][1]) / 2))
                tmp = (int(abs(lm[1][0] + lm[3][0]) / 2), int(abs(lm[1][1] + lm[3][1]) / 2))
                center = (int((center[0]+tmp[0])/2), int((center[1]+tmp[1])/2)) #中心
                cv.ellipse(self.img, center, (long, short), 0, 0, 360, self.value['color'], -1)
                cv.ellipse(self.mask, center, (long, short), 0, 0, 360, self.value['val'], -1)

                self.value = self.DRAW_FG

            elif k == ord('5') and sys.argv[2] == "nose" :#鼻を丸で抽出

                #円
                radius = int(np.linalg.norm(rect[0] - rect[2]) / 2) #半径
                center = int((rect[0]+rect[2])/2), int((rect[1]+rect[3])/2)
                
                cv.circle(self.img, center, radius,  self.value['color'], -1)
                cv.circle(self.mask, center, radius, self.value['val'], -1)

                self.value = self.DRAW_FG

            elif k == ord('s'): # save image
                bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
                bgr = cv.split(self.output)
                bgra = bgr + [mask2]
                self.output_alpha = cv.merge(bgra)
                
                if sys.argv[2] == "nose":
                    #鼻パーツ保存
                    img_key_name = filename.split(".",1)[0] + "_" + str(key) + ".png"
                    cv.imwrite(img_key_name,self.output_alpha[rect[1]-30:rect[3]+30,rect[0]-10:rect[2]+10])

                else:
                    #顔パーツ保存
                    img_key_name = filename.split(".",1)[0] + "_" + str(key) + ".png"
                    cv.imwrite(img_key_name,self.output_alpha[rect[1]:rect[3],rect[0]:rect[2]])

                
                print(" Result saved as image \n")

            elif k == ord('r'): # reset everything
                print("resetting \n")
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = self.DRAW_FG
                self.img = self.img2.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
                self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown
            elif k == ord('n'): # segment the image
                print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n""")
                try:
                    if (self.rect_or_mask == 0):         # grabcut with rect
                        bgdmodel = np.zeros((1, 65), np.float64)
                        fgdmodel = np.zeros((1, 65), np.float64)
                        cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif self.rect_or_mask == 1:         # grabcut with mask
                        bgdmodel = np.zeros((1, 65), np.float64)
                        fgdmodel = np.zeros((1, 65), np.float64)
                        cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
                except:
                    import traceback
                    traceback.print_exc()

            mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
            self.output = cv.bitwise_and(self.img2, self.img2, mask=mask2)

        print('Done')
        cv.imwrite(filename.rstrip('.png') + '_mask.png', mask2)#マスクを保存

if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
