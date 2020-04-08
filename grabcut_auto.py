
#-*- cording:utf-8 -*-
#!/usr/bin/env python
'''
===============================================================================
Grubcutと顔認識を用いてイラストの顔パーツを自動で分割します。
USAGE:コマンドラインにて以下を入力してください
    python grabcut_auto.py <画像ファイル> <パーツ名>
    ＊パーツ名は"right_eye", "left_eye", "mouth", "face", "bangs", "nose"のどれかを入力
    ＊パーツ名を入力しなかった場合は、マウス操作で切り取りたい部分を選択
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
    BLUE = [255,0,0]        # 四角の色
    RED = [0,0,255]         # 曖昧な背景領域
    GREEN = [0,255,0]       # 曖昧な前景領域
    BLACK = [0,0,0]         # 明確な背景領域
    WHITE = [255,255,255]   # 明確な前景領域

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
    DRAW_PR_BG = {'color' : RED, 'val' : 2}

    # フラグ
    rect = (0,0,1,1)
    drawing = False         # 描写のフラグ
    rectangle = False       # 四角を描写するフラグ
    rect_over = False       # 四角を書き終えたフラグ
    rect_or_mask = 100      # 四角かマスクを選択するフラグ
    value = DRAW_FG         # 前景描写を初期化
    thickness = 3           # ブラシサイズ

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
        
            else:# パーツを指定しなかった場合、切り取りたい部分を手動で選択
                print("右クリック＆ドラッグで切り取りたいパーツを囲んでください\n")
        
        else:#　コマンド引数で画像ファイルを指定しなかった場合デフォルトの画像を使う
            print("画像ファイルを入力していないため、test.pngを読み込みます\n")
            print("正しい入力方法: python grabcut_auto.py <filename> \n")
            filename = 'test.png'

        #　画像読み込み
        self.img = cv.imread(cv.samples.findFile(filename))
        self.img2 = self.img.copy()                                # 画像をコピー
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # マスクを初期化
        self.output = np.zeros(self.img.shape, np.uint8)           #　出力画像
        self.rect_or_mask = 0

        if len(sys.argv) >= 3:#コマンドラインで顔パーツを入力した場合
            self.img = self.img2.copy()
            #　ランドマークで座標設定した顔パーツを四角で囲む
            cv.rectangle(self.img, (rect[0], rect[1]), (rect[2], rect[3]), self.BLUE, 2)
            self.rect = (min(rect[0], rect[2]), min(rect[1], rect[3]), abs(rect[0] - rect[2]), abs(rect[1] - rect[3]))
        else:
            pass

        self.rect_over = True
        print("画像に変化があるまで'n'を数回押してください")

        # 入力ウィンドウと出力ウィンドウの表示
        cv.namedWindow('output', cv.WINDOW_NORMAL)
        cv.namedWindow('input', cv.WINDOW_NORMAL)
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.img.shape[1]+10,90)

        while(1):

            #入力、出力画像の出力
            cv.imshow('output', self.output)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)
    
            #　キーボード入力
            if k == ord('q'): #終了
                break
            elif k == ord('0'): # 背景領域を選択
                print("左のマウスボタンで背景領域を選択してください\n")
                self.value = self.DRAW_BG
            elif k == ord('1'): # 前景領域を選択
                print("左のマウスボタンで前景領域を選択してください\n")
                self.value = self.DRAW_FG
            elif k == ord('2'): # 曖昧な背景領域を選択
                print("左のマウスボタンで曖昧な背景領域を選択してください\n")
                self.value = self.DRAW_PR_BG
            elif k == ord('3'): # 曖昧な前景領域を選択
                print("左のマウスボタンで曖昧な前景領域を選択してください\n")
                self.value = self.DRAW_PR_FG
            elif k == ord('4') :#白目を楕円で抽出
                print("白目部分を抽出します")
                
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
                print("鼻を抽出します")

                #円
                radius = int(np.linalg.norm(rect[0] - rect[2]) / 2) #半径
                center = int((rect[0]+rect[2])/2), int((rect[1]+rect[3])/2)
                
                cv.circle(self.img, center, radius,  self.value['color'], -1)
                cv.circle(self.mask, center, radius, self.value['val'], -1)

                self.value = self.DRAW_FG

            elif k == ord('s'): # 出力画像を保存
                bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
                bgr = cv.split(self.output)
                bgra = bgr + [mask2]
                self.output_alpha = cv.merge(bgra)
                
                if len(sys.argv) >= 3 and sys.argv[2] == "nose":
                    #鼻パーツ保存
                    img_key_name = filename.split(".",1)[0] + "_" + str(key) + ".png"
                    cv.imwrite(img_key_name,self.output_alpha[rect[1]-30:rect[3]+30,rect[0]-10:rect[2]+10])

                elif len(sys.argv) >= 3:
                    #顔パーツをコマンドラインで選択した場合、顔パーツをクロップして保存
                    img_key_name = filename.split(".",1)[0] + "_" + str(key) + ".png"
                    cv.imwrite(img_key_name,self.output_alpha[rect[1]:rect[3],rect[0]:rect[2]])

                else:#コマンドラインでパーツを選択しなかった場合、パーツに名前を付けて保存する
                    img_g = cv.cvtColor(self.output_alpha, cv.COLOR_BGR2GRAY)#グレースケールへ変換
                    x, y, w, h = cv.boundingRect(img_g)

                    print("パーツの名前：")
                    parts = input()

                    #顔パーツ保存
                    img_key_name = filename.split(".",1)[0] + "_" + parts + ".png"
                    cv.imwrite(img_key_name,self.output_alpha[y:y+h,x:x+w])
               
                print(" 出力結果を保存しました \n")
      
            elif k == ord('r'): # リセット
                print("リセットします \n")
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = self.DRAW_FG
                self.img = self.img2.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # マスクを初期化
                self.output = np.zeros(self.img.shape, np.uint8)           # 出力画像
            elif k == ord('n'): # 切り取り

                print("key 0-5を選択して描写することで、切り取る部分を調整することができます。\n")
                print("描写が終わったら'n'を押してください\n")
                try:
                    if (self.rect_or_mask == 0):         # 四角で囲んだ部分を前景領域抽出
                        bgdmodel = np.zeros((1, 65), np.float64)
                        fgdmodel = np.zeros((1, 65), np.float64)
                        cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif self.rect_or_mask == 1:         # マスクで前景領域抽出
                        bgdmodel = np.zeros((1, 65), np.float64)
                        fgdmodel = np.zeros((1, 65), np.float64)
                        cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
                except:
                    import traceback
                    traceback.print_exc()

            mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
            self.output = cv.bitwise_and(self.img2, self.img2, mask=mask2)

        print("終了")
        #cv.imwrite(filename.rstrip('.png') + '_mask.png', mask2)#マスクを保存

if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
