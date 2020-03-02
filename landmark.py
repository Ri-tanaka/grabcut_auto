import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw
from CFA import CFA

class Landmark():

    # コンストラクタ
    def __init__(self, input_file):
        input_img_name = input_file
        self.num_landmark = 24
        self.img_width = 128
        checkpoint_name = 'checkpoint_landmark_191116.pth.tar'

        face_detector = cv2.CascadeClassifier('lbpcascade_animeface.xml')
        landmark_detector = CFA(output_channel_num=self.num_landmark + 1, checkpoint_name=checkpoint_name).cpu()

        normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        train_transform = [transforms.ToTensor(), normalize]
        train_transform = transforms.Compose(train_transform)

        img = cv2.imread(input_img_name)
        faces = face_detector.detectMultiScale(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        for x_, y_, w_, h_ in faces:

            # 顔のサイズを調整
            self.x = max(x_ - w_ / 8, 0)
            rx = min(x_ + w_ * 9 / 8, img.width)
            self.y = max(y_ - h_ / 4, 0)
            by = y_ + h_
            self.w = rx - self.x
            self.h = by - self.y


            # 画像変換
            img_tmp = img.crop((self.x, self.y, self.x+self.w, self.y+self.h))
            img_tmp = img_tmp.resize((self.img_width, self.img_width), Image.BICUBIC)
            img_tmp = train_transform(img_tmp)
            img_tmp = img_tmp.unsqueeze(0).cpu()

            # ヒートマップを推定
            self.heatmaps = landmark_detector(img_tmp)
            self.heatmaps = self.heatmaps[-1].cpu().detach().numpy()[0]

    def get_landmark(self, key):
        res = np.empty((0, 2))
        for i in range(self.num_landmark):
            heatmaps_tmp = cv2.resize(self.heatmaps[i], (self.img_width, self.img_width), interpolation=cv2.INTER_CUBIC)
            landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
            landmark_y = landmark[0] * self.h / self.img_width
            landmark_x = landmark[1] * self.w / self.img_width

            if key == "right_eye" and (i == 10 or i == 11 or i == 12 or i == 13 or i == 14):
                res = np.append(res, [[self.x + landmark_x, self.y + landmark_y]], axis = 0)

            if key == "left_eye" and (i == 15 or i == 16 or i == 17 or i == 18 or i == 19):
                res = np.append(res, [[self.x + landmark_x, self.y + landmark_y]], axis = 0)
      

            #顔パーツ(右目、左目、口、顔、前髪、鼻)の座標を取得
            if i == 0:
                r_ear_rx,r_ear_ry = landmark_x,landmark_y
            elif i == 2:
                l_ear_lx,l_ear_ly = landmark_x,landmark_y
            elif i == 9:
                nose_x, nose_y = landmark_x, landmark_y
            elif i == 10:
                r_eye_rx,r_eye_ry = landmark_x,landmark_y
            elif i == 11:
                r_eye_ux,r_eye_uy = landmark_x,landmark_y
            elif i == 12:
                r_eye_lx,r_eye_ly = landmark_x,landmark_y
            elif i == 13:
                r_eye_dx,r_eye_dy = landmark_x,landmark_y
            elif i == 15:
                l_eye_rx,l_eye_ry = landmark_x,landmark_y
            elif i == 16:
                l_eye_ux,l_eye_uy = landmark_x,landmark_y
            elif i == 17:
                l_eye_lx,l_eye_ly = landmark_x,landmark_y
            elif i == 18:
                l_eye_dx,l_eye_dy = landmark_x,landmark_y
            elif i == 20:
                mouth_rx,mouth_ry = landmark_x,landmark_y
            elif i == 21:
                mouth_ux,mouth_uy = landmark_x,landmark_y
            elif i == 22:
                mouth_lx,mouth_ly = landmark_x,landmark_y
            elif i == 23:
                mouth_dx,mouth_dy = landmark_x,landmark_y

        res = res.astype('int64')

        #　認識する顔パーツの長方形座標
        if key == "right_eye":#右目を認識して矩形を自動で作成する
            rect = (int(self.x + r_eye_rx - 15) , int(self.y + r_eye_uy - 8),int(self.x + r_eye_lx + 8),int(self.y + r_eye_dy + 5))
        elif key == "left_eye":#左目を認識
            rect = (int(self.x + l_eye_rx - 8) , int(self.y + l_eye_uy - 8),int(self.x + l_eye_lx + 15),int(self.y + l_eye_dy + 5))
        elif key == "mouth":#口
            rect = (int(self.x + mouth_rx - 8), int(self.y + mouth_uy - 8),int(self.x + mouth_lx + 8),int(self.y + mouth_dy + 5))
        elif key == "face":#顔
            rect = (int(self.x),int(self.y),int(self.x + self.w),int(self.y + self.h))
        elif key == "bangs":#前髪
            rect = (int(self.x),int(self.y),int(self.x + self.w),int(self.y + self.h/2 + 20))
        elif key == "nose":#鼻
            rect =(int(self.x + nose_x - 10),int(self.y + nose_y + 10),int(self.x + nose_x + 10),int(self.y + nose_y - 10))
            

        return res, rect
