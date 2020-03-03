# キャラクター画像の顔パーツ自動分割

## 概要
Live2D制作の前準備である、キャラクター画像の"顔パーツをそれぞれのレイヤーに分割する"という作業を効率化するためのツールを作成しました。

Live2Dを制作するためには、キャラクターイラストの目、口、髪の毛、などのパーツを予め別レイヤーに分割して用意しておかなくてはいけません。しかし、レイヤー分けしていない既存の画像をLive2Dに取り込んで動かしたい場合には、手作業でパーツごとにレイヤー分けをする必要があります。その作業はとても時間がかかり、面倒であるため、このツールにより、対話的前景領域抽出のgrabcutのアルゴリズムとアニメ画像の顔認識のプログラムをベースとして、目、鼻、口などの顔パーツを認識し、自動で切り取りを行うことを目的としました。

## 機能一覧
〇grabcut_auto - [grabcutのプログラム](https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py) をベースとして、以下の機能を追加しました。    
・顔パーツ(右目、左目、鼻、口、前髪、顔)を顔認識によって認識し、前景領域抽出を行う機能  
・マウスで描写することによって、前景領域と背景領域を選択する機能  
・出力画像をクロップして、それぞれのパーツごとに透過で保存する機  
・目の切り取りを選択後、白目部分も切り取り可能に  
・鼻の自動切り取り  

〇landmark.py - [アニメ顔用のランドマーク検出プログラム](https://github.com/kanosawa/anime_face_landmark_detection) をベースとして、検出した顔ランドマークの座標をもとに、それぞれのパーツごとの座標設定を追加しました。

## デモリールと使い方

![grabcutauto_demo](https://user-images.githubusercontent.com/61644695/75748092-0200b380-5d62-11ea-9dcf-58dcea4ffeb9.gif)
#### パーツごとにクロップして保存されたファイル
![保存されたファイル](file_result.png)

USAGE:　コマンドラインにて以下を入力します。抽出する顔パーツは"right_eye", "left_eye", "nose", "mouth", "face", "bangs"のうちのどれかを選んで入力します。    
    'python grabcut_auto.py <画像ファイル名> <抽出する顔パーツ>'   
    
    1.入力ウィンドウと出力ウィンドウが開きます。  
    2.入力ウィンドウ上で、抽出する顔パーツが矩形で囲まれます。  
    3.'n'を数回押すことによって前景抽出を行います。  
    4.以下のキーを入力し、前景領域と背景領域をマウスによる描写で選択し、'n'を押すことで抽出したい部分を調整することができます。  

Key '0' - 明確な背景領域をマウスで描写  
Key '1' - 明確な前景領域をマウスで描写  
Key '2' - 曖昧な背景領域をマウスで描写  
Key '3' - 曖昧な前景領域をマウスで描写  
key '4' - 白目を前景抽出する  
key '5' - "nose"を選択して鼻を前景抽出する  
Key 'n' - 前景抽出をアップデートする  
Key 'r' - リセット  
Key 's' - 出力を保存  
key 'q' - 終了
