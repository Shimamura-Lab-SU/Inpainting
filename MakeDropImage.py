from PIL import Image
import numpy as np
import os
import random
import math
#
import cv2
from copy import copy

class Rectangle(object):
    def __init__(self, x1, y1, x2, y2):
        if not is_rectangle(x1, y1, x2, y2):
            raise ValueError("Coordinates are invalid.\n" +
                             "Rectangle" + str((x1, y1, x2, y2)))
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

def is_rectangle(x1, y1, x2, y2):
    return x1 <= x2 and y1 <= y2

#交差判定
def has_intersect(a, b):
    return max(a.x1, b.x1) <= min(a.x2, b.x2) \
           and max(a.y1, b.y1) <= min(a.y2, b.y2)


def Get_Pathlists(soutai_dir):
    file_list = []

    path_list = []
    #実行ファイルの絶対パスの取得
    path = os.path.abspath(soutai_dir)
    for root, dirs, files in os.walk(path):
        for fileo in files:
            fulpath = root + "\\" + fileo
            path_list = np.append(path_list,fulpath)
            file_list = np.append(file_list,fileo)
    return([path_list,file_list])

if __name__ == "__main__":
    [path_list,file_list] = Get_Pathlists("./test/input")
    print(file_list)
    print(path_list)

#マスクを埋め込む座標

    px = 0
    py = 0

#とりあえず1つ画像を読み込む


#各ファイルに対して画像処理を行う

    #穴の個数の定義
    hole_N = 3
    hole_size = 64
    hole_interval = 8 #この数値の分intersect(交差判定)の当たり判定を大きくする
    padding = 16 # 端に発生させすぎないよう端からこのピクセル分は発生確率を0にする

    CenterHall = True #中央に穴を作るか

    iter1 = 0
    maskpath = "test/whitemask.jpg"
    mask_raw = Image.open(maskpath).convert('RGB')
    #マスク画像のサイズを補正
    mask = mask_raw.resize((hole_size, hole_size))
    p = 0


    for path1 in path_list:
        RectangleList = []
        p = p + 1
        image = Image.open(path1).convert('RGB')
        image_orgin = copy(image)
        w, h = image.size 
        mask_image = Image.new('1',(w,h),0) #同一サイズの黒塗りを用意

        #エッジフィルタ後の画像
        image_array = np.asarray(image)
        gray_array  = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        canny_img_array = cv2.Canny(gray_array, 50, 110)

        if(CenterHall == False) :
            #ランダムにNコの穴をあけるモード
            for iter1 in range(hole_N) :
                try_num = 0
                while True:
                    
                    px = random.randint(0 + padding,image.size[0] - hole_size - padding)
                    py = random.randint(0 + padding,image.size[1] - hole_size - padding)
                    if(len(RectangleList) == 0):
                        break
                    N = 0
                    for rect in RectangleList:
                 #px,pyが他のあなと重複していないかを調べる
                 #右上と左下のチェック
                    
                        if has_intersect(Rectangle(px - hole_interval,py - hole_interval,px+hole_size + hole_interval,py+hole_size + hole_interval),rect) == True:
                            print("intersect:" + str(iter1) + ":::locate::" + path1)
                        
                        else :
                            N = N + 1
                        if N >= iter1:
                            break
                        try_num = try_num + 1
                        if(try_num >= 100):
                            break
                rect_tmp = Rectangle(px - hole_interval,py - hole_interval,px+hole_size + hole_interval,py+hole_size + hole_interval)
                RectangleList.append(rect_tmp)
                image.paste(mask,(px,py))
                mask_image.paste(mask,(px,py))
        else :
            #自動的に中心から穴を作るモード
            #pは中心座標を取得する
            px =  math.floor(image.size[0] / 2) - math.floor(mask.size[0] / 2)
            py =  math.floor(image.size[1] / 2) - math.floor(mask.size[1] / 2)
            image.paste(mask,(px,py))
            mask_image.paste(mask,(px,py))
        #穴あき画像の出力
        image.save("test/output/" + str(p) + ".jpg")
        image_orgin.save("test/output_orgin/" + str(p) + ".jpg")
        mask_image.save("test/output_mask/" + str(p) + ".jpg")
        #エッジの出力
        canny_img = Image.fromarray(canny_img_array)
        canny_img.save("test/output_edge/" + str(p) + ".jpg")
##filepath = "test/test01.jpg"
#image = Image.open(filepath).convert('RGB')
#image.paste(mask, (px,py))
#image.save("test/crop.jpg")


           