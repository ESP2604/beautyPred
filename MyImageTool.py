import cv2
import numpy as np

import re

def extract_filename_without_extension(img_file):
    """
    从文件路径中提取扩展名前直到第一个空格的字符串。

    :param img_file: 文件路径字符串
    :return: 返回扩展名前到空格的部分
    """
    # 找到最后一个 '.' 的位置
    dot_pos = img_file.rfind('.')
    if dot_pos == -1:
        return None  # 如果没有找到 '.'，返回 None

    # 提取扩展名前的部分
    file_name_part = img_file[:dot_pos]  # 获取扩展名前的字符串

    # 如果找到了空格，返回扩展名前到空格之间的部分
    return file_name_part.split(' ')[-1]


def create_centered_resized_image(image):
    # 讀取圖片
    
    if image is None:
        print("圖片為空")
        return None
    image = image.copy()

    # 獲取圖片尺寸
    height, width = image.shape[:2]

    # 計算新的尺寸 (等比例縮小兩倍)
    new_width = width //2
    new_height = height //2

    # 縮小圖片
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 創建空白圖像
    blank_image = np.full((new_height * 2, new_width * 2, 3), 255, dtype=np.uint8)

    # 計算縮小後圖片在空白圖像上的起始坐標（使其置中）
    start_x = (blank_image.shape[1] - new_width) // 2
    start_y = (blank_image.shape[0] - new_height) // 2

    # 將縮小後的圖片放置在空白圖像上
    blank_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image

    # return blank_image
    return blank_image , start_y ,  new_height, start_x ,  new_width
# 以下代碼只有在直接運行此腳本時才會執行
if __name__ == "__main__":
    img_file = "TestImage/FC2-PPV-1932742 980円【フルサイズ４K動画】リアル妖精！人気ロリTi○Toker！身バレ厳禁！スタイル抜群8.5頭身ロリ美少女♪生ハメ中出し！18歳JD1ねるちゃん第1章_本編 9acf60ca.jpg"
    filename = extract_filename_without_extension(img_file)
    print(filename)  # 输出: 9acf60ca
    
    # 使用函數
    image_path = r'TestImage\6513cfcd89ee5.png'  # 替換為您的圖片路徑
    image_path =r'TestImage\saika_kawakita__official\saika_kawakita__official_1634213912_2684303308468052771_49779563569.jpg'
    # centered_image = create_centered_resized_image(image_path)
    # if centered_image is not None:
    #     cv2.imshow('Centered Resized Image', centered_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()