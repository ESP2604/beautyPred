from openpyxl import Workbook
from openpyxl.drawing.image import Image

from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage
import io, os
from openpyxl import Workbook ,load_workbook
class MyExcelData:
    def __init__(self):
        self.score = None
        self.image_name= None
        self.image_path= None
        self.avatar_path= None
        self.row = None
        self.padding_score= None
        self.padding_avatar_path = None

def resize_and_compress_image_in_memory(input_image_path, scale_factor, quality=85):
    # 加载原始图像
    img = PILImage.open(input_image_path)
    
    # 如果图像是RGBA模式，则转换为RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # 计算新的尺寸
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    
    # 调整图像尺寸
    img_resized = img.resize((new_width, new_height), PILImage.ANTIALIAS)
    
    # 保存调整后的图像到内存
    img_byte_arr = io.BytesIO()
    img_resized.save(img_byte_arr, format='JPEG', quality=quality)
    
    # 将内存中的图像数据定位回起始位置
    img_byte_arr.seek(0)
    
    return img_byte_arr
    
def insertImage(ws, img_path ,anchor ,row, scale_factor =1,quality=20):
        # 设置单元格的宽度和高度
    desired_cell_width =  50 # 根据需要调整
    desired_cell_height = 150  # 根据需要调整
    ws.column_dimensions[anchor[0]].width = desired_cell_width
    # ws.column_dimensions[get_column_letter(3)].width = desired_cell_width
    # ws.column_dimensions[get_column_letter(4)].width = desired_cell_width
    # ws.column_dimensions[get_column_letter(5)].width = desired_cell_width
    # for row in range(1, 5):  # 假设您要调整前四行的高度
    ws.row_dimensions[row].height = desired_cell_height

    # 加载并添加图片
    # img_path = "path_to_your_image.png"  # 替换为您的图片路径
    
    img_byte_arr = resize_and_compress_image_in_memory(img_path, 
        scale_factor=scale_factor, quality=quality)  # 根据需要调整scale_factor和quality
    img = Image(img_byte_arr)

    # 调整图片大小以适应单元格
    cell_ratio = desired_cell_height / desired_cell_width
    img_ratio = img.height / img.width
    if img_ratio > cell_ratio:
        # 图片更高，以高度为准进行缩放
        scale = (desired_cell_height*2) / img.height
    else:
        # 图片更宽，以宽度为准进行缩放
        scale = (desired_cell_width*3) / img.width

    img.width *= scale
    img.height *= scale

    # 将图片放置到特定的位置（例如，A1单元格）
    img.anchor = anchor

    # 将图片添加到工作表
    ws.add_image(img)
    



# filePath 檔案路徑
# count 最重試次數
def reTrySave( count, wb,filePath= 'output.xlsx',):
    reTry = 0
    while reTry < count:
        try:
            filename = os.path.basename(filePath).replace(".xlsx", "")
            parentdir = os.path.dirname(filePath)
            # 保存工作簿
            wb.save(f'{os.path.join(parentdir, filename)}({reTry}).xlsx')
            break  # 如果保存成功，退出循环
        except PermissionError:
            reTry += 1  # 增加重试次数
        except Exception as e:
            print(f"保存失败：{e}")
            return  # 如果遇到其他异常，停止重试并退出

    if reTry >= count:
        print("重试次数已达上限，保存失败。")

filename = 'output'
sheet_name = 'sheet'
wb = None
ws = None
if os.path.exists(filename):
    wb = load_workbook(filename)
else:
    wb = Workbook()
# 检查工作簿中是否存在名为 "sheetname" 的工作表
if sheet_name in wb.sheetnames:
    # print("工作表已经存在")
    ws = wb[sheet_name]
else:
# print("工作表不存在")
    ws = wb.create_sheet(sheet_name)
def writeExcel(data,i,):

    # 写入分数和图像名称
    ws[f'A{i}'] = data.score
    ws[f'B{i}'] = data.image_name
    ws[f'F{i}'] = data.padding_score
    # 插入图像
    if os.path.exists(data.image_path):
        insertImage(ws, data.image_path,f'C{data.row}', data.row ,scale_factor =1,quality=60)
    # 插入头像
    if os.path.exists(data.avatar_path):
        insertImage(ws, data.avatar_path,f'D{data.row}' , data.row,scale_factor =1,quality=60)
    # 插入padding過後的頭向
    if os.path.exists(data.padding_avatar_path):
        insertImage(ws, data.padding_avatar_path,f'E{data.row}' , data.row,scale_factor =1,quality=60)

    return wb
