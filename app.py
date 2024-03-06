
from flask import Flask, request ,jsonify

from PIL import Image
from waitress import serve
import excelTool, os, cv2
from FaceRatingTool import FaceRating , CropFace, faceRatingPre ,crop_image_into_parts  ,AvatarImage
IMAGE_DIM = 224   # required/default image dimensionality

# FLASK 判斷色情圖片接口
app = Flask(__name__)

cropFace = CropFace()
faceRating = FaceRating()
def ex(actorImage, index):
    # 将结果图像保存
    save_path = os.path.join('tmp/', f'info_tmp{index}.jpg')
    # save_path=""
    cv2.imwrite(save_path, actorImage.originalImage)
    # 将结果图像保存
    save_info_path =  os.path.join('tmp/', f'original_tmp{index}.jpg')
    # save_path=""
    cv2.imwrite(save_path, actorImage.infoImage)


    # score = padding_score
    data = excelTool.MyExcelData()
    data.score = actorImage.score
    data.padding_score = 0
    data.image_path = save_path
    data.image_name = os.path.basename(save_path)
    data.avatar_path = save_path
    data.padding_avatar_path = ''
    data.row = index
    print(f'{index}分数[1~5]: original:{data.score} padding:{data.padding_score}')
    return excelTool.writeExcel(data, index)
   
    

@app.route('/beautyPrediction/uploadCropColxRow', methods=['POST'])
def uploadCropColxRow():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    col = request.form.get('col')
    col = int(col)
    row = request.form.get('row')
    row = int(row)
    if file.filename == '':
        return "No selected file"

    if file:
        images =   crop_image_into_parts(file,col, row)
        result = []
        
        for item in images:
            # print(item.shape)
            faces = cropFace.detect(item)
            if(len(faces) > 1 or len(faces) == 0): continue
            result.append (faces[0])

        for item in result:
            # 捕捉到的臉部標記
            item.facerect()
        faceRatingPre(faceRating, result)

        
        
        wb = None
        for idx, item in enumerate(result):
            print(f'result[{idx}].score = {item.score}')
            wb = ex(item, idx+1)
        excelTool.reTrySave(10, wb)
        
        
        return jsonify('{ok:"ok"}')
    else:
        return "Failed to upload file"

        

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8000)

    
