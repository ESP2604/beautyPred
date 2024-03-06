
from flask import Flask, request ,jsonify
from waitress import serve
from excelTool import  ExcelSaver
from FaceRatingTool import FaceRating , CropFace, faceRatingPre ,crop_image_into_parts
IMAGE_DIM = 224   # required/default image dimensionality
import time
# FLASK 判斷色情圖片接口
app = Flask(__name__)

cropFace = CropFace()
faceRating = FaceRating()
excelSaver = ExcelSaver()
def ex(actorImage, index):
    data =[ 
        {'column':'A','row':index, 'value': actorImage.score},
         {'column':'B', 'row':index,'value':actorImage.infoImage },
        {'column':'C', 'row':index,'value':actorImage.faceImage },
    ]
    print(f'{index}分数[1~5]: original:{actorImage.score} padding:{actorImage.score}')
    excelSaver.writeExcel(data, index)
   
    

@app.route('/beautyPrediction/uploadCropColxRow', methods=['POST'])
def uploadCropColxRow():
    start_time = time.time()
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
            # 把臉橋正
            item.correct_face_tilt()
        faceRatingPre(faceRating, result)

        
        excelSaver.load_or_create_workbook()
        for idx, item in enumerate(result):
            print(f'result[{idx}].score = {item.score}')
            ex(item, idx+1)
        excelSaver.reTrySave(10)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"執行時間: {elapsed_time} 秒")
        return jsonify('{ok:"ok"}')
    else:
        return "Failed to upload file"

        

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)

    
