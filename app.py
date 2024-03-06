
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
        # file.save(path_str)
        # np_file, file_paths = loadRequestFile(file)
        # np_file, file_paths = convert_image(images, file)
        # formatted_str = "%s size: %s" % (file.filename, (IMAGE_DIM))
        # print(formatted_str)
        
        
        wb = None
        for idx, item in enumerate(result):
            print(f'result[{idx}].score = {item.score}')
            wb = ex(item, idx+1)
        excelTool.reTrySave(10, wb)
        
        
        return jsonify('{ok:"ok"}')
    else:
        return "Failed to upload file"

@app.route('/nsfw/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    if file:

        # file.save(path_str)
        np_file, file_paths = nfswTool.loadRequestFile(file)
       
        formatted_str = "%s size: %s" % (file.filename, (IMAGE_DIM))
        print(formatted_str)
        probs = predict.classify_nd(model, np_file)
        # os.remove(path_str)
        d = dict(zip(file_paths ,probs))
        return jsonify(d)
    else:
        return "Failed to upload file"
        
def test():
    pathStr = u'NonDRM_[FHD].mp4.jpg'
    pathStr = u'[久久美剧www.jjmjtv.com]蛇蝎美人.Femme.Fatales.S01E01.Chi_Eng.720p.HDTV.x264.AC3.iNT-ShinY b1b9120f.jpg'
    pathStr =u'[TSKS][Babysitter][E002(720P)][KO_CN] 9e540669.jpg'
    # pathStr = u'[喜爱夜蒲]Lan.Kwaii.Fong.2011.BluRay.720p.2Audio.x264.AC3-CnSCG[国粤双语2.7G] 9ae37afe.jpg'
    # pathStr = u'5fa79a42dc13a7029f0986fdbc239b62-720p 2e27f45a.jpg'
    # pathStr = u'[SHANA]1 (1) 71504577.jpg'
    pathStr = u'[OPFansMaplesnow][one_piece][2022][Special][01][1080p] 5555a6ba.jpg'
    file = Image.open(pathStr)

  
if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8000)
    # app.run(port=8080)
    
   
# {'drawings': 6.272925929806661e-07, 'hentai': 0.0023030538577586412, 'neutral': 0.002372985938563943, 'porn': 0.9951841235160828, 'sexy': 0.00013913711882196367}}
# drawings 是否為動畫
# hentai 是否變態
# sqlite CREATE TABLE "web_scraping_movie" (
# 	"id"	INTEGER,
# 	"title"	TEXT,
# 	"original_title"	TEXT,
# 	"sort_title"	TEXT,
# 	"custom_rating"	TEXT,
# 	"series"	TEXT,
# 	"mpaa"	TEXT,
# 	"studio"	TEXT,
# 	"year"	TEXT,
# 	"outline"	TEXT,
# 	"plot"	TEXT,
# 	"runtime"	TEXT,
# 	"director"	TEXT,
# 	"poster"	TEXT,
# 	"thumb"	TEXT,
# 	"fanart"	TEXT,
# 	"maker"	TEXT,
# 	"label"	TEXT,
# 	"num"	TEXT,
# 	"premiered"	TEXT,
# 	"release_date"	TEXT,
# 	"release"	TEXT,
# 	"website"	TEXT,
# 	"video_hash_path"	TEXT NOT NULL,
#     "drawings" REAL,
#     "hentai" REAL,
#     "neutral" REAL,
#     "porn" REAL,
#     "sexy" REAL,
# 	PRIMARY KEY("id" AUTOINCREMENT)
# ); 
# ALTER TABLE web_scraping_movie ADD COLUMN drawings REAL;
# ALTER TABLE web_scraping_movie ADD COLUMN hentai REAL;
# ALTER TABLE web_scraping_movie ADD COLUMN neutral REAL;
# ALTER TABLE web_scraping_movie ADD COLUMN porn REAL;
# ALTER TABLE web_scraping_movie ADD COLUMN sexy REAL;
#  新增 drawings ,  hentai , neutral, porn, sexy 四個Real類型