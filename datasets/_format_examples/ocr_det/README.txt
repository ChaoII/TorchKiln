任务: ocr_det

组织形式:
images/<任意名>.jpg  +  train.txt / val.txt(内联 JSON)

train.txt 每行: 图片相对路径 <TAB> [{"transcription": 文本, "points": [[x,y]×4](像素), "difficult": false}]
配置: Architecture.task=det(data_dir=datasets/_format_examples/ocr_det)

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
