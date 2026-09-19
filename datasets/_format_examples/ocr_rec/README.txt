任务: ocr_rec

组织形式:
images/*.jpg + train.txt / val.txt + dict.txt(字符集)

train.txt 每行: 路径 <TAB> 文本
配置: Architecture.task=rec,Global.character_dict_path=.../dict.txt

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
