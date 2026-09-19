任务: semantic

组织形式:
images/<split>/*.jpg + masks/<split>/*.png(同名)+ train.txt/val.txt

masks: uint8 单通道,像素值=类别索引;忽略像素由 ignore_index 指定(默认 255)

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
