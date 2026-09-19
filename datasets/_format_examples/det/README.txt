任务: detect / obb

组织形式:
images/<split>/*.jpg + labels/<split>/*.txt + data.yaml + train.txt/val.txt

每行: cls cx cy w h   (全部归一化 0~1)
旋转框(obb): cls cx cy w h angle(弧度;配置加 Train.dataset.box_format=xywhr)

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
