任务: plate_rec

组织形式:
images/*.jpg(48×168)+ train.txt / val.txt

每行: 路径 c1..c7 颜色号
c: 字符表下标(pytorchx/nn/plate.py::PLATE_CHARSET,78 项,0=CTC blank)
颜色: 0黑 1蓝 2绿 3白 4黄;双层牌设 Train.dataset.double_plate=true

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
