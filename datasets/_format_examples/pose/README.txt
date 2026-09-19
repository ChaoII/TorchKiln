任务: pose / plate_det

组织形式:
images/<split>/*.jpg + labels/<split>/*.txt

每行: cls cx cy w h 然后 K 组 (px py v);v: 0=未标注 1=遮挡 2=可见
配置: Architecture.Head.kpt_shape=[17,3]
车牌四角点(无可见性): kpt_shape=[4,2] -> cls cx cy w h p1x p1y p2x p2y p3x p3y p4x p4y

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
