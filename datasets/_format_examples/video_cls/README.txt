任务: video_cls(视频行为)

组织形式:
视频文件(*.mp4)或帧目录 + train.txt / val.txt

每行: 路径(视频或帧目录) 行为类别号
采样: transform.num_segments × frames_per_seg 帧均匀分段;image_size/crop_size 控制分辨率

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
