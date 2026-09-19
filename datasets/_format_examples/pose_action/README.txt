任务: pose_action(骨架行为)

组织形式:
seqs/*.npy + train.txt / val.txt

每行: 路径 行为类别号
npy: (T, V, C) 或 (C,T,V);C 默认 2(x,y);V 默认 17(COCO)
配置: transform.clip_len 统一时长,Backbone.num_joints,Head.num_classes

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
