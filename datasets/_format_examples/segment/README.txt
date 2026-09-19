任务: segment

组织形式:
images/<split>/*.jpg + labels/<split>/*.txt(多边形)

每行: cls x1 y1 x2 y2 ...(归一化多边形,≥3 点)
或用位图掩码: masks/<split>/0001.png(255=前景)

(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)
