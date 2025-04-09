## Quick Start

1. 安装依赖
   `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

2. 运行
   `python main.py`

3. 说明
   - 输入：视频 ID 列表
   - 输出：视频特征（.pkl 文件）
     fea/fakesv/
     ├── metainfo.json # 视频元数据（ID/帧率/标注等）
     ├── fakesv_segment_duration.json # 片段持续时间统计
     ├── preprocess_ocr/
     │ ├── sam/ # OCR 版面特征（每视频的 SAM 检测结果）
     │ └── ocr_phrase_fea.pkl # OCR 文本语义特征
     ├── preprocess_text/
     │ ├── sem_text_fea.pkl # 文本语义特征（768 维）
     │ └── emo_text_fea.pkl # 文本情感特征
     ├── preprocess_audio/ # 音频情感特征（.pkl 文件）
     └── preprocess_visual/ # 视觉特征（TransNetV2 片段划分结果）
