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

4. 数据增强对齐
   原始视频
   │
   ▼
   metainfo.json → 视频元数据 → 数据过滤 → FakingRecipe_Dataset
   │
   ├─ preprocess_visual/ → 视觉特征 → raw_visual_frames
   ├─ preprocess_audio/ → 音频特征 → raw_audio_emo
   ├─ preprocess_text/ → 文本特征 → all_phrase_semantic_fea
   └─ preprocess_ocr/ → OCR 特征 → ocr_phrase_fea
   │
   ▼
   数据增强对齐 → collate_fn_FakeingRecipe → 模型输入

5. 模型训练
   graph TD
   A[模式选择] -->|train| B[初始化三阶段数据]
   A -->|inference_test| C[加载单测试文件]
   B --> D[配置优化器/调度器]
   D --> E[训练循环]
   E --> F[验证与早停判断]
   C --> G[加载预训练参数]
   G --> H[批量推理]
   H --> I[生成预测文件]
