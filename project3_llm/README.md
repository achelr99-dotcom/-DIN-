# 项目3：大模型推荐轻量化部署与RLHF方案设计

一句话：设计LoRA微调+4-bit量化+GGUF压缩+FastAPI全链路，并探索DPO实现偏好对齐。

- 压缩：8倍模型压缩（FP32→4-bit）
- Scaling：基座模型可替换LLaMA-3B/7B，数据千级→百万级
- RLHF：以用户点击/时长为隐式奖励，通过DPO优化生成策略
- 岗位匹配：高效训练/推理 + RLHF + LLM知识对齐 → OneRec加分项
