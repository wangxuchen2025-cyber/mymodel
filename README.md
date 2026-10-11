# MARS-CD

**Map-Anchored Relational Streaming Change Detection** — 面向地理空间约束的多源遥感建筑物实例级变化监测（倾斜无人机视频 × 历史 SHP / 地图 / DOM / 历史倾斜视频）。

* 研究方案（论文式）：[`docs/research_proposal.md`](docs/research_proposal.md)
* 逐层递进推导与迭代记录（v1 → v2 → v3，含失败尝试与修正）：[`docs/research_iteration.md`](docs/research_iteration.md)

## 网络（v3）

| 模块 | 作用 | 代码 |
|---|---|---|
| A 多源地理先验（离线） | DOM/地图可靠性门控融合、SHP token、SHP 边界栅格、T1 图集与高度后验 | `marscd/models/ortho_encoder.py` |
| B 倾斜编码器 | 流式 FPN 骨干 | `marscd/models/oblique_backbone.py` |
| v2-I 结构头 | 高度无关的接地线 GCL + 屋顶边；高度扫描；contact snake | `marscd/models/structure.py` |
| C CARR | 偏置 Kalman + BEV 相关 + 影像空间 GCL↔SHP 渲染比较 + 特征度量 LM | `marscd/models/registration.py` |
| D PASID | 棱柱锚定流式实例解码（竖直角线可变形采样、图集记忆） | `marscd/models/prism_decoder.py` |
| v2-III PTA | 棱柱纹理图集（表面坐标索引的视角校正记忆） | `marscd/models/atlas.py` |
| E RIGM | 关系不变的一元图匹配 | `marscd/models/graph_matcher.py` |
| v2-II HSR | 一阶 → 二阶 QAP（神经渐进指派 + SC²）→ 运动场 EM（变化 = 离群类） | `marscd/models/relational_assignment.py` |
| F 变化推理 | 留一法局部刚体残差分解 + 图集比较 | `marscd/models/change_head.py` |
| v3 证据决策 | 主观逻辑 + 新颖度折扣：拆除 vs 未观测 | `marscd/models/evidential.py` |

## 运行

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests                      # 17 项单元 / 机制 / 端到端测试
PYTHONPATH=. python tools/train_synthetic.py --iters 60     # 合成数据端到端冒烟训练
PYTHONPATH=. python tools/benchmark_latency.py --config deploy --half   # 在 4090 上测逐模块时延
```

合成数据（`marscd/data/synthetic.py`）只用于检验各条代码路径与机制，真实数据的训练需要离线空三得到的逐帧真值位姿（见方案 §4、§11）。
