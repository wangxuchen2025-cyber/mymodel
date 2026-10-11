# MARS-CD 逐层递进研究记录（v1 → v2 → v3）

> 课题：面向地理空间约束的多源遥感建筑物实例级变化监测
> 核心难点：倾斜多视角之间利用空间关系实现实例匹配关联
> 本文档记录**推导链条、失败尝试与修正依据**；最终方案的论文式描述见 `research_proposal.md`。
> 文中所有“实验”均为**机制级 oracle / 合成实验**（`tests/` 中可复现），用于检验每个机制的推导前提，不代表真实数据精度。

---

## 0. 已确认的问题设定

| 项目 | 设定（两轮问答确认） |
|---|---|
| 时相 | 前时相 T1 = 历史无人机倾斜视频 + 历史 SHP + 历史栅格地图 + 历史卫星/航摄 DOM；后时相 T2 = 当前无人机倾斜视频（不区分 T1 内部各源的具体年代） |
| 位姿元数据 | 仅消费级 GNSS + 云台角（米级位置偏差、度级航向偏差） |
| SHP | 仅轮廓，无高度属性，无 DEM |
| 历史影像 | 正射 DOM + 栅格地图 |
| 输出 | 实例级 + 地理坐标（可回写 SHP） |
| 变化类别 | 新增、拆除、扩张、收缩（扩张/收缩仅指平面占地） |
| 部署 | 服务器单卡 RTX 4090/3090，1080p 视频，≥10 FPS 流式 |

---

## 1. Layer 0：问题本质——一个联合推断问题

**符号.** 世界系 $W$（局部 ENU / 投影坐标）。T2 位姿 $\mathcal{T}=\{T_t\}$，GNSS 读数 $\mathcal{G}=\{g_t\}$，低频偏置 $b_t$。T2 建筑实例 $X^2=\{x_i=(F_i,z_i,h_i,\mathcal{A}_i)\}$（足迹多边形、基底高程、高度、表面外观）；T1 实例 $X^1=\{x_j\}$（$F_j$ 来自 SHP）。对应 $\sigma$：$X^1\rightleftharpoons X^2$ 的部分单射（含“出生/死亡”）。观测 $\mathcal{Z}=\{I_t\}$。

$$
p(\mathcal{T},X^2,\sigma\mid\mathcal{Z},\mathcal{G},X^1)\;\propto\;
\underbrace{\prod_t p(I_t\mid X^2,T_t)}_{\text{倾斜观测模型}}\;
\underbrace{p(\mathcal{T}\mid\mathcal{G})}_{\text{GNSS 偏置}}\;
\underbrace{p(X^2\mid X^1,\sigma)}_{\text{变化模型}}\;
\underbrace{p(\sigma\mid X^1,X^2)}_{\text{空间关系先验}}
$$

**由分解直接导出的模块：**

| 因子 | 结构 | 对应模块 |
|---|---|---|
| $p(\mathcal{T}\mid\mathcal{G})$ | $T_t=g_t\oplus b_t$，$b_t$ 随机游走 | CARR：偏置 Kalman + 相关体 + LM |
| $p(I_t\mid X^2,T_t)$ | 按“是否依赖高度”可分解（见 v2-I） | 结构头（GCL/屋顶边）+ PASID |
| $p(X^2\mid X^1,\sigma)$ | 匹配对 = 局部刚体（配准残差）∘ 稀疏非刚体（真实变化）；未匹配 = 出生/死亡 | RRD + 变化分类 |
| $p(\sigma\mid\cdot)$ | 一阶（外观/形状）+ 二阶（空间关系保持）+ 全局（一致运动场）的 MRF | RIGM → HSR |

**推论（贯穿全部迭代的主线）：** 配准、关联、变化检测是同一后验的三个侧面——
*变化 = 联合“配准-关联”问题中的离群类（outlier class）*。这一视角决定了：
① 配准必须对变化鲁棒（变化区域不参与配准）；② 关联的 dustbin 质量就是出生/死亡（新增/拆除）后验；
③ 平面扩张/收缩是去除局部刚体运动后的残差。

---

## 2. v1：基线框架（MARS-CD v1）

非对称“离线 T1 / 在线 T2”范式：

* **A 多源地理先验编码（离线）**：DOM/地图 → 可靠性门控融合 BEV；SHP → 实例 token；历史倾斜视频证据注入。
* **B 流式倾斜编码器**：轻量 FPN（部署时替换为 DINOv3 蒸馏骨干）。
* **C CARR**：IPM 抬升 + (x,y,yaw) 穷举相关 + 贝叶斯偏置先验 + 特征度量 LM + GNSS 偏置 Kalman；变化区域降权（流式 EM）。
* **D PASID**：先验/跟踪/发现三类棱柱查询，**竖直角线可变形采样**（Prism Deformable Cross-Attention）。
* **E RIGM**：几何结构嵌入（GeoTransformer 式）+ 跨图偏置 + 节点自适应 dustbin Sinkhorn。
* **F RRD**：留一法局部刚体（ICP-Procrustes）去除配准残差 → 双向符号距离 → 扩张/收缩。

v1 的 RRD 机制实验：注入 (1.5, −1.0) m + 0.02 rad 配准误差，ICP 5 次迭代后未变化建筑 IoU 从 0.67–0.77 恢复到 ≈1.00，扩张建筑保持 0.65（理论 0.667），平移被精确恢复（`tests/test_core.py`）。

---

## 3. Layer 1：以审稿人视角自审 v1

| # | 缺陷 | 依据 |
|---|---|---|
| W1 | **空间关系只用到一阶**：RIGM 的关系嵌入仅作注意力偏置，匹配分数只看单个对应，未约束“对应之间”的一致性 | 外观相似的成排建筑（城中村、工业园）下，单对应打分不可分 |
| W2 | **倾斜几何建模弱**：IPM 抬升把屋顶当地面；高度仅为隐变量 | 定量：相机高 $H$=80 m，水平距 $r$=150 m，楼高 $h$=20 m 时，屋顶点 IPM 错位 $rh/(H-h)$ = **50 m** |
| W3 | **跨时相比较依赖视角**：两期航迹不同，比较视角相关特征会把“视角差异”误判为变化 | 倾斜数据集上 2D CD 方法出现排名倒置（iVISION-2DCD, 2026） |
| W4 | **“拆除/未观测”判定是启发式**（noisy-or 可观测性），没有不确定性语义 | 运营中“误报拆除”代价最高 |

---

## 4. Layer 2：v2 机制推导与迭代

### 4.1 v2-I 高度解耦的倾斜观测模型：接地线（Ground-Contact Line, GCL）

**推导.** 地平面 $z=z_g$ 上的点满足单应 $\mathbf{u}\sim K[r_1\;r_2\;r_3z_g+t](x,y,1)^\top=H_g(x,y,1)^\top$，可**精确**逆映射；高 $h$ 的屋顶点经 $H_g^{-1}$ 抬升错位 $\Delta=rh/(H-h)$，随高度和距离急剧增大。立面与地面的接触线位于地平面上——**它是倾斜影像中唯一与未知楼高无关的建筑结构**。据此把观测按高度依赖性分解：

* **GCL 通道（与高度无关）** → 配准（对 SHP 边界）、足迹边精化（contact snake）；
* **屋顶边通道（依赖高度）** → 逐实例一维“高度扫描”后验。

两个通道的监督均由 GT 足迹/高度/位姿经 z-buffer 可见性**自动渲染**，无需额外标注（`structure.render_structure_targets`）。

**迭代过程（失败 → 修正）：**

1. *v2-I.0 BEV“抬升-比较”*：把 GCL 概率图抬升到地面网格，与 SHP 边界软栅格相关。**失败**：GT 位姿下抬升线的 SHP 支持度 0.21，与错误位姿下 0.21 无差别，后验平坦。
   *诊断*：步长 4 的像素在 150 m 处覆盖约 3 m × 6 m（地面透视压缩 ≈ 1/sin 俯角），细线被抬升成随距离变化的各向异性“块”。
2. *v2-I.1 传感器空间“渲染-比较”*：对每个 (yaw, x, y) 假设 $G$，真位姿为 $G\circ T_{prior}$，故世界点 $X$ 的成像为 $\pi(G^{-1}(X);T_{prior})$；把**面向相机**的 SHP 边界（地面高度）投影到影像，在 GCL 图上取均值作为结构似然。检测器噪声在影像空间是各向同性的。假设投影的几何正确性验证：与 GT 投影最大差 4×10⁻⁵ px。
3. *分辨率约束（新发现）*：256×160、f=180 px 时 1 m@150 m ≈ 1.2 px，3 m 误差不足一个步长-4 单元 → 无法分辨；960×544、f=768 px 时 1 m ≈ 5 px。

**实验（无纹理地面，仅用结构线，注入误差 3.61 m / 2.5°，8 个随机场景，`tests/test_v2_mechanisms.py`）：**

| 证据通道 | 分辨率 | 平移误差 | 航向误差 |
|---|---|---|---|
| 接地线 GCL | 256×160 | 3.09 m | 0.74° |
| 屋顶边 | 256×160 | 8.21 m | 4.85° |
| **接地线 GCL** | **960×544** | **0.18 m（最大 0.30 m）** | **0.37°** |
| 屋顶边 | 960×544 | 7.60 m | 4.90°（劣于未校正先验） |

结论：① 高度解耦推导成立——依赖高度的证据在倾斜视角下不仅无用而且**有偏**；② 结构头必须在 960×544 输入的步长 4 上运行（写入实时预算）。

**高度扫描（Instance Height Sweep）.** 高度假设 $h_m$（3–120 m 对数间隔），每帧增量
$\Delta L_m=\beta\,(s_m-\bar s)\,v$，$s_m$ 为高度 $h_m$ 处屋顶轮廓投影点的屋顶边概率均值，$v$ 为可见比例；后验沿视频累积（常数内存，带方差）。oracle 屋顶边下，9 m / 27 m 两栋楼 5 帧后估计误差 < 20%。

**Contact snake.** 沿每个足迹顶点外法向在地面采样 GCL 剖面（±3 m，7 点）→ 法向位移 + 置信度，叠加到 PASID 每层的顶点更新（与高度无关的足迹约束）。

相关工作定位：LOFT/BONAI、OBM/OBMv2、PolyFootNet、DragOSM（2025，指出离底视角屋顶-立面边界模糊）、ObliCity（2026，屋顶-足迹位移校正）均在**影像域**回归偏移；早期倾斜航摄方法用手工线段/立面竖直性。本文首次把接地线的**高度不变性**作为深度框架中配准、足迹与关联的共同几何原语。

### 4.2 v2-II 层级空间关系推理（Hierarchical Spatial-Relation reasoning, HSR）

**推导.** 将 $p(\sigma)$ 写成带出生/死亡状态的成对 MRF：

$$
\log p(\sigma)=\sum_j u(j,\sigma_j)+\sum_{j<k}\psi\big((j,\sigma_j),(k,\sigma_k)\big)+\log p(G\mid\sigma)
$$

* **一阶** $u$：RIGM（外观、形状、来源）；
* **二阶** $\psi$：空间关系保持——Lawler 二次指派（QAP）。候选集上的兼容度
  $C_{mn}=\exp\!\big(-\tfrac12(\tfrac{d^1_{ij}-d^2_{ab}}{\tau_{mn}})^2\big)\cdot\sigma(\mathrm{MLP}(\mathbf{r}_{mn}))$，
  $\mathbf{r}_{mn}$ 含距离差、方位差、邻接关系一致性、尺寸序一致性；容差 $\tau$ 随建筑尺度增大（扩张会移动质心）；
  再用 SC² 二阶相容 $C^{(2)}=C\odot(CC)$ 抑制偶然相容；
* **全局**：一致运动场。对每个候选做留一法局部刚体估计（排除同一 T1 建筑的其它候选），其对齐残差 $e_m$ 以**内点/离群点对数似然比**进入：$\ell_m=\kappa-e_m^2/2\sigma_g^2$（CPD 式均匀离群分量，$\kappa=\log\frac{1}{u_{out}\sqrt{2\pi}\sigma_g}$）。

求解：神经渐进指派（Graduated Assignment，Gold & Rangarajan）+ 带 dustbin 的 log-Sinkhorn + 温度退火，外层 EM 交替“软指派 ↔ 运动场”：
$Q_m=s_m+\lambda\frac{(C^{(2)}P)_m}{\sqrt{\deg_m}}+\ell_m,\quad P\leftarrow\mathrm{Sinkhorn}_\beta(Q;\alpha,\beta)$。
**dustbin 质量 = 出生/死亡（新增/拆除）后验**——变化是联合配准-关联问题的离群类。

**迭代过程（基准：5×5 外观相同建筑群，残余配准误差 (4,−3) m + 0.03 rad，1 栋拆除 + 1 栋新建，外观分数 σ=1.2 噪声、真对仅 +1.0 信号；10 个随机种子）：**

| 版本 | 改动 | 关联准确率 |
|---|---|---|
| 一元 OT（基线） | — | 19.2% |
| v2-II.0 | 一元 top-k 候选 + 按候选数归一 | 48% |
| v2-II.1 | **几何 kNN ∪ 一元 top-k 候选**（候选召回 62.5% → 100%） | 68% |
| v2-II.2 | 按**相邻建筑数的平方根**归一（k 个独立投票：信号 ∝ k，噪声 ∝ √k） | 73% |
| 误差分解 | 错配 **0**，全部为“真对被判入 dustbin”；运动残差：真候选 0.44 m vs 假候选 27.35 m | — |
| **v2-II.3** | 全局项改为**内点/离群点对数似然比**（奖励几何验证，而非仅惩罚不一致） | **99.2%（最低 96%）** |

层级消融：一元 19.2% → +二阶关系（无全局 EM）44.4% → +全局运动场 EM 99.2%。
鲁棒性：残余误差 (9,−6) m → 94.8%；旋转 0.08 rad → 99.2%；外观噪声 σ=2.0 → 87.2%；σ=2.0 且信号减半 → 78.4%（一元仅 14–18%）。
在完整流水线中（未训练权重、3 个合成场景），HSR 把跨时相关联 NLL 从 1.9–2.2（一元）降至 0.41–0.82。

相关工作定位：SC²-PCR / MAC / TurboReg（ICCV 2025）/ PointTruss（NeurIPS 2025）解决点云对应剪枝；GMTR、ICLR 2025 部分图匹配、TPAMI 2026 双专家框架解决通用 QAP；RPM-Net / DeepGMR / CPD 联合估计对应与变换。本文把三者统一到**建筑实例级、地理坐标、带变化语义的 dustbin** 中，并给出“变化 = 离群类”的概率解释。

### 4.3 v2-III 棱柱纹理图集（Prism Texture Atlas, PTA）

**动机（W3）.** 用**表面坐标**而非视角来索引证据：墙面单元 $(s,\tau)$（周长弧长 × 归一化高度），屋顶单元 $(u,v)$（主轴网格，限于多边形内部）。每帧把影像特征反投影到可见单元，权重
$w=\mathbb{1}[\text{在像内}]\cdot\mathbb{1}[\text{朝向相机}]\cdot\mathbb{1}[\text{未被遮挡(z-buffer)}]\cdot\cos\angle(\text{视线},\mathbf{n})$，
累积 $(A,W)$——每个实例常数内存，与视频长度无关。

* 作为 PASID 的**实例记忆**（替代 v1 的“视角多样槽”）：查询对图集 token（单元均值 + 单元位置嵌入）做交叉注意力；
* 跨时相比较：RRD 刚体对齐后按表面位置配对单元，仅在**两期都观测到**的单元上计算外观残差；另给出“**新墙比例**”（T2 墙在 T1 中无对应 → 扩张证据）与“**丢失墙比例**”（收缩证据）——与视角和光照都无关的几何证据。

oracle 验证：背向相机的墙面单元权重严格为 0，屋顶在俯视下可见（`tests/test_v2_mechanisms.py`）。
相关工作：经典立面纹理一致性融合（Wang et al.；Böhm 2004）与 3D 资产 UV 融合（AssetGen 2026）面向重建/生成；本文把表面参数化图集用作**变化检测的视角校正记忆与比较空间**。PRoPE（NeurIPS 2025）/ RayRoPE（2026）通过相机相对编码让注意力感知视锥几何；PTA 则直接在共享世界系的物体表面坐标中对齐证据。

### 4.4 集成中发现并修复的系统性问题

| 现象 | 根因 | 修正 |
|---|---|---|
| 未训练模型关联 NLL 异常高（17.1，应 ≈ log(M+N)≈3.7），dustbin 分数达 27.5 | **流式递归状态范数爆炸**：解码器残差流作为跟踪状态逐帧传播，无输出归一化，T2 特征范数 441（T1 约 2） | 传播状态前加 LayerNorm（范数稳定在 √d=8），NLL 降至 0.4–0.8 |
| 同一建筑的锚定轨迹与发现轨迹共享 GT | 目标定义歧义 | GT 匹配优先选择锚定轨迹，其次观测最多者 |
| 图集投影层无梯度 | 证据累积时被截断 | 训练时图集在片段内保持可微（截断 BPTT），推理时 no_grad |
| 周期性跨时相推理耗时 111.6 s（196 栋 × 196 轨迹，CPU） | 留一法局部刚体的 IRLS 显式构造 [P,P,K,2] 残差张量（HSR 候选 P≈1500） | 改写为**充分统计量上的核回归**：鲁棒权重在各自对的局部运动下计算后，加权 Procrustes 只需 $\sum r,\sum rp,\sum rq,\sum r(p\cdot q),\sum r(p\times q)$ 五个逐对统计量，局部拟合退化为若干 P×P 矩阵向量积；耗时降至 0.83 s（≈135×），精度不变（HSR 99.2%） |

---

## 5. Layer 3：v3 迭代（对 v2 再审视）

### 5.1 证据式决策：拆除 vs 未观测（W4）

每个 T1 建筑累积**按可观测度加权**的占用证据：
$r_j\mathrel{+}=o_{jt}\,\mathrm{softplus}(\ell_{jt}),\ s_j\mathrel{+}=o_{jt}\,\mathrm{softplus}(-\ell_{jt})$，
主观逻辑二项意见 $b=\frac{r}{r+s+2},\ d=\frac{s}{r+s+2},\ u=\frac{2}{r+s+2}$。
未匹配的 T1 建筑仅当 $u<u_{max}$ 时报告“拆除”，否则报告“未观测”；占用信念高但未匹配时额外标记“重建或误匹配”供人工复核。训练用证据 Beta 损失（期望交叉熵 + 误导证据的 KL 正则）。

### 5.2 相关帧的有效样本量折扣（对 v2 的再审视）

10 FPS 相邻帧高度相关，逐帧当作独立观测会让 $u$ 过早塌缩（相关观测的有效样本量远小于名义样本量，如 121 个自相关观测仅等价于 29 个独立观测）。用图集度量**新颖度**：
$\nu=\frac{\sum_c \Delta W_c/(1+W_c)}{\sum_c\Delta W_c}\in(0,1]$，证据权重 $o\cdot\nu$。

| 8 帧观测 | 逐帧独立计数 $u$ | 新颖度折扣 $u$ |
|---|---|---|
| 高度相关（重复看同一面） | 0.075 | **0.192**（≈2.7 个有效观测） |
| 多样视角（每帧新表面） | 0.075 | 0.075 |

### 5.3 场景级外观残差归一化

两期光照/季节不同会产生全局外观偏移。借鉴相对辐射归一化中“伪不变特征/无变化集”的思想：以 HSR 中**高置信且充分共视**的匹配对为无变化集，减去其图集残差的中位数，在特征空间去除全局偏移。

### 5.4 与高度无关的新建筑提议

v2 已证明 IPM 抬升对高出地面的结构有偏；发现查询的 BEV 热图输入加入抬升后的 GCL 概率（只有接地线在地面上是精确的），使新建筑提议锚定在高度无关的证据上。

---

## 6. 当前网络：MARS-CD v3

```
离线 T1                                                在线 T2（每帧）
DOM(卫星/航摄)+栅格地图 ─┐                              帧 → B 倾斜编码器 ─→ v2-I 结构头(GCL / 屋顶边, 步长4)
SHP 足迹 ────────────────┼─→ A 地理先验记忆               │
历史倾斜视频 ─(B,C,D)────┘   (+SHP边界栅格, T1图集,        GNSS+云台 → C CARR: 偏置KF + BEV特征相关
                              T1高度后验)                     + 影像空间 GCL↔SHP 渲染比较 + 特征度量LM
                                                                 │
                                       v2-I 高度扫描(逐实例一维后验) │ contact snake
                                                                 ↓
                                  D PASID: 先验/跟踪/发现 棱柱查询 ← 图集记忆(v2-III)
                                                                 ↓
                         跟踪库: 图集累积, 高度后验, z-buffer 可观测度 → v3 新颖度折扣证据
                                                                 ↓ (周期性/航段结束)
                         E RIGM 一元 → v2-II HSR(二阶QAP + 运动场EM, 变化=离群类)
                                                                 ↓
                         F RRD + 图集比较(场景归一) + 分类 → v3 证据式地理变化报告(SHP)
```

---

## 7. 端到端合成训练（冒烟测试，非精度结论）

见 `research_proposal.md` §7（训练日志与指标随代码一并给出；合成数据只验证可训练性）。

---

## 8. 下一层待深挖（v4 方向）

1. **边级空间关系**：HSR 目前基于质心关系，对连排/大型不规则建筑不够；下一步以 GCL 边段为节点构建边级关系图（边-边方位/共线/间距保持），与实例级 HSR 分层耦合。
2. **非平顶屋顶**：棱柱假设对坡屋顶只在檐口成立；扩展为“檐口高度 + 屋脊高度”的二维扫描或屋顶类型分类。
3. **无 DEM 地形**：以接地线高度一致性联合估计分片地面高程场（每个实例基底高程已为可学习残差）。
4. **跨时相相对几何注意力**：借鉴 PRoPE/RayRoPE，把 T1/T2 两期视锥关系编码进图集比较的注意力。
5. **几何基础模型蒸馏**：StreamVGGT/SAM-V 等作为离线教师提供遮挡与深度监督，在线保持轻量。
6. **主动观测**：利用证据不确定性 $u$ 与新颖度 $\nu$ 生成“补飞建议”（哪些建筑需要哪个方向的视角）。

## 9. 需要你确认的问题

1. 训练数据能否做离线空三（Pix4D/Metashape/ContextCapture）得到**逐帧真值位姿**与 DSM（作为高度伪标签）？本方案的配准与高度监督默认依赖它。
2. 视频分辨率、焦距（或视场角）、典型航高与俯仰角？v2-I 的结论要求工作距离上 1 m ≥ 约 4 px（960×544 输入下成立）。
3. 屋顶类型构成（平顶为主还是坡屋顶较多）与地形（平原城区 / 山地）？决定 v4 第 2、3 项的优先级。

---

## 参考文献（本轮检索）

* iVISION-2DCD（2026）: https://arxiv.org/html/2607.03553v1
* ObliCity（2026）: https://arxiv.org/pdf/2607.25210 ；DragOSM（2025）: https://arxiv.org/pdf/2509.17951
* BONAI/LOFT: https://arxiv.org/pdf/2204.13637 ；OBM/OBMv2, PolyFootNet: https://arxiv.org/pdf/2408.08645
* OrienterNet: https://ar5iv.org/html/2304.02009 ；PiLoT v2（2026）: https://arxiv.org/abs/2606.31098
* SC²-PCR / MAC: https://ar5iv.labs.arxiv.org/html/2305.10854 ；TurboReg（ICCV 2025）: https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_TurboReg_TurboClique_for_Robust_and_Efficient_Point_Cloud_Registration_ICCV_2025_paper.pdf ；PointTruss（NeurIPS 2025）: https://proceedings.neurips.cc/paper_files/paper/2025/file/49ea0f02a4e7da79bc121e29337389ac-Paper-Conference.pdf
* 部分图匹配（ICLR 2025）: https://proceedings.iclr.cc/paper_files/paper/2025/file/cfa8440d500a6a6867157dfd4eaff66e-Paper-Conference.pdf ；TPAMI 2026 部分/噪声对应图匹配: https://pubmed.ncbi.nlm.nih.gov/41801786/
* RPM-Net: https://arxiv.org/pdf/2003.13479 ；DeepGMR: https://arxiv.org/pdf/2008.09088 ；GeoTransformer: https://arxiv.org/pdf/2202.06688
* PRoPE（NeurIPS 2025）: https://github.com/liruilong940607/prope ；RayRoPE（2026）: https://arxiv.org/pdf/2601.15275 ；DPPE（2026）: https://arxiv.org/pdf/2606.31585
* IGGT: https://arxiv.org/pdf/2510.22706 ；PanSt3R: https://arxiv.org/pdf/2506.21348 ；SAM-V（2026）: https://arxiv.org/pdf/2609.25490 ；MV-SAM: https://arxiv.org/html/2601.17866v1
* MatchAnything（2025）: https://arxiv.org/pdf/2501.07556
* 证据深度学习与主观逻辑: https://arxiv.org/pdf/2309.13207
* 相关观测有效样本量（IMEKO TC4 2007）: https://imeko.org/publications/tc4-2007/IMEKO-TC4-2007-039.pdf
* 相对辐射归一化（无变化集 EM）: https://arxiv.org/pdf/2111.12406
* 倾斜影像立面/足迹经典方法: https://www.researchgate.net/publication/263446651_Building_Footprints_Extraction_from_Oblique_Imagery
