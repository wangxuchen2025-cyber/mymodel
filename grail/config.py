"""GRAIL-CD 配置。

坐标约定：地图系为 AOI 局部 ENU（x 东、y 北，米）；当前时相的一切几何先用"粗元数据"投到地图系，
称为 GNSS 系；历史数据位于真实地图系。二者之间的全局相似变换 T 与逐栋残差 δ 由 GRAIL 层估计。
"""
from dataclasses import dataclass, field
from typing import Tuple

from geomac.config import BackboneCfg

HIST_SOURCES: Tuple[str, ...] = ("shp", "aerial", "hist_uav", "sat", "map")   # 也是历史集合融合的参考优先级


@dataclass
class DetCfg:
    """当前视频分支：倾斜原始帧上的屋顶实例 + 相对高度 η。"""
    dim: int = 128
    n_new: int = 60                 # 每帧新建检测查询数
    n_track: int = 64               # 每帧最多注入的跟踪先验查询
    n_layers: int = 3
    heads: int = 8
    n_points: int = 4
    n_poly: int = 16
    emb_dim: int = 64               # 实例外观嵌入
    eta_max: float = 0.8            # η = h / H 上限


@dataclass
class GrailCfg:
    """空间约束对齐关联层。"""
    dim: int = 128
    emb_dim: int = 64
    n_layers: int = 3               # 几何自注意力 + 交叉注意力 交替层数
    heads: int = 4
    knn: int = 8                    # 空间关系邻域
    sigma_d: float = 15.0           # 距离嵌入尺度（米）
    em_iters: int = 3               # 对应–变换 EM 展开次数
    sinkhorn_iters: int = 20
    hough_bins_theta: int = 21
    hough_bins_t: int = 41
    hough_kernel: float = 1.5       # 投票核宽（以格网间距为单位）
    hough_nms_t: int = 5            # 平移维非极大抑制窗口（格）
    sigma_rel: float = 2.0          # 邻域一致性核宽（米）
    sigma_delta: float = 1.5        # 逐栋残差先验（米）
    shape_freq: int = 6
    # 消融开关
    use_rel: bool = True            # 组内几何关系自注意力
    use_nc: bool = False            # 跨组二阶邻域一致性（两轮实验均无增益且显存开销大，默认关闭）
    use_hough: bool = True          # Hough 投票初始化
    use_shape: bool = True          # 形状重叠项 + 变化感知 M 步
    use_tor: bool = False           # 格网拓扑序关系（TOR）：Hough 投票门控 + E 步一致性项
    tor_sigma: float = 0.35         # 序号一致性核宽（户）
    tor_vote_pow: float = 2.0       # Hough 投票按一致性加权的指数
    tor_estep: bool = False         # 是否把序一致性加入 E 步逐对打分（实验表明会误伤被变化打断的排，默认关）
    tor_select: bool = True         # 多假设择优时按序一致性加权软内点
    lattice_expand: bool = True     # 格网补全假设：把前两个 Hough 峰沿历史布局的周期向量 ±v 平移作为额外候选
    hard_refine: int = 3            # EM 后"软到硬"精化轮数（硬互匹配 + 无先验 Umeyama，仅精化变换）
    idf_vote: bool = False          # 形状独特性（逆频率）加权投票与假设择优（免训练，可在推理时开启）
    idf_tau: float = 0.35           # 形状相似核宽（描述子 L1 / log 面积）
    idf_pow: float = 0.5            # 权重 = u^pow（软化，避免完全压制重复建筑）


@dataclass
class MemCfg:
    max_tracks: int = 256
    min_obs_stable: int = 4         # 观测次数达到后视为稳定
    max_age: int = 30               # 连续未观测帧数超过则冻结
    birth_thresh: float = 0.5
    keyframe_move_m: float = 3.0    # 天底点移动超过该距离才触发关键帧


@dataclass
class GrailCDCfg:
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    det: DetCfg = field(default_factory=DetCfg)
    grail: GrailCfg = field(default_factory=GrailCfg)
    mem: MemCfg = field(default_factory=MemCfg)
    raster_res: float = 0.5         # 历史栅格源分辨率（米/像素）
    raster_hw: Tuple[int, int] = (512, 512)
    max_hist: int = 256
    pair_patch: int = 32            # 几何变化头的成对栅格边长


def small_cfg() -> GrailCDCfg:
    cfg = GrailCDCfg()
    cfg.backbone.img_hw = (144, 256)
    cfg.backbone.depth = 4
    cfg.backbone.dim = 192
    cfg.backbone.heads = 3
    cfg.backbone.tap_layers = (0, 1, 2, 3)
    cfg.det.n_new = 24
    cfg.det.n_track = 24
    cfg.det.n_layers = 2
    cfg.raster_hw = (192, 192)
    cfg.max_hist = 48
    cfg.mem.max_tracks = 64
    return cfg
