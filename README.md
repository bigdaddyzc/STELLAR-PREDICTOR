# Stellar Predictor

基于轨道规律分析（Titius-Bode 法则 + Hill 稳定性）预测行星系统中可能存在的未知天体，并推导其物理参数（质量、半径、密度、温度、表面重力等）。

Predict unknown celestial bodies in planetary systems through Titius-Bode law fitting and Hill-radius stability analysis, with derived physical parameters.

## 快速开始 / Quick Start

```bash
# 安装 / Install
cd stellar-predictor
pip install -e ".[dev,notebooks]"

# 启动 Web 界面 / Launch web interface
stellar-predictor serve --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000

# 运行测试 / Run tests
pytest tests/ -v

# CLI 使用 / CLI usage
stellar-predictor predict --target Uranus --exclude Neptune --years 165
```

## Web 界面 / Web Interface

访问 `http://127.0.0.1:8000` 后可进行以下操作：

- **星系浏览器**：5 个行星系统（太阳系、TRAPPIST-1、Kepler-11、Kepler-33、HD 219134）
- **天体分布示意图**：2D 轨道俯瞰图，球体标记行星，菱形标记预测天体
- **预测报告**：中英双语，包含半长轴、轨道周期、质量、半径、密度、平衡温度、表面重力、Hill 球半径、行星类型、系统年龄等参数
- **预测间隙**：按置信度排序的间隙卡片，显示 TB 分数和稳定性分数

## 项目结构 / Project Structure

```
stellar_predictor/
├── data/              # 数据获取与模型 / Data acquisition & models
│   ├── models.py          # CelestialBody, StellarSystem, ExoplanetSystem, GapResult
│   └── fetcher.py         # JPL Horizons 数据获取
├── physics/           # 物理引擎 / Physics engine
│   ├── nbody.py           # REBOUND N-body 积分器
│   ├── kepler.py          # Kepler 方程求解
│   ├── residuals.py       # 残差分析 + Lomb-Scargle 周期图
│   └── properties.py      # M-R 关系、平衡温度、表面重力、Hill 球等
├── patterns/          # 轨道规律分析 / Orbital pattern analysis
│   ├── titius_bode.py     # Titius-Bode 法则拟合（含质量加权）
│   ├── stability.py       # Hill 半径、稳定性区域检测
│   └── predictor.py       # 间隙预测（TB + 稳定性 + 跨间隙一致性）
├── prediction/        # 预测流水线 / Prediction pipeline
│   └── pipeline.py        # PredictionPipeline.analyze(system)
├── inference/         # 参数推断 / Parameter estimation
│   ├── optimizer.py       # 差分进化优化器
│   └── candidate.py       # 候选天体参数
├── verification/      # 验证 / Verification
│   └── perturbation.py    # 扰动注入交叉验证
├── visualization/     # 可视化 / Visualization
│   ├── plotly_viz.py      # 天体分布图、TB 拟合图、间距分析图
│   └── orbit_plot.py      # Matplotlib 轨道图
├── web/               # Web 服务 / Web interface
│   ├── app.py             # FastAPI 应用
│   ├── tasks.py           # 后台分析任务、报告生成
│   └── routes/            # API 路由（系统、分析、可视化）
└── static/            # 前端静态资源 / Frontend assets
    ├── index.html         # 主页面
    ├── css/app.css        # 暗色主题样式
    └── js/app.js          # 异步轮询、Plotly 渲染、报告展示
```

## 预测原理 / Prediction Method

### 轨道规律分析 / Pattern-Based Analysis

1. 提取行星数据（名称、半长轴、质量），按轨道距离排序
2. 拟合 Titius-Bode 法则：a_n = α × β^n（对数线性回归，可选质量加权）
3. 计算相邻行星对的 Hill 稳定性
4. 对每个间隙计算 TB 分数 + 稳定性分数 → 综合置信度
5. 预测位置取几何平均，偏置向稳定区域
6. 跨间隙一致性验证：非相邻间隙若满足 β^(j-i) 则加分
7. 系统级分数归一化：combined_score /= sqrt(max_score)

### 物理参数推导 / Physical Properties

基于分析公式推导预测天体的物理参数：
- **质量**：Hill 稳定性约束范围
- **半径**：分段 M-R 关系（岩石 → 海王星类 → 气态巨行星）
- **密度**：由质量和半径自洽计算
- **平衡温度**：T_eq = T_* × √(R_* / 2a) × (1 - A)^0.25
- **表面重力**：g = GM/R²
- **Hill 球半径**：R_H = a × (m / 3M)^(1/3)

## 演示 / Demo

### 海王星预测 / Neptune Prediction

经典验证：从太阳系移除海王星，通过天王星轨道扰动重新预测其存在。

```python
from stellar_predictor.data import DataFetcher
from stellar_predictor.detection import OrbitalResidualMethod
from stellar_predictor.physics import NBodySimulator

# 获取不含海王星的太阳系
fetcher = DataFetcher()
system = fetcher.fetch_system("solar_system", exclude=["Neptune"])

# 模拟天王星完整轨道
full_system = fetcher.fetch_system("solar_system")
sim = NBodySimulator(full_system)
result = sim.simulate(t_end=165, n_steps=500)

# 运行扰动探测
method = OrbitalResidualMethod()
detection = method.detect(system, result.positions["Uranus"], result.times, "Uranus")
print(detection.best.summary())
```

### 规律分析 / Pattern Analysis

```python
from stellar_predictor.prediction import PredictionPipeline
from stellar_predictor.data.models import ExoplanetSystem

# 加载 TRAPPIST-1 系统
system = ExoplanetSystem(name="TRAPPIST-1", stellar_mass=0.089)
# ... 添加行星数据 ...

# 运行规律分析
pipeline = PredictionPipeline()
result = pipeline.analyze(system)
for gap in result.predicted_gaps:
    print(f"{gap.inner_planet} → {gap.outer_planet}: "
          f"a={gap.predicted_a:.2f} AU, score={gap.combined_score:.2f}")
```

## 依赖 / Dependencies

- **rebound** — N-body 积分器
- **scipy** — 差分进化优化、Lomb-Scargle 周期图
- **plotly** — 交互式 Web 可视化
- **fastapi** + **uvicorn** — Web 服务
- **numpy** — 数组运算
