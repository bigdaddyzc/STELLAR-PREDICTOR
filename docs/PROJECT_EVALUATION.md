# Stellar Predictor — 项目评估报告与优化方案

> 评估日期：2026-06-15 · 评估范围：全仓库（53 个 Python 文件，~5535 行源码 + 685 行测试）
> 评估方法：通读核心算法代码 · 运行测试套件（47 passed, 覆盖率 43%）· 运行 `eval_accuracy.py` · 缺陷复现验证

---

## 1. 总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | ★★★★☆ | 分层清晰（Data→Patterns→Properties→Viz），模块职责明确 |
| 代码质量 | ★★★☆☆ | 核心可读，但存在超长方法、大量魔法数、若干死代码模块 |
| 科学严谨性 | ★★★☆☆ | TB+Hill+共振多信号融合思路合理，但评分体系高度启发式、缺乏校准 |
| 正确性 | ★★☆☆☆ | **存在一处影响核心功能的真实缺陷**（偏心率特性全程失效） |
| 测试覆盖 | ★★☆☆☆ | 43%，核心物理模块 `properties.py`、web、可视化为 0% |
| 工程规范 | ★★★☆☆ | 有 CI，但版本号不一致、跨平台编码问题、无 lint/类型检查 |

**一句话结论**：算法框架与产品形态完成度高，但"偏心率感知稳定性"这一被反复宣传的特性因键名不匹配而**完全失效**，且评分体系充满未经校准的经验参数，科学可信度被高估。建议优先修复正确性缺陷，再做评分体系的可验证化改造。

---

## 2. 关键缺陷（按严重程度排序）

### 🔴 P0-1：偏心率感知稳定性分析全程失效（核心功能 Bug）

**位置**：`stellar_predictor/patterns/stability.py:191`

```python
ecc = p.get("e", 0.0) or 0.0      # ← 读取键 "e"
```

但所有 `ExoplanetSystem` 的行星数据存储用的是键 **`"eccentricity"`**：
- `web/tasks.py:223` → `"eccentricity": ecc`
- `scripts/eval_accuracy.py:80` → `"eccentricity": ecc`

**后果**：`extract_planet_data_full()` 对全部 5 个系统中的 4 个系外系统，偏心率**永远读成 0.0**。复现验证：

```
$ python -c "...extract HD 219134..."
eccentricities extracted: [0.0, 0.0, 0.0, 0.0]   # 实际 c/d/e 偏心率为 0.062/0.138/0.34
```

这意味着 README 与 CLAUDE.md 中反复宣传的"v0.3 eccentricity-aware Hill stability"对 Web/CLI 主路径**是死代码**。HD 219134 的 e 行星 e=0.34，影响显著却被忽略。

**修复**：键名统一。`extract_planet_data_full` 改为兼容读取：
```python
ecc = p.get("eccentricity", p.get("e", 0.0)) or 0.0
```
并新增针对该路径的回归测试（当前 `tests/` 无任何系外系统偏心率断言）。

---

### 🟠 P1-2：`eval_accuracy.py` 在 Windows 下崩溃（编码）

**现象**：脚本输出含 `✅`/`❌`/`─` 等字符，Windows 默认 GBK 控制台抛 `UnicodeEncodeError`，脚本在第一个系统就中断（line 146）。

README 已用 `PYTHONIOENCODING=utf-8` 绕过，但脚本本身不应依赖外部环境。

**修复**：脚本开头加
```python
sys.stdout.reconfigure(encoding="utf-8")
```

---

### 🟠 P1-3：版本号三处不一致

| 位置 | 版本 |
|------|------|
| `pyproject.toml:3` | `0.1.0` |
| `web/app.py:29` | `0.1.0` |
| `README.md` / git 历史 | `v0.5` |

**修复**：以 `pyproject.toml` 为单一真源，`app.py` 用 `importlib.metadata.version("stellar-predictor")` 动态读取，统一升到 `0.5.0`。

---

### 🟡 P2-4：`ExoplanetSystem` 数据契约未落实

`models.py:99` 文档声称行星 dict 键为 `name, a, mass, period, e, i, radius_earth`，但实际只填了 `name, a, mass, eccentricity`。`plotly_viz.py:323` 又去读 `actual_params.get("eccentricity")`——键名在不同文件间漂移，正是 P0-1 的根因。

**修复**：用 `@dataclass` 或 `TypedDict` 把行星结构固化，杜绝字符串键漂移。

---

## 3. 科学严谨性分析

### 3.1 Titius-Bode 拟合：R² 被结构性高估

`fit_log_linear` / `fit_with_skips` 通过**枚举起始索引（0–4）× 枚举跳隙组合**来挑最大 R²。这本质是在扩大假设空间后取最优，R² 必然虚高——8 行星太阳系报 R²=0.998 部分来自这种搜索自由度，而非定律本身的预测力。

**已有缓解**：代码计算了 LOOCV RMSE（`_loocv_rmse`），这是正确方向。但 LOOCV 仅用于"warning 提示"，**未进入评分权重**——`_compute_dynamic_weights` 只看 `r_squared`，不看 `loocv_rmse`。

**建议**：用 LOOCV RMSE（而非 R²）驱动 TB 置信度权重，这才是抗过拟合的诚实指标。

### 3.2 评分体系：大量未校准魔法数

`predictor.py` 中硬编码常量举例（非穷尽）：
- `ratio_excess > 1.15 → tb_score = 0.3 + 0.4*(excess-1)`（line 252-254）
- 共振高斯 `exp(-3.0 * (rel/tol)²)`、阶数惩罚 `1.0 - 0.2*(order-1)`（line 172-179）
- 稳定性 sigmoid `2/(1+exp(-r/2.5))-1`（line 191）
- 共振拉偏系数 `0.3`（line 299）、交叉一致性 `boost=0.15*max(0,1-rel*5)`（line 467）
- 系统级归一化 `score/(score+0.15)`（line 136）

这些常量**没有任何数据拟合或文献依据**，且 `combined_score` 没有概率含义——0.78 不代表"78% 概率有行星"。当前"精度表"（README §Accuracy）只在已知系统上做事后描述，**没有留出验证集**，无法证明泛化能力。

**建议**：
1. 把魔法数集中到 `config/settings.py`（部分已做，但 predictor 内仍散落大量内联常量）。
2. 建立**留一法系统级验证**：对每个已知系统，逐个"挖掉"一颗真实行星，检验模型能否在正确位置以高分召回它。脚本已有 Neptune/Uranus 两例雏形（`eval_accuracy.py:267+`），应推广为全 5 系统的自动化召回率/位置误差指标，并纳入 CI。

### 3.3 `_cross_reference` 方法过长

单个方法 273 行（predictor.py:205-477），混合了 TB 评分、稳定性评分、共振、子隙检测、外缘外推、交叉一致性、不确定度传播七件事。圈复杂度高，难测试、难维护、难审查。

**建议**：拆为 `_score_primary_gap` / `_detect_sub_gaps` / `_predict_outer_edge` / `_apply_cross_gap_consistency` 等独立可测函数。

---

## 4. 测试与工程

### 4.1 覆盖率（`pytest --cov`，43% 总体）

| 模块 | 覆盖率 | 风险 |
|------|--------|------|
| `physics/properties.py` | **0%** | 🔴 核心 M-R 关系/温度/重力公式无任何测试 |
| `web/*` | 0% | 🟡 API、任务编排、报告生成全无测试 |
| `visualization/*` | 0% | 🟢 可接受（图形输出） |
| `cli.py` | 0% | 🟡 入口无 smoke test |
| `detection/`, `inference/`, `verification/` | 0–28% | 🟡 疑似遗留/未接入主流程 |
| `patterns/predictor.py` | 91% | ✅ 良好 |

**优先补测**：`properties.py`（纯函数、易测、科学关键）+ `extract_planet_data_full` 偏心率路径（防 P0-1 回归）+ web 一条 happy-path 集成测试。

### 4.2 遗留/疑似死代码

`detection/`（orbital_residual 扰动法）、`inference/`（差分进化优化器）、`verification/perturbation.py` 覆盖率极低，CLAUDE.md 标注为"legacy"/"perturbation-based mode"。主预测路径（pattern-based）并不调用它们。

**建议**：明确标注为实验性、移入 `experimental/`，或删除，避免维护负担与读者困惑。

### 4.3 工程缺口

- **无 lint / 格式化 / 类型检查**：`pyproject.toml` 的 dev 依赖只有 pytest/pytest-cov/pre-commit，但无 ruff/black/mypy 配置。`curve_fit` 的 `OptimizeWarning` 一直被吞。
- **CI 仅 GitLab**：仓库 git remote 是 GitLab，但 PR 工作流在 `main`。无 GitHub Actions（如需双平台）。
- **依赖偏重**：`emcee`/`corner`/`pandas`/`astropy` 等被列为核心依赖，但主路径未必使用（多在 legacy 模块），拖慢安装。

---

## 5. 优化方案（分阶段）

### 阶段一：正确性修复（0.5–1 天，必做）
1. **修 P0-1**：`stability.py` 偏心率键名兼容 + 新增系外系统偏心率回归测试。
2. **修 P1-2**：`eval_accuracy.py` 顶部 `sys.stdout.reconfigure(encoding="utf-8")`。
3. **修 P1-3**：统一版本号至 0.5.0，`app.py` 动态读取。
4. 固化 `ExoplanetSystem` 行星数据结构（TypedDict/dataclass）。

### 阶段二：科学可验证化（2–3 天，高价值）
5. 把 `eval_accuracy.py` 的留一法召回测试推广到全 5 系统，产出**召回率 + 位置误差 + 质量误差**三项指标，落为 `tests/test_validation/`，纳入 CI（非 slow）。
6. TB 置信度改由 **LOOCV RMSE** 驱动而非 R²。
7. 把 predictor 内联魔法数提取到 `config/settings.py`，并在文档中标注每个常量的来源/依据（或明确标为"经验值，待校准"）。

### 阶段三：可维护性（2–3 天）
8. 拆解 `_cross_reference` 273 行巨方法为 4–5 个可测子函数。
9. 补 `properties.py` 单元测试（对照已知行星：地球、海王星、木星的 R/ρ/g）。
10. 引入 ruff + mypy（宽松起步），CI 加 lint job；处理 `curve_fit` warning。

### 阶段四：清理（1 天）
11. 决策 `detection/`、`inference/`、`verification/` 去留；如保留则移入 `experimental/` 并补最小测试。
12. 精简 `pyproject.toml` 核心依赖，把 legacy 专用依赖移入 optional extras。

---

## 6. 修复优先级速查

| 优先级 | 项 | 工作量 | 影响 |
|--------|-----|--------|------|
| 🔴 P0 | 偏心率键名 Bug + 回归测试 | 2h | 恢复核心宣传特性 |
| 🟠 P1 | 评估脚本编码崩溃 | 15m | 跨平台可用 |
| 🟠 P1 | 版本号统一 | 15m | 一致性 |
| 🟡 P2 | 留一法全系统验证 + 入 CI | 1d | 科学可信度 |
| 🟡 P2 | LOOCV 驱动权重 | 3h | 抗过拟合 |
| 🟡 P2 | properties.py 测试 | 4h | 物理正确性 |
| 🟢 P3 | 拆方法 / lint / 清理依赖 | 2–3d | 可维护性 |
