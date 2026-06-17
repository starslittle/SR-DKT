# SR-DKT Harness Agent Demo

基于双状态知识追踪的个性化推荐系统：SR-DKT 估算学生的储存强度 S 与提取强度 R，Harness Agent 根据 S/R 状态给出学习行为推荐。

## 核心理论

SR-DKT 建立在认知科学的**储存/提取双重记忆模型**之上：

- **储存强度 S** — 知识的长期掌握程度，增长缓慢但衰减极慢，反映"学会了"的程度
- **提取强度 R** — 知识当下的可用性，随时间快速衰减，衰减速率由 S 控制（S 越高衰减越慢）

### 模型架构

SR-DKT 采用**双路径 LSTM** 架构：

```
┌─────────────────────────────────────────────────────────────┐
│                      SR-DKT Model                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                ┌──────────────┐          │
│  │  Storage LSTM│                │ Retrieval LSTM│          │
│  │   (lstm_s)   │                │   (lstm_r)    │          │
│  └──────┬───────┘                └───────┬──────┘          │
│         │                                │                  │
│         ▼                                ▼                  │
│     ┌───────┐                        ┌───────┐              │
│     │ fc_S  │                        │ fc_R  │              │
│     └───────┘                        └───────┘              │
│         │                                │                  │
│         ▼                                ▼                  │
│       S_t                            R_raw                  │
│         │                                │                  │
│         │    ┌────────────────────┐      │                  │
│         │───►│ 差异化遗忘衰减     │◄─────│                  │
│         │    │ λ_s < λ_r (慢 vs 快)│      │                  │
│         │    └────────────────────┘      │                  │
│         │                                │                  │
│         ▼                                ▼                  │
│       S_decay                        R_decay                 │
│         │                                │                  │
│         └───────►┌──────────────┐◄────────┘                 │
│                  │ 注意力融合   │                            │
│                  │ α_s·h_s + α_r·h_r │                       │
│                  └───────┬──────┘                            │
│                          │                                   │
│                          ▼                                   │
│                    ┌─────────┐                               │
│                    │ fc_pred │                               │
│                    └─────────┘                               │
│                          │                                   │
│                          ▼                                   │
│                       P(correct)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

核心公式（2026-06 架构修订后）：

```text
# 差异化遗忘：λ_s、λ_r 为两个【独立】可学习参数（softplus 保正）
λ_s = softplus(log_lambda_s)        # Storage 遗忘速率（慢）
λ_r = softplus(log_lambda_r)        # Retrieval 遗忘速率（快），独立学习，不再写死 = λ_s+…

# 时间衰减：直接作用于【进入预测的隐状态】h_s / h_r（而非仅作用于标量强度）
h_s_decay = h_s × exp(-λ_s × Δt_norm)
h_r_decay = h_r × exp(-λ_r × Δt_norm)

# 可解释双状态强度（从衰减后隐状态导出，用于门控与可视化）
S_t = sigmoid(fc_S(h_s_decay));  R_t = sigmoid(fc_R(h_r_decay))

# 注意力融合：门控由 [S_t, R_t] 生成，加权【衰减后的隐状态】
α = softmax(W × [S_t, R_t])
h_fused = α_s × h_s_decay + α_r × h_r_decay

# 预测
P(correct) = sigmoid(fc_pred(h_fused))

# 软约束损失：鼓励（而非强制）λ_r > λ_s；λ 独立后此项才有真实梯度
constraint_loss = relu(λ_s - λ_r + margin)
```

> **架构修订说明**：早期版本把遗忘衰减只施加在标量 `S_t/R_t` 上、且 `λ_r = λ_s + softplus(·)` 写死，导致 λ 对预测几乎无作用、约束损失恒为 0。现已改为：衰减直接作用于隐状态 `h_s/h_r`，且 λ_s、λ_r 独立可学习 + 软约束。修订后旧 checkpoint 失效，SR-DKT 需重新训练。

Harness Agent 只推荐 `S < 0.5` 或 `R < 0.4` 的薄弱知识点，并按优先级得分取 Top-K。

## 功能特性

- **双路径 LSTM**：Storage 分支处理主动学习行为，Retrieval 分支处理被动接收行为
- **差异化遗忘**：λ_r > λ_s 软约束，模拟 Storage 慢遗忘、Retrieval 快遗忘的认知规律
- **注意力融合**：动态权重融合双路径隐状态
- 多数据集支持：ASSISTments 2009 / 2017 / MOOCCubeX
- MOOCCubeX 行为特征：SR-DKT 使用 `user-problem.json` 与 `user-video.json` 构建 7 元组序列
- pyKT 标准基线：通过独立 `preprocess_mooc_pykt.py` 导出标准 KT CSV，不影响 SR-DKT 数据
- 推荐 Demo：选择学生 → 查看知识状态表、遗忘曲线、Harness 推荐卡片
- 实验结果页：基线模型对比表格 + AUC 柱状图 + 消融实验 + 训练曲线
- 模拟时间推移：通过 slider 模拟未来遗忘，观察 R 衰减过程
- 无数据快速启动：缺失真实数据时自动生成模拟学生状态

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：`torch>=2.0`, `streamlit`, `plotly`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`

## 快速启动

以下命令默认在 `sr_dkt_demo/` 目录下执行：

```bash
cd sr_dkt_demo
```

### 模式 A：直接看 Demo（无需训练）

```bash
pip install -r requirements.txt
streamlit run app.py
```

如果 `data/student_states.pkl`（或 2017 的 `data/student_states_2017.pkl`）不存在，`app.py` 会自动生成 50 个模拟学生的 S/R 状态，并用模拟评估指标启动页面。

在 sidebar 的"数据集"选择器中切换 **ASSISTments 2009** 或 **ASSISTments 2017** 即可查看不同数据集的效果。

### 模式 B：使用 ASSISTments 2009 真实数据

1. 下载 ASSISTments 2009 Skill-builder data（[数据源](https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistments-data)）
2. 将 CSV 放到 `data/skill_builder_data_corrected.csv`
3. 运行完整流程：

```bash
python preprocess.py              # 数据预处理
python train.py                   # 训练 SR-DKT（支持 --dataset 2009）
python export_states.py           # 导出 S/R 状态
python evaluate.py                # 计算评估指标
python baselines/run_baselines.py # 训练基线模型对比
python ablation/ablation_study.py # 消融实验
streamlit run app.py              # 启动前端
```

### 模式 C：使用 ASSISTments 2017 真实数据

1. 下载 ASSISTments 2017 数据集（[数据源](https://sites.google.com/site/assistmentsdata/datasets/2017-assistments-data)）
2. 将 CSV 放到 `data/anonymized_full_release_competition_dataset.csv`
3. 运行：

```bash
python preprocess_2017.py                    # 2017 数据预处理
python train.py --dataset 2017               # 训练 SR-DKT
python baselines/run_baselines.py --dataset 2017  # 基线对比
python ablation/ablation_study.py --dataset 2017  # 消融实验
streamlit run app.py                         # 前端切换到 2017 数据集即可查看
```

### 模式 D：使用 MOOCCubeX 跑 SR-DKT 主实验

原始 MOOCCubeX 文件应放在 `data/mooc/` 下：

```text
data/mooc/
├── relations/
│   ├── user-problem.json
│   ├── user-video.json
│   ├── exercise-problem.txt
│   └── video_id-ccid.txt
└── entities/
    ├── problem.json
    └── video.json
```

SR-DKT 预处理保持 7 元组格式：

```text
(kc_id, correct, delta_t, hint, attempt, watch_ratio, is_replay)
```

运行：

```bash
# 1. MOOCCubeX -> SR-DKT 训练数据
python preprocess_mooc.py --max-kc 30000 --min-interactions 3

# 2. 训练 SR-DKT
python train.py --dataset mooc

# 3. 评估 SR-DKT
python evaluate.py --dataset mooc

# 4. 导出学生 S/R 状态，供推荐层使用
python export_states.py --dataset mooc

# 5. 启动前端查看结果
streamlit run app.py
```

输出文件：

```text
data/mooc/train.pkl
data/mooc/val.pkl
data/mooc/test.pkl
data/mooc/meta_mooc.json
data/mooc/skill_encoder_mooc.pkl
data/mooc/best_model_mooc.pt
```

### 模式 E：MOOCCubeX 的 pyKT 标准基线数据

pyKT baseline 使用独立预处理脚本，不改动 SR-DKT 的 `train.pkl / val.pkl / test.pkl`。

```bash
# 1. 原始 MOOCCubeX -> 逐行交互 CSV
python preprocess_mooc_pykt.py

# 2. 逐行交互 CSV -> pyKT sequence CSV + data_config 片段
python export_mooc_pykt_sequences.py
```

默认策略：

- 读取同一份原始 `user-problem.json`。
- 优先读取已有 `data/mooc/train.pkl`、`val.pkl`、`test.pkl` 的用户划分，使 pyKT baseline 和 SR-DKT 使用相同 train / val / test 学生。
- 如果没有 SR-DKT split，可以显式使用 hash split。
- `export_mooc_pykt_sequences.py` 使用 fold 0 表示 train，fold 1 表示 val，fold -1 表示 test；`timestamp` 和 `duration` 会转换为 pyKT loader 需要的毫秒单位。

```bash
python preprocess_mooc_pykt.py --split-source hash
python export_mooc_pykt_sequences.py
```

输出：

```text
data/mooc_pykt/train.csv
data/mooc_pykt/val.csv
data/mooc_pykt/test.csv
data/mooc_pykt/interactions.csv
data/mooc_pykt/meta_pykt.json
data/mooc_pykt/pykt_ready/train_valid_sequences.csv
data/mooc_pykt/pykt_ready/test_sequences.csv
data/mooc_pykt/pykt_ready/test_window_sequences.csv
data/mooc_pykt/pykt_ready/data_config_moocx.json
data/mooc_pykt/pykt_ready/id_maps.json
data/mooc_pykt/pykt_ready/meta_sequences.json
```

逐行交互 CSV 字段：

```text
split,user_id,order_id,question_id,concept_id,response,timestamp,duration
```

pyKT sequence CSV 字段：

```text
fold,uid,questions,concepts,responses,timestamps,usetimes,selectmasks
```

建议用 pyKT 跑三类标准对照：

| 实验组 | 推荐模型 | 目的 |
|--------|----------|------|
| 标准 KT 基线 | DKT、DKT+、DKVMN、SAKT、AKT、simpleKT | 对比经典 KT 预测能力 |
| 时间 / 遗忘基线 | DKT-Forget、LPKT、HawkesKT | 对比时间和遗忘建模 |
| 行为增强公平性 | 本仓库 LBKT / BEKT / SR-DKT w/o video | 验证 SR-DKT 不是只靠视频特征 |

在 pyKT 仓库中使用时，将 `data/mooc_pykt/pykt_ready/data_config_moocx.json` 里的 `moocx` 配置合并到 pyKT 的 `configs/data_config.json`，或按当前 pyKT 版本的配置方式引用该片段。训练时使用 `dataset_name=moocx`，并设置 `fold=1`，使 fold 0 作为训练集、fold 1 作为验证集。

第一批当前阶段必跑：

```text
DKT, DKVMN, SAKT, AKT, simpleKT, DKT-Forget, SR-DKT
```

其中 `DKT / DKVMN / SAKT / AKT / simpleKT / DKT-Forget` 均作为 pyKT 统一框架下的正式 baseline；`SR-DKT` 使用本仓库代码训练和评估。本仓库自实现的 `DKT / DKVMN / SAKT / AKT` 仅用于预实验、调试或附录补充，不作为最终主表的优先结果。

pyKT 第一批 baseline 命令模板如下。不同 pyKT 版本可能提供 `wandb_{model}_train.py` 包装脚本；若使用通用入口，可在 pyKT 的 `examples/` 目录下按以下形式运行：

```bash
cd /path/to/pykt-toolkit/examples

CUDA_VISIBLE_DEVICES=0 python wandb_train.py --dataset_name=moocx --model_name=dkt --emb_type=qid --fold=1 --use_wandb=0 --add_uuid=0
CUDA_VISIBLE_DEVICES=0 python wandb_train.py --dataset_name=moocx --model_name=dkvmn --emb_type=qid --fold=1 --use_wandb=0 --add_uuid=0
CUDA_VISIBLE_DEVICES=0 python wandb_train.py --dataset_name=moocx --model_name=sakt --emb_type=qid --fold=1 --use_wandb=0 --add_uuid=0
CUDA_VISIBLE_DEVICES=0 python wandb_train.py --dataset_name=moocx --model_name=akt --emb_type=qid --fold=1 --use_wandb=0 --add_uuid=0
CUDA_VISIBLE_DEVICES=0 python wandb_train.py --dataset_name=moocx --model_name=simplekt --emb_type=qid --fold=1 --use_wandb=0 --add_uuid=0
CUDA_VISIBLE_DEVICES=0 python wandb_train.py --dataset_name=moocx --model_name=dkt_forget --emb_type=qid --fold=1 --use_wandb=0 --add_uuid=0
```

如果当前 pyKT 版本的 DKT-Forget 模型名使用连字符或其他别名，以该版本 `configs/kt_config.json` 和 `pykt/models` 中的实际模型名为准。

## 降本与跨框架对齐工作流（10% 子集 + pyKT）

为在有限算力下完成多组对比，并保证 SR-DKT 与 pyKT 基线**可比、不被质疑**，采用以下工作流。

### 一、切 10% 分层训练子集（只采样 train，val/test 全量）

```bash
python subsample_mooc.py --src-dir data/mooc --out-dir data/mooc_10pct --ratio 0.1 --seed 42
```

- 按序列长度分位数分 10 层，逐层等比例采样，分布与全量一致；固定种子可复现。
- val/test/meta/encoder 通过硬链接复用全量文件（零额外占盘）。
- 额外产出 `data/mooc_10pct/subset_train_user_ids.json`，供下一步对齐 pyKT。

### 二、把 pyKT 训练 fold 过滤到同一批 10% 学生

```bash
python filter_pykt_to_subset.py \
    --ids data/mooc_10pct/subset_train_user_ids.json \
    --pykt-dir data/mooc_pykt/pykt_ready \
    --out-dir  data/mooc_pykt_10pct/pykt_ready
```

- 仅过滤 pyKT 的 fold 0（训练）到这 10% 学生；fold 1（验证）与 test 保持全量，与 SR-DKT 评估集一致。
- 自动重写 `data_config` 的 `dpath`。pyKT 训练时把 `dataset_name=moocx` 指向该目录。

### 三、双框架对齐验证（消除“跨框架混淆”质疑的关键）

标准基线在官方 pyKT 框架跑、SR-DKT 在自研框架跑，二者可比性需验证。`baseline_models.py`
提供了 **pyKT 同款标准 DKT（`DKT-Vec`，向量输出 + gather 下一题）** 作为对齐锚点：

```bash
# 自研框架跑 pyKT 同款 DKT（10% 数据）
python baselines/run_baselines.py --dataset mooc --model DKT-Vec

# 官方 pyKT 跑 DKT（同一 10% 数据 data/mooc_pykt_10pct）
# 详见下方 pyKT 命令模板
```

若两者 AUC 差 `< ~0.005`，即证明“自研框架 ≈ pyKT 框架”，SR-DKT 便可与 pyKT 基线表同列比较。
注意：本仓库另有一个**标量输出**的 `DKT`（不知道下一题），仅作历史/调试用，不可与 pyKT 数字对比。

## MOOCCubeX 实验推荐顺序

正式论文实验建议按以下顺序跑：

```bash
# A. SR-DKT 主模型
python preprocess_mooc.py --max-kc 30000 --min-interactions 3
python train.py --dataset mooc
python evaluate.py --dataset mooc

# B. pyKT 标准基线数据
python preprocess_mooc_pykt.py
python export_mooc_pykt_sequences.py
# 然后在 pyKT 中训练第一批正式 baseline:
# DKT / DKVMN / SAKT / AKT / simpleKT / DKT-Forget

# C. 本仓库行为增强基线
python baselines/run_baselines.py --dataset mooc

# D. SR-DKT 消融 / 公平性实验
python ablation/ablation_study.py --dataset mooc

# E. 推荐层评估
python export_states.py --dataset mooc
python evaluate.py --dataset mooc
```

如果服务器有多张 GPU，可以把 baseline 和消融拆开并行跑。单模型 / 单变体模式会写独立结果文件，避免多个进程同时覆盖同一个总结果 JSON：

```bash
mkdir -p logs

# 单独跑某个 baseline
CUDA_VISIBLE_DEVICES=0 python baselines/run_baselines.py --dataset mooc --model DKT > logs/baseline_dkt.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python baselines/run_baselines.py --dataset mooc --model SAKT > logs/baseline_sakt.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python baselines/run_baselines.py --dataset mooc --model AKT > logs/baseline_akt.log 2>&1 &

# 单独跑某个 SR-DKT 消融 / 公平性变体
CUDA_VISIBLE_DEVICES=0 python ablation/ablation_study.py --dataset mooc --variant full > logs/ablation_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ablation/ablation_study.py --dataset mooc --variant no_storage > logs/ablation_no_storage.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ablation/ablation_study.py --dataset mooc --variant no_video > logs/ablation_no_video.log 2>&1 &
```

可单跑的 baseline 名称：

```text
DKT, DKT-Vec, DKT-F, SAKT, AKT, LBKT, DKVMN, BEKT, SR-DKT
```

> `DKT-Vec` 为 pyKT 同款标准 DKT（跨框架对齐锚点）；`DKT` 为标量输出旧版，仅作历史/调试用。

可单跑的消融变体：

```text
full, no_storage, no_retrieval, shared_decay, no_attention, single_path, no_video
```

### 并行实验说明

当前代码可以单独跑每个 baseline 模型，也可以单独跑每个 SR-DKT 消融 / 公平性变体，适合在多 GPU 服务器上并行实验。

| 实验类型 | 单跑参数 | 是否适合并行 | 说明 |
|----------|----------|--------------|------|
| 本仓库 baseline | `--model` | 是 | 每次只训练一个模型，结果写入独立 JSON |
| SR-DKT 消融 / 公平性实验 | `--variant` | 是 | 每次只训练一个变体，结果写入独立 JSON |
| SR-DKT 主模型 | `train.py --dataset mooc` | 可单独跑 | 只建议启动一个主模型训练进程，避免覆盖 `best_model_mooc.pt` |
| pyKT baseline | pyKT 自己的训练参数 | 是 | 在 pyKT 仓库中按模型分别启动 |

单模型 baseline 的输出文件示例：

```text
data/mooc/baseline_results_mooc_dkt.json
data/mooc/baseline_results_mooc_sakt.json
data/mooc/baseline_results_mooc_akt.json
```

单变体消融的输出文件示例：

```text
data/mooc/ablation_results_mooc_full.json
data/mooc/ablation_results_mooc_no_storage.json
data/mooc/ablation_results_mooc_no_video.json
```

训练曲线日志会额外保存到独立目录，避免并行进程抢写同一个日志文件：

```text
data/training_logs/dkt.json
data/training_logs/sakt.json
data/training_logs/full_sr_dkt.json
data/training_logs/w_o_video_features.json
```

并行跑时建议注意：

- 不要同时启动多个完全相同的模型 / 变体，否则 checkpoint 和日志含义会混乱。
- `baselines/run_baselines.py --dataset mooc` 不带 `--model` 时会顺序跑全部 baseline，适合单进程，不适合和多个单模型进程混跑。
- `ablation/ablation_study.py --dataset mooc` 不带 `--variant` 时会顺序跑全部消融，并生成总表和柱状图；并行单变体跑完后，如需总图，可以再单独整理结果或重新跑一次全量消融。
- `training_logs.json` 仍保留给前端或顺序实验读取；并行实验以 `data/training_logs/*.json` 的独立日志为准。

论文结果建议拆成三张表：

| 表 | 内容 |
|----|------|
| 标准 KT 对比 | pyKT-DKT / pyKT-SAKT / pyKT-AKT / pyKT-DKVMN vs SR-DKT |
| 时间 / 遗忘对比 | pyKT-DKT-Forget / pyKT-LPKT / pyKT-HawkesKT vs SR-DKT |
| SR-DKT 消融 | Full / w/o Storage / w/o Retrieval / shared decay / no attention / single path / w/o video |

## 训练配置

所有训练脚本使用统一的配置：

| 参数 | 值 | 说明 |
|------|------|------|
| `batch_size` | 128 | 批次大小 |
| `max_seq_len` | 200 | 最大序列长度；与官方 pyKT 数据对齐，便于跨框架验证 |
| `hidden_size` | 128 | LSTM 隐层维度 |
| `epochs` | 50 | 最大训练轮数 |
| `patience` | 8 | Early stopping 耐心值 |
| `seed` | 42 | 随机种子 |
| `lambda_constraint_weight` | 0.01 | λ 软约束损失权重（λ_s、λ_r 独立可学习） |

> MOOCCubeX 大规模实验采用 **10% 分层训练子集 + 全量验证/测试集** 降本，见下节工作流。

各模型独立学习率：

| 模型 | 学习率 | 说明 |
|------|--------|------|
| DKT / DKT-F / LBKT / SR-DKT | 0.001 | 标准学习率 |
| SAKT | 5e-4 | 较小学习率（Transformer 类） |
| AKT | 1e-4 | 最小学习率（注意力机制复杂） |

## 消融实验设计

SR-DKT 消融 / 公平性实验包含 7 组配置，验证各组件贡献：

| 配置名 | 说明 | 验证点 |
|--------|------|--------|
| **Full SR-DKT** | 完整模型：双路径 LSTM + 差异化遗忘 + 注意力融合 | 基线对比 |
| **w/o Storage** | 去掉 lstm_s，只用 lstm_r，S 固定为 0.5 | Storage 分支必要性 |
| **w/o Retrieval** | 去掉 lstm_r，只用 lstm_s，R 固定为 0.5 | Retrieval 分支必要性 |
| **w/o 差异化遗忘** | `shared_decay`，令 λ_s = λ_r | 差异化遗忘必要性 |
| **w/o 注意力融合** | `no_attention`，固定平均融合 | 注意力机制必要性 |
| **单路径退化** | `single_path`，只保留一条 LSTM | 双路径结构必要性 |
| **w/o 视频特征** | `watch_ratio = 0` 且 `is_replay = 0` | 控制视频特征带来的公平性问题 |

当前主消融脚本复用 `model.py` 中的 embedding 版 `ablation_mode`，避免 MOOCCubeX 30,001 个 KC 下 one-hot 输入导致 OOM。

## 评估指标

### 模型层（Knowledge Tracing）

| 指标 | 说明 |
|------|------|
| AUC | ROC-AUC，衡量答题预测的排序能力 |
| ACC | 准确率（阈值 0.5） |
| F1 | F1-score |

### 推荐层（KT-Harness）

| 指标 | 说明 |
|------|------|
| WCG | WeakConcept-Gain，推荐薄弱知识点后学生正确率的中点增益 |
| HR@10 | Hit Rate，Top-10 推荐中有实际增益的知识点比例 |
| NDCG@10 | Normalized Discounted Cumulative Gain，衡量推荐排序质量 |

## 实验结果

### ASSISTments 2009

| 模型 | AUC | ACC | F1 |
|------|------|------|------|
| DKT | 0.822 | 0.783 | 0.847 |
| DKT-F | 0.821 | 0.781 | 0.846 |
| SAKT | 0.812 | 0.772 | 0.842 |
| AKT | 0.805 | 0.769 | 0.838 |
| LBKT | 0.815 | 0.775 | 0.845 |
| **SR-DKT** | **0.826** | **0.783** | **0.848** |

### ASSISTments 2017

| 模型 | AUC | ACC | F1 |
|------|------|------|------|
| DKT | 0.681 | 0.657 | 0.479 |
| DKT-F | 0.682 | 0.658 | 0.475 |
| SAKT | 0.636 | 0.643 | 0.431 |
| AKT | 0.618 | 0.633 | 0.385 |
| LBKT | 0.643 | 0.642 | 0.465 |
| **SR-DKT** | **0.705** | **0.667** | **0.517** |

### 历史消融结果（ASSISTments 2009）

下面结果来自早期 ASSISTments 2009 实验记录，保留用于参考。MOOCCubeX 正式实验以本文档上方的 7 组 `ablation_mode` 消融 / 公平性实验为准。

| 变体 | AUC | ACC | 说明 |
|------|------|------|------|
| **Full SR-DKT** | 0.826 | 0.783 | 完整模型：双路径 LSTM + 差异化遗忘 + 注意力融合 |
| w/o Storage | 0.699 | 0.623 | 去掉 Storage 分支，AUC 下降 12.7% |
| w/o Retrieval | 0.610 | 0.377 | 去掉 Retrieval 分支，AUC 下降 21.6% |
| Shared LSTM | 0.798 | 0.761 | 旧版合并双路径实验 |
| Uniform Decay | 0.682 | 0.643 | 旧版统一遗忘速率实验 |
| Mean Fusion | 0.815 | 0.775 | 旧版固定平均融合实验 |

## 文件说明

| 文件 | 作用 |
|------|------|
| `model.py` | SR-DKT 模型：双路径 LSTM + S/R 双状态头 + 差异化遗忘 + 注意力融合 |
| `app.py` | Streamlit 前端主入口，支持 2009/2017 数据集切换 |
| `train.py` | 训练脚本，支持 SR-DKT 多输出、λ 约束损失、early stopping（patience=10） |
| `preprocess.py` | 读取或生成 ASSISTments 2009 数据，计算 `delta_t`，编码知识点 |
| `preprocess_2017.py` | 读取 ASSISTments 2017 数据，列名映射和清洗 |
| `preprocess_mooc.py` | MOOCCubeX -> SR-DKT 7 元组训练数据，融合 `watch_ratio` / `is_replay` |
| `preprocess_mooc_pykt.py` | MOOCCubeX -> pyKT 标准交互 CSV，保留 `question_id` / `concept_id` / `timestamp` / `duration` |
| `export_mooc_pykt_sequences.py` | pyKT 交互 CSV -> pyKT sequence CSV，并生成 `data_config_moocx.json` |
| `export_states.py` | 导出测试学生各知识点最新 S/R 状态到 `student_states.pkl` |
| `harness_agent.py` | 根据 S/R 状态生成结构化学习推荐 |
| `evaluate.py` | 输出模型层 AUC/ACC/F1 与推荐层 WCG、HR@10、NDCG@10 |
| `demo.py` | 命令行展示单个学生的知识状态、推荐和评估指标 |
| `baselines/baseline_models.py` | 基线模型定义：DKT（标量，旧）, DKTVec（pyKT 同款，跨框架锚点）, DKT-F, SAKT, AKT, DKVMN, LBKT, BEKT |
| `baselines/run_baselines.py` | 批量训练基线模型，独立学习率配置，打印对比表格 |
| `ablation/ablation_study.py` | SR-DKT 消融 / 公平性实验（7 组配置），生成对比表格和柱状图 |
| `subsample_mooc.py` | 从全量训练集分层切 10% 子集（只采样 train），产出学生 ID 列表 |
| `filter_pykt_to_subset.py` | 把 pyKT 训练 fold 过滤到同一 10% 学生，对齐两框架的训练集 |

## 消融模式说明

消融实验通过 `SRDKT.forward(..., ablation_mode=...)` 控制：

| ablation_mode | 配置 | 关键改动 |
|---------------|------|----------|
| `full` | Full SR-DKT | 完整模型 |
| `no_storage` | w/o Storage | S 固定为 0.5，只使用 Retrieval |
| `no_retrieval` | w/o Retrieval | R 固定为 0.5，只使用 Storage |
| `shared_decay` | w/o 差异化遗忘 | Retrieval 使用 λ_s，取消 λ_r > λ_s |
| `no_attention` | w/o 注意力融合 | S/R 简单平均 |
| `single_path` | 单路径退化 | 只使用 Storage 路径 |
| `no_video` | w/o 视频特征 | 训练时将 `watch_ratio` / `is_replay` 置零 |

## 数据目录结构

```text
data/
├── skill_builder_data_corrected.csv        # 2009 原始数据
├── anonymized_full_release_competition_dataset.csv  # 2017 原始数据
├── train.pkl / val.pkl / test.pkl          # 2009 划分后的序列数据
├── train_2017.pkl / val_2017.pkl / test_2017.pkl    # 2017 划分后的序列数据
├── best_model.pt                           # 2009 最佳模型
├── best_model_2017.pt                      # 2017 最佳模型
├── skill_encoder.pkl                       # 2009 知识点编码器
├── skill_encoder_2017.pkl                  # 2017 知识点编码器
├── student_states.pkl                      # 2009 学生 S/R 状态
├── student_states_2017.pkl                 # 2017 学生 S/R 状态
├── eval_results.pkl                        # 2009 评估指标
├── baseline_results.json                   # 2009 基线对比结果
├── baseline_results_2017.json              # 2017 基线对比结果
├── ablation_results.json                   # 消融实验结果
├── training_logs.json                      # 训练曲线日志
├── meta.json / meta_2017.json              # 数据集统计信息
├── mooc/                                   # MOOCCubeX 原始文件与 SR-DKT 处理结果
│   ├── relations/user-problem.json
│   ├── relations/user-video.json
│   ├── train.pkl / val.pkl / test.pkl
│   ├── meta_mooc.json
│   └── best_model_mooc.pt
├── mooc_pykt/                              # pyKT baseline 数据
│   ├── train.csv / val.csv / test.csv
│   ├── interactions.csv
│   ├── meta_pykt.json
│   └── pykt_ready/
│       ├── train_valid_sequences.csv
│       ├── test_sequences.csv
│       ├── test_window_sequences.csv
│       ├── data_config_moocx.json
│       ├── id_maps.json
│       └── meta_sequences.json
└── ckpt/                                   # 基线模型 checkpoint
```

输出目录：

```text
results/
└── ablation_bar.png                        # 消融实验 AUC 柱状图
```

## 前端页面说明

### 推荐 Demo 页

- **数据集切换**：sidebar 横向选择 ASSISTments 2009 / 2017
- **学生选择**：下拉框选择学生 ID
- **模拟时间推移**：slider 控制额外遗忘天数（0-30 天）
- **Top-K 推荐**：slider 控制推荐数量（3-10）
- **KPI 卡片**：当前学生 ID、薄弱知识点数、平均 S、WCG
- **知识点状态表**：每个知识点的 S、R、距上次天数、遗忘风险
- **遗忘曲线**：各知识点 R 随时间衰减的可视化
- **推荐卡片**：优先级、行动类型、原因、S/R 值、优先级得分
- **运行日志**：Harness Agent 推荐过程的详细日志
- **评估报告**：AUC/ACC/WCG/HR@10/NDCG@10 + DKT vs SR-DKT 对比

### 实验结果页

- **模型对比表格**：DKT / DKT-F / SAKT / AKT / LBKT / SR-DKT 的 AUC/ACC/F1，SR-DKT 行高亮
- **AUC 柱状图**：各模型 AUC 对比，SR-DKT 用深色突出
- **消融实验表格**：去掉不同组件后 AUC/ACC 的变化
- **训练曲线**：各模型 Val AUC 随 Epoch 变化的折线图

## 引用

如果您在研究中使用了 SR-DKT，请引用：

```bibtex
@thesis{sr-dkt-2026,
  title={基于SR-DKT双状态与学习动作推荐的MOOC个性化学习路径推荐系统},
  author={Your Name},
  year={2026},
  institution={Your University}
}
```

## License

MIT License
