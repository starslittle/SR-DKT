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

核心公式：

```text
# 差异化遗忘
λ_s = softplus(log_lambda_s)        # Storage 遗忘速率（慢）
λ_r = λ_s + softplus(log_lambda_r)  # Retrieval 遗忘速率（快，硬约束 λ_r > λ_s）

# 时间衰减
S_t = S_raw × exp(-λ_s × Δt_norm)
R_t = R_raw × exp(-λ_r × Δt_norm)

# 注意力融合
α = softmax(W × [S_t, R_t])
h_fused = α_s × h_s + α_r × h_r

# 预测
P(correct) = sigmoid(fc_pred(h_fused))

# 约束损失
constraint_loss = relu(λ_s - λ_r + margin)
```

Harness Agent 只推荐 `S < 0.5` 或 `R < 0.4` 的薄弱知识点，并按优先级得分取 Top-K。

## 功能特性

- **双路径 LSTM**：Storage 分支处理主动学习行为，Retrieval 分支处理被动接收行为
- **差异化遗忘**：λ_r > λ_s 软约束，模拟 Storage 慢遗忘、Retrieval 快遗忘的认知规律
- **注意力融合**：动态权重融合双路径隐状态
- 双数据集支持：ASSISTments 2009 / 2017，前端可一键切换
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

## 训练配置

所有训练脚本使用统一的配置：

| 参数 | 值 | 说明 |
|------|------|------|
| `batch_size` | 64 | 批次大小 |
| `max_seq_len` | 200 | 最大序列长度 |
| `hidden_size` | 128 | LSTM 隐层维度 |
| `epochs` | 100 | 最大训练轮数 |
| `patience` | 10 | Early stopping 耐心值 |
| `seed` | 42 | 随机种子 |
| `lambda_constraint_weight` | 0.01 | λ 约束损失权重 |

各模型独立学习率：

| 模型 | 学习率 | 说明 |
|------|--------|------|
| DKT / DKT-F / LBKT / SR-DKT | 0.001 | 标准学习率 |
| SAKT | 5e-4 | 较小学习率（Transformer 类） |
| AKT | 1e-4 | 最小学习率（注意力机制复杂） |

## 消融实验设计

SR-DKT 消融实验包含 6 组配置，验证各组件贡献：

| 配置名 | 说明 | 验证点 |
|--------|------|--------|
| **Full SR-DKT** | 完整模型：双路径 LSTM + 差异化遗忘 + 注意力融合 | 基线对比 |
| **w/o Storage** | 去掉 lstm_s，只用 lstm_r，S 固定为 0.5 | Storage 分支必要性 |
| **w/o Retrieval** | 去掉 lstm_r，只用 lstm_s，R 固定为 0.5 | Retrieval 分支必要性 |
| **Shared LSTM** | lstm_s 和 lstm_r 合并为一个共享 LSTM | 双路径独立性优势 |
| **Uniform Decay** | λ_s = λ_r（去掉差异化遗忘） | 差异化遗忘必要性 |
| **Mean Fusion** | α_s = α_r = 0.5（固定平均融合） | 注意力机制必要性 |

每个消融变体封装为独立的类实现，训练后自动生成对比表格和柱状图。

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

### 消融实验（ASSISTments 2009）

| 变体 | AUC | ACC | 说明 |
|------|------|------|------|
| **Full SR-DKT** | 0.826 | 0.783 | 完整模型：双路径 LSTM + 差异化遗忘 + 注意力融合 |
| w/o Storage | 0.699 | 0.623 | 去掉 Storage 分支，AUC 下降 12.7% |
| w/o Retrieval | 0.610 | 0.377 | 去掉 Retrieval 分支，AUC 下降 21.6% |
| Shared LSTM | 0.798 | 0.761 | 合并双路径，AUC 下降 2.8% |
| Uniform Decay | 0.682 | 0.643 | 统一遗忘速率，AUC 下降 14.4% |
| Mean Fusion | 0.815 | 0.775 | 固定平均融合，AUC 下降 1.1% |

## 文件说明

| 文件 | 作用 |
|------|------|
| `model.py` | SR-DKT 模型：双路径 LSTM + S/R 双状态头 + 差异化遗忘 + 注意力融合 |
| `app.py` | Streamlit 前端主入口，支持 2009/2017 数据集切换 |
| `train.py` | 训练脚本，支持 SR-DKT 多输出、λ 约束损失、early stopping（patience=10） |
| `preprocess.py` | 读取或生成 ASSISTments 2009 数据，计算 `delta_t`，编码知识点 |
| `preprocess_2017.py` | 读取 ASSISTments 2017 数据，列名映射和清洗 |
| `export_states.py` | 导出测试学生各知识点最新 S/R 状态到 `student_states.pkl` |
| `harness_agent.py` | 根据 S/R 状态生成结构化学习推荐 |
| `evaluate.py` | 输出模型层 AUC/ACC/F1 与推荐层 WCG、HR@10、NDCG@10 |
| `demo.py` | 命令行展示单个学生的知识状态、推荐和评估指标 |
| `baselines/baseline_models.py` | 基线模型定义：DKT, DKT-F, SAKT, AKT, LBKT |
| `baselines/run_baselines.py` | 批量训练基线模型，独立学习率配置，打印对比表格 |
| `ablation/ablation_study.py` | SR-DKT 消融实验（6 组配置），生成对比表格和柱状图 |

## 消融变体类说明

每个消融配置封装为独立的类：

| 类名 | 配置 | 关键改动 |
|------|------|----------|
| `SRDKT` | Full SR-DKT | 完整模型（基类） |
| `SRDKTWithoutStorage` | w/o Storage | 只保留 lstm_r，S 固定为 0.5 |
| `SRDKTWithoutRetrieval` | w/o Retrieval | 只保留 lstm_s，R 固定为 0.5 |
| `SRDKTSharedLSTM` | Shared LSTM | 单一共享 LSTM 替代双路径 |
| `SRDKTUniformDecay` | Uniform Decay | λ_s = λ_r 单一遗忘速率 |
| `SRDKTMeanFusion` | Mean Fusion | α_s = α_r = 0.5 固定权重 |

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