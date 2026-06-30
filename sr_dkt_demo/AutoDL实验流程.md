# AutoDL 租卡 + 完整实验流程

> 最后更新：2026-06-17

---

## 总览

| 项目 | 内容 |
|------|------|
| 数据集 | MOOCCubeX (30,001 KCs, 1.2M 学生, 133M 交互) |
| 训练子集 | **10% 分层采样**（只采样 train，val/test 全量），见第五步 5.0 |
| 标准基线 | 在官方 **pyKT** 框架跑（DKT/DKVMN/SAKT/AKT/simpleKT/DKT-Forget） |
| 自研模型 | SR-DKT 主模型 + 7 组消融 + 行为基线（LBKT/BEKT）+ DKT-Vec 对齐锚点 |
| 对比策略 | 方案②：标准基线用 pyKT，SR-DKT 用自研框架，用 DKT-Vec 双框架对齐验证 |
| 推荐 GPU | RTX 3090/4090 (24GB VRAM) |
| 预估耗时 | 10% 子集下较全量大幅降低（按实测为准） |

> **架构修订提醒（2026-06）**：SR-DKT 的差异化遗忘已修订（衰减作用于隐状态 + λ 独立可学习），旧 checkpoint 失效，需重新训练；详见 README「核心公式」与主文档第 20 节。

---

## 第一步：注册 AutoDL

1. 打开 https://www.autodl.com
2. 注册 → 实名认证 → 充值（建议充 400 元）
3. 进入「算力市场」

---

## 第二步：租卡

筛选条件：

```
GPU: RTX 3090（24GB 显存）
CPU: ≥ 8 核
内存: ≥ 32GB
系统盘: ≥ 30GB
数据盘: ≥ 50GB（要放 5.8G 数据 + 模型 + 结果）
镜像: PyTorch 2.1+ / CUDA 12.1+ 之类
```

**推荐镜像**：选 `PyTorch 2.3.0 / Python 3.11 / CUDA 12.1` 或更新的官方镜像。

**注意**：3090 有按量计费（按时）和包天/包周两种。先按量租 1 小时测试环境，确认跑通后再切包天或直接按时。

---

## 第三步：上传文件

需要传输两部分：

| 内容 | 大小 | 传输方式建议 |
|------|------|------------|
| 项目代码（.py / .md / .txt） | ~200 KB | SSH 终端直接 git clone 或拖拽 |
| MOOCCubeX 预处理数据 | **~6 GB** | SCP 命令行（最快） |

---

### 3.1 获取 AutoDL SSH 连接信息

租好 GPU 实例后，在 AutoDL 控制台 → 实例详情 → 找到：

```
SSH 命令：ssh -p <端口号> root@<公网IP>
密码：     xxxxxxxx
```

记下这三个值：**IP、端口、密码**。

---

### 3.2 传输代码（二选一）

#### 方式 A：git clone（推荐——免打包）

在 AutoDL 的 SSH 终端里直接拉代码（前提：代码已推到 GitHub / Gitee）：

```bash
cd /root/autodl-tmp
git clone https://github.com/你的用户名/你的仓库.git SR-DKT-code
# 代码在 SR-DKT-code/sr_dkt_demo 子目录下，后续所有命令都在该目录执行
```

> ⚠️ **本项目代码在仓库的 `sr_dkt_demo/` 子目录里**，所以项目根是
> `/root/autodl-tmp/SR-DKT-code/sr_dkt_demo`。本文档后续全部以此为准。
> 务必保证**代码和 `data/` 数据在同一个目录下**，不要把数据传到别处。

#### 方式 B：SCP 上传

在**本地终端**（Git Bash / PowerShell / CMD）执行：

```bash
# 用 Git Bash（推荐）：把整个 sr_dkt_demo 放到 SR-DKT-code 下
scp -P <端口号> -r "C:/Users/10245/Desktop/SR-DKT/sr_dkt_demo" root@<IP>:/root/autodl-tmp/SR-DKT-code/
```

> ⚠️ 上面命令会把整个目录发上去（含 `data/` 子目录，即 ~6GB 数据）。如果只想发代码不含数据，先删掉本地的 `data/` 或单独 scp。

---

### 3.3 传输数据（SCP 命令行，最快）

数据有 **~6 GB**（`data/mooc/` 目录），用 Web 上传太慢（可能断），必须用 SCP。

#### 在本地 Git Bash 执行：

```bash
# 上传整个 data/mooc 目录到远程的 /root/autodl-tmp/SR-DKT-code/sr_dkt_demo/data/
scp -P <端口号> -r \
  "C:/Users/10245/Desktop/SR-DKT/sr_dkt_demo/data/mooc" \
  root@<IP>:/root/autodl-tmp/SR-DKT-code/sr_dkt_demo/data/
```

> 如果你的网络上行带宽是 50 Mbps，6 GB 大约需要 **15–20 分钟**。
> SCP 断点续传不支持，如果断了需要重新传。可以改用 `rsync`（见 3.4）。

#### 如果本地没有 SSH 客户端：

- **Git Bash**（装 Git 就自带）：右键文件夹 → Git Bash Here
- **Windows 10/11 自带 SSH**：打开 CMD → `ssh -p <端口> root@<IP>`

---

### 3.3.1 传输 pyKT 基线数据（约 7.4G，pyKT 实验必需）

跑 pyKT 标准基线（5.3）和双框架对齐（5.2）都依赖 pyKT 序列数据。**只需传
`pykt_ready/` 这一个子目录**，父目录里的 `interactions.csv` / `train.csv` 等
中间产物不用传。

```bash
# 在本地 Git Bash 执行（端口/IP 换成你的实例）
scp -P <端口号> -r \
  "C:/Users/10245/Desktop/SR-DKT/sr_dkt_demo/data/mooc_pykt/pykt_ready" \
  root@<IP>:/root/autodl-tmp/SR-DKT-code/sr_dkt_demo/data/mooc_pykt/
```

> 目标写到 `data/mooc_pykt/`（不带 `pykt_ready`），`scp -r` 会自动建出
> `pykt_ready/` 子目录。传完确认 6 个文件齐全：
> `ls /root/autodl-tmp/SR-DKT-code/sr_dkt_demo/data/mooc_pykt/pykt_ready/`
> 应看到 `train_valid_sequences.csv`(6.1G) / `test_sequences.csv` /
> `test_window_sequences.csv` / `id_maps.json`(67M) / `data_config_moocx.json` /
> `meta_sequences.json`。

---

### 3.4 备选：rsync（支持断点续传）

⚠️ **Windows 的 Git Bash 默认不带 `rsync`**（自带的 OpenSSH 只有 `scp`/`ssh`）。
如果网络不稳、想要断点续传，先装一次：

```bash
conda install -c conda-forge rsync   # 或用其它方式装 rsync
```

装好后再用 rsync（断了可重跑，只传差异部分）：

```bash
rsync -avzP -e "ssh -p <端口号>" \
  "C:/Users/10245/Desktop/SR-DKT/sr_dkt_demo/data/mooc/" \
  root@<IP>:/root/autodl-tmp/SR-DKT-code/sr_dkt_demo/data/mooc/
```

- `-avzP`：归档模式 + 压缩 + 显示进度
- 断了可以重新执行，rsync 只传差异部分

---

### 3.5 备选：AutoDL 网盘挂载（不用上传数据）

AutoDL 支持挂载百度网盘 / 阿里云盘到实例。如果你已经把数据上传到网盘：

在 AutoDL 控制台 → 实例详情 → 网盘挂载 → 授权 → 挂载到 `/root/autodl-tmp`。

然后直接在 SSH 里访问你自己的网盘文件，不需要传输。

---

### 3.6 验证数据完整性

SSH 进实例后：

```bash
ls -lh /root/autodl-tmp/SR-DKT-code/sr_dkt_demo/data/mooc/
```

应该看到：

```
train.pkl      4.6G
val.pkl        589M
test.pkl       584M
meta_mooc.json  588B
skill_encoder_mooc.pkl  1.2M
entities/
prerequisites/
relations/
```

如果文件大小不对（比如 `train.pkl` 只有几百 MB），说明传输断了，用 rsync 重传。

---

### 3.7 最终目录结构

```
/root/autodl-tmp/SR-DKT-code/sr_dkt_demo/
├── data/
│   └── mooc/
│       ├── train.pkl              (4.6 GB)
│       ├── val.pkl                (589 MB)
│       ├── test.pkl               (584 MB)
│       ├── meta_mooc.json         (588 B)
│       ├── skill_encoder_mooc.pkl (1.2 MB)
│       ├── entities/
│       ├── prerequisites/
│       └── relations/
├── model.py
├── train.py
├── evaluate.py
├── ablation_study.py
├── harness_agent.py
├── preprocess_mooc.py
├── baselines/
│   ├── baseline_models.py
│   ├── run_baselines.py
│   └── __init__.py
├── requirements.txt
└── README.md
```

---

## 第四步：安装依赖

```bash
cd /root/autodl-tmp/SR-DKT-code/sr_dkt_demo

# 安装 PyTorch 依赖（镜像已装 torch，只需补装）
pip install scikit-learn tqdm -q

# 可选：安装进度监控工具
pip install gpustat -q
```

验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 应输出: 2.x.x True

python -c "
import sys; sys.path.insert(0,'.')
from model import SRDKT
from baselines.baseline_models import DKT, DKT_F, SAKT, AKT, DKVMN, LBKT, BEKT
print('All imports OK')
"
```

---

## 第五步：运行实验

### 使用 screen 防止断连

```bash
# 创建一个持久化终端
screen -S sr_dkt

# 如果意外断连，重连后用：
# screen -r sr_dkt
```

### 5.0 切 10% 子集 + 对齐 pyKT（先做，几分钟）

```bash
cd /root/autodl-tmp/SR-DKT-code/sr_dkt_demo
mkdir -p logs   # 否则下方 tee logs/xxx.log 会报错

# ① 从全量分层切 10% 训练子集（val/test 全量复用）
python subsample_mooc.py --src-dir data/mooc --out-dir data/mooc_10pct --ratio 0.1 --seed 42

# ② 把 pyKT 训练 fold 过滤到同一批 10% 学生，并清洗非法 sequence token
python filter_pykt_to_subset.py \
  --ids data/mooc_10pct/subset_train_user_ids.json \
  --pykt-dir data/mooc_pykt/pykt_ready \
  --out-dir  data/mooc_pykt_10pct/pykt_ready
```

> **数据指向方式（重要，两类脚本不一样）**：
> - `train.py` / `baselines/run_baselines.py` **支持 `--data-dir`**，直接
>   `--data-dir data/mooc_10pct` 即读 10% 子集，无需软链接。5.1/5.2/5.4 已带上。
> - `ablation/ablation_study.py` / `evaluate.py` **不支持 `--data-dir`**，硬编码读
>   `data/mooc`（全量）。要让它们也跑在 10% 子集上，需先把 `data/mooc` 指向子集
>   （见 5.5 的软链接步骤），跑完再还原；否则消融会在全量上训练、非常慢。

### 5.1 SR-DKT 训练

```bash
python train.py --dataset mooc --data-dir data/mooc_10pct --lambda_weight 0.01 2>&1 | tee logs/train.log
```

输出文件：`best_model_mooc.pt` + `data/training_logs.json`。训练后重点检查：λ_s/λ_r 是否收敛、是否保持 λ_r>λ_s、Constraint 是否不再恒为 0。

### 5.2 双框架对齐验证（DKT-Vec vs pyKT-DKT）

```bash
# 自研框架跑 pyKT 同款标准 DKT（向量输出 + gather 下一题）
python baselines/run_baselines.py --dataset mooc --data-dir data/mooc_10pct --model DKT-Vec 2>&1 | tee logs/dktvec.log
```

与下方 pyKT 的 DKT 在同一 10% 数据上比 AUC，差 `< ~0.005` 即证明两框架等价。

### 5.3 标准基线（官方 pyKT，数据指向 data/mooc_pykt_10pct）

```bash
cd /path/to/pykt-toolkit/examples
# 将 data/mooc_pykt_10pct/pykt_ready/data_config_moocx.json 合并进 pyKT configs/data_config.json
for m in dkt dkvmn sakt akt simplekt dkt_forget; do
  CUDA_VISIBLE_DEVICES=0 python wandb_train.py --dataset_name=moocx --model_name=$m --emb_type=qid --fold=1 --use_wandb=0 --add_uuid=0
done
```

### 5.4 行为增强基线（自研框架，pyKT 没有）

```bash
# 可单跑并行；--model 为单数形式
python baselines/run_baselines.py --dataset mooc --data-dir data/mooc_10pct --model LBKT 2>&1 | tee logs/lbkt.log
python baselines/run_baselines.py --dataset mooc --data-dir data/mooc_10pct --model BEKT 2>&1 | tee logs/bekt.log
```

> ⚠️ 参数是 `--model`（单数），一次一个模型，结果写独立 JSON，适合多 screen 并行。不带 `--model` 则顺序跑全部。

### 5.5 消融实验（7 组）

消融脚本是 `ablation/ablation_study.py`（带 `--variant`、支持 mooc），**不是**根目录
那个旧的 `ablation_study.py`。它不支持 `--data-dir`，读的是 `data/mooc`。要在 10%
子集上跑，先把 `data/mooc` 临时指向子集：

```bash
# ① 让 data/mooc 指向 10% 子集（hardlink 的 val/test 不受影响）
mv data/mooc data/mooc_full
ln -s mooc_10pct data/mooc

# ② 跑 7 组消融（不带 --variant 顺序跑全部；带 --variant 可单跑并行）
python ablation/ablation_study.py --dataset mooc 2>&1 | tee logs/ablation.log
#   单跑示例：python ablation/ablation_study.py --dataset mooc --variant no_video

# ③ 跑完还原 data/mooc 指回全量
rm data/mooc && mv data/mooc_full data/mooc
```

7 组配置：full / no_storage / no_retrieval / shared_decay / no_attention / single_path / no_video，每组独立训练。

输出文件：`data/ablation_results.json`

### 5.6 综合评估（~1-2 小时）

`evaluate.py` 同样不支持 `--data-dir`，硬编码读 `data/mooc`。评估用的是 val/test
（两者本来就全量），但它会按 `data/mooc/best_model_mooc.pt` 找模型——而 5.1 训练
出的 checkpoint 在 `data/mooc_10pct/best_model_mooc.pt`。所以评估前同样需要 5.5 的
软链接（让 `data/mooc` 指向 `mooc_10pct`），评估完再还原：

```bash
mv data/mooc data/mooc_full && ln -s mooc_10pct data/mooc   # 若 5.5 已做可跳过
python export_states.py --dataset mooc 2>&1 | tee logs/export_states.log
python evaluate.py --dataset mooc 2>&1 | tee logs/evaluate.log
rm data/mooc && mv data/mooc_full data/mooc
```

输出：模型层指标 (AUC/ACC/F1) + 推荐层指标 (WCG/HR@10/NDCG@10)

---

## 第六步：下载结果

实验全部结束后，将结果打包下载：

```bash
cd /root/autodl-tmp/SR-DKT-code/sr_dkt_demo

# 打包结果（模型在 10% 子集目录下；baseline 结果用通配符兜底两处位置）
tar -czf results_mooc.tar.gz \
  data/mooc_10pct/best_model_mooc.pt \
  data/training_logs.json \
  data/mooc_10pct/*baseline_results*mooc*.json \
  data/*baseline_results*mooc*.json \
  data/ablation_results.json \
  logs/ 2>/dev/null

# 下载（在本地终端执行，替换为实际实例IP）
scp root@<实例IP>:/root/autodl-tmp/SR-DKT-code/sr_dkt_demo/results_mooc.tar.gz .
```

或在 AutoDL 控制台 → 文件管理 → 下载 `results_mooc.tar.gz`。

---

## 费用预算

采用 10% 子集后耗时较全量大幅下降，下表为相对量级参考（以实测为准）：

| 实验 | 相对耗时 | 备注 |
|------|------|------|
| 切 10% + 对齐 pyKT | 几分钟 | 纯数据操作 |
| SR-DKT 训练（10%） | 低 | 修订架构后需重训 |
| DKT-Vec 对齐验证 | 低 | 显存约 3GB（vocab≈30002 输出） |
| pyKT 标准基线（10%） | 中 | 6 个模型，pyKT 框架 |
| 行为基线 LBKT/BEKT | 低 | 自研框架 |
| 7 组消融 | 中 | SR-DKT 变体 |

> 10% 子集 + maxlen=200 下，整体远低于全量的 ~75h。建议先按量计费跑通 SR-DKT + DKT-Vec 验证，再批量跑其余。

---

## 注意事项

1. **断连保护**：所有训练命令必须在 `screen` 或 `tmux` 中运行，SSH 断开会杀死前台进程。
2. **磁盘空间**：数据盘至少预留 50GB（数据 5.8G + 模型 0.5G + 日志若干）。
3. **显存监控**：训练中运行 `watch -n 1 gpustat` 查看显存使用，正常应 < 20GB。
4. **日志检查**：如果某个模型训练失败，查看 `logs/` 下对应日志。
5. **按量计费提醒**：如果选按量，实例停止后**只收存储费**（约 0.001元/GB/h），不会收 GPU 费。
6. **先验证后批量**：建议先跑 SR-DKT 训练（8h），确认无误后再批量跑 baselines + ablation。

---

## 快速启动命令汇总

```bash
# 1. 创建 screen 持久会话
screen -S sr_dkt

# 2. 进入项目目录
cd /root/autodl-tmp/SR-DKT-code/sr_dkt_demo

# 3. 创建日志目录
mkdir -p logs

# 4. 先切 10% 子集 + 对齐 pyKT（几分钟）
python subsample_mooc.py --src-dir data/mooc --out-dir data/mooc_10pct --ratio 0.1 --seed 42
python filter_pykt_to_subset.py --ids data/mooc_10pct/subset_train_user_ids.json \
  --pykt-dir data/mooc_pykt/pykt_ready --out-dir data/mooc_pykt_10pct/pykt_ready
# 若日志显示替换 token > 0，查看 data/mooc_pykt_10pct/pykt_ready/filter_clean_report.json

# 5. 依次执行（train/run_baselines 用 --data-dir 指向 10% 子集）
time python train.py --dataset mooc --data-dir data/mooc_10pct --lambda_weight 0.01 2>&1 | tee logs/train.log
time python baselines/run_baselines.py --dataset mooc --data-dir data/mooc_10pct --model DKT-Vec 2>&1 | tee logs/dktvec.log
time python baselines/run_baselines.py --dataset mooc --data-dir data/mooc_10pct --model LBKT 2>&1 | tee logs/lbkt.log
time python baselines/run_baselines.py --dataset mooc --data-dir data/mooc_10pct --model BEKT 2>&1 | tee logs/bekt.log
# 消融/评估不支持 --data-dir：先把 data/mooc 指向子集，跑完还原（见 5.5/5.6）
mv data/mooc data/mooc_full && ln -s mooc_10pct data/mooc
time python ablation/ablation_study.py --dataset mooc 2>&1 | tee logs/ablation.log
time python export_states.py --dataset mooc 2>&1 | tee logs/export_states.log
time python evaluate.py --dataset mooc 2>&1 | tee logs/evaluate.log
rm data/mooc && mv data/mooc_full data/mooc
# 标准基线 DKT/DKVMN/SAKT/AKT/simpleKT/DKT-Forget 在官方 pyKT 跑（见 5.3）

# 6. 打包结果
tar -czf results_mooc.tar.gz data/mooc_10pct/best_model_mooc.pt data/training_logs.json data/mooc_10pct/*baseline_results*mooc*.json data/*baseline_results*mooc*.json data/ablation_results.json logs/ 2>/dev/null

# 6. Ctrl+A D 分离 screen（后台继续运行）
# 7. screen -r sr_dkt 重新连接
```
