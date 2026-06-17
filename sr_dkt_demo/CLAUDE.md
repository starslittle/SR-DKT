# 项目规则 (SR-DKT)

## 📌 文档同步规则（最重要，每次改动必须遵守）

本项目有**两份必须保持同步**的文档，职责分工如下：

| 文档 | 角色 | 内容范围 |
|------|------|---------|
| `README.md` | 项目说明书 + 通用手册 | 理论/架构、每个脚本的通用用法、训练配置、消融设计、评估指标、结果表、文件说明 |
| `AutoDL实验流程.md` | 云端实操 runbook | 在 AutoDL 上从注册→跑通→下结果的具体步骤（含路径/scp/screen/GPU 等环境细节） |

**规则：任何改动只要影响下列内容，必须同时检查并更新这两份文档的对应位置——**

- 改了**脚本的命令行参数 / 用法**（如新增/删除 `--data-dir`、`--model`、`--variant`）
  → 更新 `README.md` 对应模式 / 脚本说明，**和** `AutoDL实验流程.md` 第五步对应命令
- 改了**实验流程 / 数据路径 / 步骤顺序**
  → 更新 `AutoDL实验流程.md` 对应步骤，**和** `README.md` 的工作流段落
- 改了**默认值 / 配置**（batch_size、子集比例、lr 等）
  → 更新 `README.md`「训练配置」表，必要时同步 `AutoDL实验流程.md`

**反重复原则**：完整命令序列只在 `AutoDL实验流程.md` 维护一份；`README.md` 里用一句话
指向它（"完整云端步骤见 AutoDL实验流程.md"），不要在两处各抄一遍命令，否则必然 drift。

**提交前自检**：改动涉及上述内容时，在 commit 前确认两份文档与代码一致；若只改了其中一份，
要么补另一份，要么明确说明为何不需要。

## 📒 实验进度日志规则（跨会话交接）

`实验进度.md` 是**跨会话/跨机器的进度交接文件**，专为"上下文满了换会话"设计。

**规则：每完成一个环节、或产出一次实验统计（如某模型的 AUC、某步过滤的命中数），
立即更新 `实验进度.md`**——至少更新「阶段状态表」「关键结果」「当前下一步」三处，并刷新顶部
「最后更新」日期。这样任何新会话只要读 `实验进度.md` + 本文件 + `README.md` + `AutoDL实验流程.md`
就能无缝接手，不丢上下文。

它走 git 同步（同下），所以本地/另一台/云端三处一致。

## 🔄 多机同步规则（本地 / 另一台 / AutoDL 云端）

- **代码 `*.py` + 文档（含 `AutoDL实验流程.md`）**：走 git。任一端改完先 `commit && push`，
  其他端 `git pull`。这是三处保持一致的唯一通道。
- **数据**（`data/mooc`、各子集、`pykt_ready`、结果）：不进 git（太大）。
  输入 local→云单向传；结果云→local 下载。不做双向镜像。
- AutoDL 项目根固定为 `/root/autodl-tmp/SR-DKT-code/sr_dkt_demo`。

## 关键事实（避免重复踩坑）

- MOOCCubeX 主实验设定：**10% 分层训练子集（96,044 学生）+ 全量 val/test 评估**。
- `train.py` / `baselines/run_baselines.py` 支持 `--data-dir data/mooc_10pct`；
  **`ablation/ablation_study.py` 和 `evaluate.py` 不支持**，需用软链接让 `data/mooc` 指向子集。
- pyKT 标准基线在官方框架跑；`DKT-Vec` 是跨框架对齐锚点（与 pyKT-DKT 比 AUC，差 <0.005 即等价）。
- `pip install pykt-toolkit` 不含 `examples/wandb_train.py` 和 `configs/`，跑 pyKT 基线需另 clone 源码。
