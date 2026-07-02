# Looped World Models (LoopWM)

arXiv:2606.18208v1 — 验证实现

## 做了什么

核心贡献的独立复现：参数共享的循环动力学核 + 谱稳定性约束 + 自适应早退。

验证了论文的三个核心机制在 890K 参数/128 维的小尺度下全部生效。

## 测试结果

### 合成验证（9/9 PASS）

```
✅ 谱稳定 — 大输入 100x 跑 20 步 norm 递减
✅ A_bar ∈ (0,1) — 数学保证，零调参
✅ 25 步循环核 — h_norm 全程稳定
✅ Gate 结构 — 形状/范围/变异性正常
✅ 检索精炼循环 — 无 NaN，h_norm 稳定
✅ Poisson 采样 — mean≈5，值域 1-10
```

### 多任务训练（MSE 全观测重构）

在三个任务（navigate/collect/transform）混合训练，15 epoch 收敛:

| 指标 | 初始 | 最终 |
|------|------|------|
| Train Loss | 0.215 | 0.019 |
| Val MSE | 29.39 | 4.29 |
| CosSim | 0.885 | 0.979 |

### 自适应深度差异

| 任务 | AvgLoops | CosSim | 行为 |
|------|----------|--------|------|
| **collect** | **2.00** | 0.993 | 100% 第 2 步早退 |
| **navigate** | 10.00 | 0.991 | Gate 0.7-0.8，阈值敏感 |
| **transform** | 10.00 | 0.957 | 更低门控，确实更困难 |

**结论：自适应深度按场景分配计算量，collect（简单场景）100% 早退，transform（困难场景）跑满 10 步。与论文预期一致。**

## 项目结构

```
loopwm/
├── kernel/
│   └── recurrent.py          ← RecurrentDynamicsKernel（可导入）
│   └── __init__.py
├── textworld_env.py           ← 合成文本环境
├── train.py                   ← 单任务训练
├── train_multitask.py         ← 多任务训练 + 自适应深度分析
├── tests/
│   └── test_dynamics_kernel.py ← 9 个合成测试
└── README.md
```

## 环境

- Python 3.12
- PyTorch 2.12.1
- MPS (Apple M1) / CPU
- 无需 ScienceWorld/Java

## 复现

```bash
pip install torch
python tests/test_dynamics_kernel.py   # 9/9 合成测试
python train_multitask.py              # 多任务训练 + 深度分析
```
