# 项目阶段性总结：Order Policy / Block-Level 顺序学习

## 1. 研究背景与核心目标

当前研究的核心问题是：

> 在离散序列生成中，不同的 reveal / autoregressive order 会如何影响训练动力学、收敛速度和最终性能？  
> 是否可以在**不直接给出 l2r 先验**的前提下，让模型通过训练信号**自发学到接近 l2r 的好顺序**？

目标不是简单复现 l2r，而是希望回答：

1. 为什么 l2r 会更好  
2. Random order 的问题到底在哪里  
3. 是否能从这些差异中抽取有效的 order signal  
4. 再将这个 signal 传给 order head，让模型自己找到更好的顺序

---

## 2. 最初的关键实验观察

### 2.1 AR / Random 的 loss 曲线差异

最开始的实验图显示：

- **AR / l2r 的 loss 曲线更平稳**
- **Random 的 loss 曲线更像阶梯式下降**
- AR 在前几步下降更快，并较早进入平台
- Random 前中期 loss 长时间偏高，但后期会降得更低

这说明：

> AR 和 Random 的差别不只是“谁更低”，而是 **trajectory 的时间结构不同**。

更准确地说：

- **AR 的优势**：更早、更平稳地组织有效上下文  
- **Random 的特征**：信息释放更晚、改善更拖长、后期任务更容易

---

### 2.2 True-token probability 与 Predictive Entropy 的额外观察

后续又打印了：

1. 每一步真实目标 token 的平均概率
2. 每一步预测分布的熵

得到的结论是：

- AR 更早进入“中等正确率 + 中等熵 + 稳定”的状态
- Random 后期虽然 true-token probability 更高、entropy 更低，但这种确定性释放来得太晚
- Random 后期的低 loss / 低 entropy 更像是：
  - reveal 足够多以后
  - 剩余 token 变成了更简单的条件恢复问题

因此：

> 不能把 Random 后期更低的 loss / entropy / 更高的正确概率，直接当成“好顺序”信号。  
> 它更可能代表“问题在后期才变简单”，而不是顺序本身更优。

---

## 3. 最初的 order head 方案与发现的问题

### 3.1 最初尝试的思路

最开始尝试过直接从 Random trajectory 中构造 utility，例如：

- future loss drop
- 某个 token/block reveal 后，未来若干步 loss 下降了多少

然后：

- 用这个 utility 做 target
- 用 listwise ranking loss 训练 order head
- 再用 learned prefix 去训练 backbone
- 最终形成 4-stage 训练流程：
  1. Random backbone
  2. order head pretrain
  3. policy_backbone
  4. weak joint

---

### 3.2 后续发现的核心问题

随着实验推进，发现：

> Random trajectory 内部的“future loss drop”并不等价于“好顺序”。

原因是：

- Random 后期天然更低
- future loss 更低、entropy 更低，不一定表示顺序更好
- 很可能只是因为 reveal 越多，剩余任务越简单

所以如果直接奖励：

- future mean loss 更低
- future entropy 更低

就会偏向：

> **后期天然容易的 token/block**  
> 而不是  
> **真正应该前置、能帮助结构更早建立的 token/block**

这成为后续 signal 设计需要重点解决的问题。

---

## 4. 研究原则的澄清

后续讨论中，研究原则逐渐明确为：

### 不希望做的事情
- 不直接用 l2r rank 当 target
- 不用 Kendall 距离做训练 loss
- 不把 AR 当 critic / teacher / baseline
- 不在 loss 中显式写“请靠近 l2r”

### 希望做的事情
- 可以观察 AR 和 Random 的差异
- 但只能把这些差异抽象成**trajectory 的统计性质**
- 再用这些统计性质来定义“好顺序”
- 如果模型最终靠近 l2r，那是**结果**，不是**先验**

一句话总结：

> 不能直接拟合 l2r，只能学习“好 trajectory 的性质”。

---

## 5. 从 token-level 转向 block-level

由于 token-level 顺序空间太大、信号太噪，后续决定先做 block-level：

- 原始序列长度仍为 256
- 划分为 16 个 block
- 每个 block 16 个 token
- **block 内部顺序固定为 l2r**
- 只学习 **16 个 block 之间的顺序**

### 转向 block-level 的动机
1. 顺序空间从 256! 降到 16!  
2. block 比单 token 携带更多上下文信息  
3. block-level trajectory 更稳定、更容易分析  
4. 更适合作为第一阶段实验平台

---

## 6. Block-level 第一版 signal 设计

### 6.1 最初设计
在 block-level 上，第一版 signal 设计为：

1. future mean loss
2. future variance

但后来发现：

- 直接用 raw sliding-window absolute values 仍然会偏向后段
- 因为 Random trajectory 天然越到后面越容易

---

### 6.2 后续修正为 residual signal

因此改成：

> **step-wise residual signal**

具体做法：

- 对每个 block reveal step 维护 baseline
- 计算：
  - future mean loss residual
  - future variance residual
- 再组合成 utility

形式上大致为：

\[
U_t = -\alpha \tilde{A}_t - \beta \tilde{V}_t
\]

其中：

- \(\tilde{A}_t\)：future mean loss 相对该 step baseline 的残差
- \(\tilde{V}_t\)：future variance 相对该 step baseline 的残差

这样可以去掉“平均而言越往后越容易”的 step bias。

---

## 7. 训练 block-level order head 后的现象

在 block-level residual 方案下训练第二阶段 `order_head` 时，观察到：

- prefix mean 仍然偏高
- prefix 中最大 index 仍然可以到 15
- 说明 order head 依然会把后面的 block 放进 prefix

这说明：

> 即使做了 step-wise residual，当前 signal 仍然更容易挑出“后期天然更容易 / 更稳定的 block”，  
> 而不是“真正该前置的结构块”。

后续打印出的 residual 大致为：

- mean residual ≈ -0.05
- variance residual ≈ 0.1

这进一步说明：

- baseline 已经去掉了一部分后期天然更低的趋势
- 但还不足以让 signal 变成真正的“前置顺序价值 signal”

---

## 8. Pair 枚举实验：验证 backbone 的局部顺序感知能力

为了判断问题到底在 backbone 还是在 signal，后续设计了一个关键诊断实验：

### 实验设计
固定一个 Random backbone checkpoint，不训练 order head。  
枚举 16 个 block 的所有有序前缀 pair：

\[
16 \times 15 = 240
\]

即：

- 固定前两个 block 为 `(i,j)`
- 后续 block 按固定规则补齐
- 统计：
  - `prefix2_auc`
  - `prefix4_auc`
  - `full_loss`

---

### 实验结论

1. **backbone 已经具备明显的局部顺序区分能力**
   - 最优和最差 pair 的 prefix2_auc 差距明显

2. **局部方向性是存在的**
   - `(0,1)` 明显优于 `(1,0)`
   - `(1,2)` 明显优于 `(2,1)`
   - `(7,8)` 明显优于 `(8,7)`  
   等等

3. **最优 pair 往往不是前段，而是中后段相邻块**
   - 例如 `(11,12)`, `(12,13)`, `(8,9)`, `(7,8)`

这说明：

> Random backbone 已经学到的是 **局部连续性 + 局部方向性**，  
> 但还没有自然学到 **前置性**。

这也解释了为什么当前 order head 会把后段 block 放进 prefix。

---

## 9. Local swap 实验：验证局部顺序修正能力

进一步设计了 `local_swap_eval`：

- 从 random block order 出发
- 只允许 **adjacent swap**
- 评价函数采用 `prefix_auc_2`
- greedy 地进行 1 步、3 步、10 步 local swap

---

### 结果总结

#### 9.1 有一部分样本完全动不了
很多样本在第一步之后就已经没有可接受的 swap 了：

- `swap_idx = -1`
- `improved = 0`

说明：

> 对这些样本，当前顺序在“只允许 adjacent swap + 只看 prefix2”的局部邻域里，已经是局部最优或平坦区域。

---

#### 9.2 有一部分样本能改 1 步
一些样本能够通过一次局部相邻交换显著改善 prefix2，但之后就停住。

说明：

> 模型已经能修掉一些明显的局部顺序错误。

---

#### 9.3 少数样本能连续改 2～5 步
有些样本在多步 greedy swap 下，prefix2_auc 可以连续下降。

说明：

> Random backbone checkpoint 确实内生地包含一定的局部顺序修正能力。

---

### 关键结论

这批结果说明：

1. **问题不是“模型完全没有顺序信息”**
2. **问题也不是“步数不够”**
3. 真正的问题更像是：

> **当前 local search 容易卡在局部最优**

原因可能包括：

- 动作空间太弱（只允许 adjacent swap）
- 评价函数太短视（只看 `prefix_auc_2`）

---

## 10. 当前阶段最重要的认识

经过这些实验，研究问题已经进一步收敛。

### 目前已经确认的事实
1. Random backbone 已经有局部顺序感知能力  
2. 它已经能感知：
   - 局部连续性
   - 局部方向性
3. 但它不会天然收敛到全局 l2r
4. 当前 residual future mean / variance signal 仍然偏向后段简单块
5. 当前 greedy adjacent swap 也容易卡在局部最优

---

## 11. 当前最重要的 motivation

所以目前最重要的研究动机已经变成：

> **先在 swap / search 框架下，找到一个真正会把顺序往“前置、连续、近似 l2r”方向推的 signal；  
> 再把这个 signal 传播给 order head 去训练。**

也就是说：

### 当前最重要的任务不是
- 继续盲调 order head 架构
- 或盲目继续训练第二阶段

### 而是
- 先把 `local_swap_eval` 当成一个 **signal benchmark**
- 测试不同 quality / signal 是否能把 random order 推向更好的结构
- 再将筛选出来的 signal 作为训练目标传播给 order head

---

## 12. 当前最合理的下一步计划

### Step 1：固定 checkpoint，不训练
在 swap/search 框架下测试不同 signal。

固定：
- checkpoint
- 样本
- 初始 random block order
- 搜索动作

变化：
- 只换 quality / signal

---

### Step 2：比较不同 signal 的效果
观察它们是否能让搜索结果：

- prefix AUC 更好
- 不再总偏向后段
- 局部连续片段更多
- 更前置
- Kendall 对 l2r 上升（只做分析）

---

### Step 3：选出最靠谱的 signal
这个 signal 不要求直接“像 l2r”，但应满足：

- 更前置
- 更连续
- 更稳定
- 不总被后期简单块欺骗

---

### Step 4：再把这个 signal 用作训练目标
一旦 signal 在 local search 上被验证有效，再将其用于：

- order head 的 utility target
- 或 local policy 的 reward
- 或 future order learning 的 curriculum criterion

---

## 13. 当前一句话总结

> 我们已经证明：Random backbone 本身具备局部顺序判别和局部顺序修正能力；当前的主要瓶颈不再是“有没有顺序信号”，而是“什么样的 signal 能把这些局部顺序信息组织成更前置、更连续、近似 l2r 的全局顺序”。因此，下一步最合理的方向是先在 swap/search 实验中筛选 signal，再将筛选出的 signal 传播到 order head 训练中。

---