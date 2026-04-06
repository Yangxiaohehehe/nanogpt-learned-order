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

## 14. 最新阶段进展：Block32、AR-likeness 与候选池排序实验

在 16-block 设置下，许多现象已经比较清楚，但仍存在一个疑问：

> 当前 block 粒度是否过粗，以至于一些 trajectory signal 太容易被任务简化效应掩盖？

因此后续引入了 **32-block 设置**：

- 序列长度仍为 256
- `block_len = 8`
- 一共 `32` 个 block
- block 内部仍固定为 l2r

这样做的目的是：

1. 让顺序问题比 16-block 更细粒度  
2. 保留 block-level 的稳定性  
3. 检查 AR / Random 差异是否在更细粒度下更明显  

---

### 14.1 AR-likeness score：在 random checkpoint 内部比较 AR mode 与 Random mode

后续引入了一个新的诊断问题：

> 在**同一个 random checkpoint** 下，只切换 `mode=AR` 和 `mode=Random`，能否仅依靠 trajectory 统计量稳定地区分两者？

这里定义了一组 early-trajectory component：

1. `area`
   - 前若干步 reveal loss 的平均值
   - 代表“前期整体是否更低”
2. `slope`
   - `L_1 - L_K`
   - 代表“前期下降是否更快”
3. `variance`
   - 前若干步 loss 本身的方差
   - 代表“点相对均值是否分散”
4. `total variation`
   - 相邻 step loss 差值绝对值之和
   - 代表“曲线是否平滑、是否带 staircase 感”
5. `late_drop`
   - early 末步到全 trajectory 末步之间仍发生多少下降
   - 代表“改善是否被拖到后半段”

在 `block32`、`200 batches` 平均的设定下，`AR-likeness` 结果明显比 16-block 更强：

- 在 random-only checkpoint 上，`AR mode` 与 `Random mode` 已经可以被更稳定地区分
- 最有用的项不再是单独的 `variance`
- 更有效的组合变成：
  - `area`
  - `slope`
  - 尤其是 `area + slope`

这说明：

> 当 block 粒度变细后，AR 的“前期更快组织上下文”的优势在同一个 random backbone 中会更明显地显现出来。

但另一方面，后续 search benchmark 也说明：

> “能区分 AR 与 Random trajectory” 不等于 “直接把 random order 推回近似 l2r”。  
> 也就是说，AR-likeness score 作为诊断量是有效的，但直接作为 search signal 时仍不够稳定。

---

### 14.2 候选池排序实验：这些 trajectory 指标是否真的偏好 l2r？

为了更直接回答“这些指标到底会不会偏好 l2r”，后续设计了一个更严格的候选池实验：

- 固定一个 `block32 random checkpoint`
- 固定一个候选池：
  - 多个 random block orders
  - 再额外加入 `l2r`
- 对每个候选顺序，在 `200 batches` 上取平均分
- 然后看 `l2r` 在不同指标下的排名

最关键的一轮配置为：

- `block32`
- `batch_size = 64`
- `num_batches = 200`
- `256 random candidates + 1 l2r candidate`

结果非常关键：

- `l2r` 在 `area` 下排第 1
- `l2r` 在 `slope` 下排第 12
- `l2r` 在 `variance` 下排第 216
- `l2r` 在 `total_variation` 下排第 1
- `l2r` 在 `late_drop` 下排第 4
- `l2r` 在 `area_plus_slope` 下排第 1
- `l2r` 在 `area_plus_tv` 下排第 1

这个结果说明了两件很重要的事：

1. **这些指标并不是和 l2r 无关**
   - 相反，在更大的候选池中，`l2r` 在多个关键指标下依然能排在最前面

2. **不同“波动类”指标含义完全不同**
   - `variance` 差，并不表示 AR 不稳定
   - 它只是说明 AR 在前 8 步下降很快，因此点相对均值更分散
   - `total_variation` 高，则说明 AR 的前期下降虽然幅度大，但相邻步之间更平滑、更连续

因此现在可以更明确地说：

> `variance` 不适合作为主信号；  
> 真正更接近“AR 优点”的是：
> - `area`
> - `total_variation`
> - `late_drop`
> - 以及组合指标 `area_plus_tv`

换句话说：

> AR 的优势更像是“前期更低、更平滑、且改善不被拖到后段”，  
> 而不是单纯“方差更小”。

---

### 14.2 补充：完整顺序（full-order）评分实验

后续又将同一候选池实验扩展为同时计算：

- `early_*` 指标：只看前 `25%` block reveal steps
- `full_*` 指标：看完整 `32` 个 block steps 的整条 trajectory

设置保持不变：

- `block32`
- `256 random candidates + 1 l2r candidate`
- `200 batches`
- `batch_size = 64`

在这组 full-order 指标下，`l2r` 的排名为：

- `full_area`: 第 `1`
- `full_slope`: 第 `248`
- `full_variance`: 第 `1`
- `full_total_variation`: 第 `1`
- `full_max_step_jump`: 第 `1`
- `full_area_plus_tv`: 第 `1`
- `full_area_plus_slope`: 第 `234`

这个结果很关键，因为它说明：

1. `l2r` 的优势并不只存在于前 `25%` 的 early window  
2. 在完整 trajectory 上，`l2r` 依然会被“低 + 稳”类指标稳定排到最前  
3. 真正不稳定、容易误导的依然是 `slope` 一类指标

但这里要特别注意两个解释上的区别。

#### (1) `full_area` 的含义

`full_area` 排第 1，**并不等于**“AR 在每个阶段都同样更低”，也不等于“它的优势机制和 early area 完全一样”。

更准确地说：

- `early_area` 低，主要表示 AR 在前期更快组织上下文，因此前几步 loss 更低
- `full_area` 低，则表示 **把整条 trajectory 平均起来以后**，AR 的整体 loss 仍然更低

这里面当然包含了一个重要原因：

- Random 在前期 loss 很高，确实会把 full-area 拉大很多

所以：

> `full_area` 更像是“整条轨迹的平均质量更好”，  
> 而不是“AR 在 full window 上和 early window 里靠同一种机制获胜”。

换句话说，`full_area` 的结果支持：

- AR 的前期优势足够强
- 强到即使后期 Random 可能更低，整条轨迹平均下来 AR 依然更优

#### (2) `early_total_variation` 和 `full_total_variation` 都低，但原因不完全一样

这也是一个很重要的区分。

`early_total_variation` 低，主要表示：

- AR 在前期虽然下降快
- 但相邻 reveal step 之间的变化更连续、更平滑
- 没有 Random 那种明显的 staircase / delayed-release 结构

所以 early-TV 低更像是：

> **前期组织过程更顺滑**

而 `full_total_variation` 低，除了包含上述原因之外，更重要的还在于：

- AR 更早进入平台期
- 后续很多 step 的变化已经变得很小
- 因此整条曲线累计起来的总变化量也更低

所以 full-TV 低更像是：

> **前期平滑 + 更早到达平台期**

也就是说：

- `early_tv` 主要刻画“前期怎么下降”
- `full_tv` 则同时刻画“前期怎么下降”以及“之后多久进入稳定平台”

因此我们现在可以更细地表述 AR 的优势：

> AR 的优势并不是单一的“更低”或“更稳”，  
> 而是：
> - 前期更快建立有效上下文
> - 前期下降更连续、更少 staircase
> - 并且更早进入稳定平台期

---

### 14.3 研究结论的进一步收敛

经过 block32 和大候选池实验后，现在已经可以区分两类结论：

#### 已经基本证明的事情

1. 仅从 AR / Random trajectory 差异中抽取出的统计性质，**确实包含会偏好 l2r 的 signal**
2. 这些 signal 不需要显式给出 l2r label，依然会在大样本平均意义上把 l2r 排到前面
3. 最可信的 signal family 已经逐渐收敛到：
   - early area
   - total variation
   - late-drop penalty
   - 以及 `area + stability` 组合

#### 还没有完全证明的事情

1. 这些 signal 是否已经足以让一个训练好的 random backbone **事后** recover 到近似 l2r
2. 这些 signal 如何在训练中被稳定传播，进而真正改变 backbone 的 order bias

所以当前最合理的表述是：

> 我们已经证明了 signal validity，  
> 但 training propagation 仍需进一步验证。

---

## 15. order head 角色的重新定义：从 utility ranker 到 preference module

上述结果还带来了另一个重要变化：

> `order head` 不应再被理解为“直接恢复最终全局顺序”的模块，  
> 而更适合作为一个“顺序偏好模块（preference module）”。

原因是：

- 直接从 noisy residual utility 回归 block rank，容易被 late easy block 欺骗
- 直接让它输出全局近似 l2r 顺序，任务过重
- 但让它学习“候选顺序 A 是否优于候选顺序 B”，则更符合当前证据

因此，当前代码已经开始改成：

1. 对一个样本采样多个 candidate orders
2. 用 early-shaping quality 比较这些候选
3. 选出 `preferred order` 与 `other order`
4. 让 order head 学习：
   - preferred 的 prefix 概率要高于 other

这意味着：

- `order head` 不再是 utility regressor
- 而是 **pairwise order preference learner**
- 它的角色更像：
  - order sampler
  - proposal distribution learner
  - curriculum controller

一句话总结就是：

> 不是不要 order head，  
> 而是要把它从“最终顺序恢复器”改成“训练过程中逐步引导 order 分布的偏好模块”。

---

## 16. 当前阶段的综合结论

到目前为止，这条研究线已经收敛出一个比较清晰的结论：

> 我们并不需要直接把 l2r 当作 teacher label；  
> 只要从 AR / Random trajectory 的统计差异中抽取出“前期更低、更平滑、且改善不拖到后段”的性质，这些性质本身就已经足以在大样本平均意义上显著偏好 l2r。  
> 因此，下一阶段的关键问题不再是“有没有这样的信号”，而是“如何把这些信号以 preference / curriculum 的方式稳定传播到训练过程中，让模型逐步形成偏向 l2r 的 order policy”。

---
