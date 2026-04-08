**项目目标**

我在做一个关于 `AOGPT / block order` 的实验，核心问题是：

- 给定一段 token 序列，把它切成若干 `block`
- 训练时不固定使用标准 `l2r` 顺序，而是研究不同 `block reveal / block training order`
- 我想知道：
  - 模型是否会在 `Random block order` 训练下，自发学出某种接近 `l2r` 的局部或全局结构
  - 能否从模型自身的 trajectory signal 里恢复出更好的 order
  - 最终能否不用显式喂 `l2r`，而是通过结构挖掘和 curriculum，让模型逐步学出更合理的顺序偏置

**核心研究假设**

我目前的核心假设是：

- 模型在 `Random block order` 训练下，并不是完全没有结构
- 它会学到一些局部方向性、局部连续性
- 这些结构可以通过 trajectory-level signal 被挖掘出来
- 然后再反过来用于下一阶段训练，形成自举式 curriculum

也就是说，不是直接监督模型学 `l2r`，而是：

- `random warmup`
- `structure mining`
- `segment aggregation`
- `segment-guided random training`
- `re-mining`

**我目前做过的实验**

1. `AR-likeness benchmark`
- 比较同一个 random backbone 下：
  - `AR mode`
  - `Random mode`
- 目标是验证 trajectory signal 是否真实存在
- 结果：`AR` 和 `Random` 的 trajectory 统计差异是存在的，signal 不是假的

2. `metric candidate leaderboard`
- 在候选池里放入 `l2r + random orders`
- 用不同 signal 排序，看能不能把 `l2r` 排到前面
- 结果：
  - `area / total_variation / area_plus_tv` 这类指标能够稳定识别并偏好 `l2r`
- 这说明：
  - signal 有效
  - 问题不在于 signal 不存在

3. `staged prefix / local search`
- 试过只优化 prefix 或局部 swap 搜索
- 结果：
  - 可以找到更高 signal 的 order
  - 但这些 order 并不会自然接近 `l2r`
- 说明：
  - “优化 signal” 不等于 “恢复 `l2r`”

4. `order head preference probe`
- 训练 `order head` 去学 pairwise preference
- 结果：
  - head 能学会 preference task
  - 但学到的东西并不会自动泛化成 `l2r-like preference`
- 说明：
  - 问题不是 head 不会学
  - 而是 candidate / supervision 太弱

5. `structured_candidate_benchmark`
- 用 frozen random checkpoint 对所有 ordered pairs `(i,j)` 打分
- 现在 pair score 采用：
  - 前两个 block step 的 `area + tv`
- 再把高分 pair 聚合成 `segments`
- 用这些结构化 segments 构造 candidate pool
- 再比较：
  - `structured pool`
  - `random pool`
- 结果：
  - structured pool 明显优于 random pool
  - 在 `prefix_mean_index / adjacent_pairs / longest_run / kendall_tau / early_area_plus_tv` 上都有提升
- 这是目前非常关键的正结果

6. `segment curriculum`
- 从头 random 训练一个模型
- 训练一段后跑 `structured_candidate_benchmark`
- 得到 top pairs 和 aggregated segments
- 下一阶段训练时混合：
  - pure random
  - segment-guided random
- 然后继续训练，再重新挖结构
- 这是当前主线实验

**当前代码主线**

目前已经有这些脚本：

- `structured_candidate_benchmark.py`
  - 挖 pair score
  - 聚合 top pairs 成 segment
  - 比较 structured pool vs random pool

- `segment_curriculum_runner.py`
  - 跑 staged curriculum
  - 流程是：
    - warmup random
    - benchmark
    - segment-guided mixed training
    - benchmark
    - 再训练

- `train.py`
  - 已支持：
    - `use_order_head = False`
    - `segment_guided_ratio`
    - `segment_source_json`
    - segment-guided random sampling

**我当前跑过的规模**

1. `block16`
- 效果最好，最终聚合顺序已经很接近 `l2r`
- 最终聚合顺序相对 `l2r`：
  - `tau = 0.8167`
  - `distance = 0.0917`

2. `block32`
- 目前最平衡、最适合作为主实验线
- 结构明显比 random 强
- 最终聚合顺序相对 `l2r`：
  - `tau = 0.6371`
  - `distance = 0.1815`

3. `block64`
- 仍然有明确正结果
- 说明方法不是只在小规模成立
- 但全局整合明显更难
- 最终聚合顺序相对 `l2r`：
  - `tau = 0.2857`
  - `distance = 0.3571`

**目前最重要的发现**

1. signal 是真的
- `area / tv / area_plus_tv` 这类 trajectory signal 确实能区分好坏 order
- 也确实能偏好 `l2r`

2. 模型在 random 训练下会学到局部连续结构
- 特别是在 pair 和 segment 层面
- 常常能自动恢复出连续的 block 段

3. 真正的瓶颈不是 signal，也不是 order head
- 而是 `candidate generation`
- 垃圾候选太多，导致 global order 很难恢复

4. structured candidate / segment curriculum 是有效的
- 把模型当前学到的局部结构反哺进下一阶段训练，是目前最有希望的主线

**目前最大的难题**

最大难题是：

- signal 是 trajectory-level 的
- 但 order space 是 combinatorial 的
- block 数一变大，pair 数暴涨
- 如果继续全量枚举 pair / candidate，计算会爆炸，而且 candidate pool 会重新变脏

例如：
- `block16`: `240` ordered pairs
- `block32`: `992`
- `block64`: `4032`
- 如果以后 token 更长、block 更多，全枚举不可行

所以当前真正卡住的问题是：

> 如何把 trajectory-level signal 变成更平滑、可传播、可局部更新的结构信号，而不是每次都在大候选空间里暴力试错。

**我目前的理解**

目前我认为：

- 这不是一个简单的“继续做 pair ranking”问题
- 下一阶段需要把 sequence-level / trajectory-level score，分解成更局部的 prefix / segment advantage
- 也就是让 signal 从：
  - whole-order score
  变成
  - step-wise / segment-wise utility

这个方向有点类似 RL 里：
- PPO 的 advantage decomposition
- DPO / GRPO 的 relative preference / group ranking

但我不想直接照搬 RLHF 方法，而是想借鉴它们“把全局反馈传播成局部训练信号”的思想

**我现在最想推进的方向**

1. 继续做更大规模时，不再全量枚举 pair
- 需要 prefilter / coarse-to-fine / hierarchy

2. 把最终聚合的 segment 当成新的 unit
- 在 unit space 继续 pair2 / ranking / decode

3. 研究如何从 trajectory signal 提炼 prefix / segment advantage
- 让搜索和训练过程更平滑，而不是只靠离散候选池排序

**一句话总结**

我目前已经证明了：

- random-trained backbone 里存在可用的局部顺序结构
- structured candidate generation 和 segment curriculum 可以稳定优于 random baseline
- `block16 -> block32 -> block64` 说明这条路线不是假的，但随着规模增大，瓶颈迅速转向 candidate pool 垃圾和 global decoding

当前最核心的问题已经不是“有没有 signal”，而是：

> 如何把 trajectory-level signal 变成可扩展、可传播、可解码的局部结构信号，从而在更大 block / 更长 token 长度下恢复出更好的全局 order。