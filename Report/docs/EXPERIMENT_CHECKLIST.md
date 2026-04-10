# 实验执行清单

这份清单对应当前阶段最推荐的 three-step 路线：

1. `prefix_only`：先验证 prefix signal 是否有效
2. `suffix_only`：在固定 prefix 下验证 suffix continuation signal
3. `two_stage`：把 prefix 和 suffix 串起来做完整顺序构造

所有命令都默认在：

```bash
source /home/devbox/project/bin/activate
cd /home/devbox/project/AOGPT-test-order/nanogpt_learned_order
```

默认 checkpoint：

- `out/out-wikitext103-random-b32/ckpt.pt`

---

## 实验 1：Prefix-only benchmark

目的：

- 验证 `area_plus_tv` 是否能稳定选出更像 l2r 的 prefix
- 不要求完整顺序最优

运行命令：

```bash
python staged_order_benchmark.py \
  --experiment prefix_only \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/staged_prefix_only_b32 \
  --num_batches 50 \
  --batch_size 4 \
  --prefix_len 8 \
  --beam_width 4 \
  --expand_candidates 8 \
  --rollout_repeats 4 \
  --prefix_eval_window 8 \
  --tv_weight 0.3
```

重点看：

- `best_prefix_or_order.mean_prefix_mean_index`
- `prefix_view.mean_adjacent_pairs`
- `prefix_view.mean_kendall_tau`
- 相对 `random_baseline` 是否更好

预期结果：

- `best_prefix_or_order` 的 prefix 更前置
- `prefix_view` 的 contiguous run 更长
- `prefix_view` 的 Kendall 高于 random baseline

如果这一步没有明显提升，就不要继续做 training propagation。

---

## 实验 2：Suffix-only benchmark

目的：

- 在 prefix 已固定时，测试 continuation / suffix 是否能比随机补全更好

### 2A. random prefix 条件下

```bash
python staged_order_benchmark.py \
  --experiment suffix_only \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/staged_suffix_only_b32_random_prefix \
  --num_batches 50 \
  --batch_size 4 \
  --prefix_len 8 \
  --beam_width 4 \
  --expand_candidates 8 \
  --rollout_repeats 4 \
  --suffix_eval_window 8 \
  --tv_weight 0.3 \
  --suffix_prefix_source random
```

### 2B. l2r prefix 条件下

```bash
python staged_order_benchmark.py \
  --experiment suffix_only \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/staged_suffix_only_b32_l2r_prefix \
  --num_batches 50 \
  --batch_size 4 \
  --prefix_len 8 \
  --beam_width 4 \
  --expand_candidates 8 \
  --rollout_repeats 4 \
  --suffix_eval_window 8 \
  --tv_weight 0.3 \
  --suffix_prefix_source l2r
```

重点看：

- `suffix_view.mean_adjacent_pairs`
- `suffix_view.mean_longest_run`
- `best_prefix_or_order.mean_kendall_tau`

预期结果：

- 在 `l2r prefix` 条件下，suffix 应比 random suffix 更连续
- 如果 `random prefix` 条件下提升很小，而 `l2r prefix` 条件下提升明显，说明 suffix 问题更依赖 prefix 质量

---

## 实验 3：Two-stage benchmark

目的：

- 真正测试 prefix signal + suffix continuation 联合后，完整顺序是否比 random 更接近 l2r

运行命令：

```bash
python staged_order_benchmark.py \
  --experiment two_stage \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/staged_two_stage_b32 \
  --num_batches 50 \
  --batch_size 4 \
  --prefix_len 8 \
  --beam_width 4 \
  --expand_candidates 8 \
  --rollout_repeats 4 \
  --prefix_eval_window 8 \
  --suffix_eval_window 8 \
  --tv_weight 0.3
```

重点看：

- `best_prefix_or_order.mean_kendall_tau`
- `best_prefix_or_order.mean_prefix_mean_index`
- `best_prefix_or_order.mean_adjacent_pairs`
- `best_prefix_or_order.mean_longest_run`

预期结果：

- 比 `random_baseline` 更前置
- contiguous run 更长
- Kendall 更高
- 但通常仍然达不到 `l2r_reference`

如果这一步成立，说明：

- prefix signal 是有效的
- suffix continuation 也提供了额外收益
- 接下来可以考虑训练时传播

---

## 实验 4：Order-head preference pretrain

目的：

- 不再让 order head 直接回归 utility
- 而是学习 candidate order preference

当前代码中训练逻辑已经改为 preference 风格。建议先用 block32 config 跑短程实验：

```bash
python train.py config/WikiText103/block32/policy/prefix_policy_order_head.py --max_iters=1000
```

重点看日志：

- `train/preference_loss`
- `val/preference_loss`
- `train/preference_accuracy`
- `val/preference_accuracy`
- `train/prefix_mean_index`
- `val/kendall_tau_l2r`

预期结果：

- `preference_accuracy` 应明显高于 `0.5`
- `preference_gap` 应逐步上升
- order head 生成的 prefix 更前置

---

## 实验 5：Preference-guided backbone training

目的：

- 验证 preference signal 能否真正改变 backbone 的训练 order bias

建议先从较弱版本开始：

- prefix 由更优 candidate / preference 选择
- suffix 暂时随机补全

这一步目前建议在 order-head preference 结果稳定后再做。

预期结果：

- generate-step loss 曲线更像 AR
- prefix metrics 随训练变好
- eval-only Kendall 缓慢上升

---

## 成功标准

如果以下三件事同时成立，就说明这条路线是通的：

1. `prefix_only` 明显优于 random baseline
2. `two_stage` 比 `prefix_only` 进一步提高 full-order metrics
3. preference-trained order head 的 `preference_accuracy` 稳定高于随机

---

## 当前最推荐的主信号

当前阶段优先使用：

- `area_plus_tv`

理由：

- 它在大候选池实验中会把 `l2r` 排到最前面
- 它同时抓住：
  - 前期更低
  - 前期更平滑
- 比单独 `variance` 更合理
- 比单独 `slope` 更不容易被“前面特别差、后面掉很多”的顺序欺骗
