# Project Progress Summary: Order Policy / Block-Level Order Learning

## 1. Research Background and Core Goal

The core research question is:

> In discrete sequence generation, how does the reveal / autoregressive order affect optimization dynamics, convergence speed, and final performance?  
> Can we let the model **spontaneously discover a good order close to left-to-right (l2r)** **without explicitly giving it l2r as a prior**?

The goal is **not** simply to reproduce l2r, but to understand:

1. Why l2r works better  
2. What is fundamentally wrong with Random order  
3. Whether we can extract an effective order signal from these differences  
4. Whether this signal can be propagated to an order head so the model can discover a better order by itself

So this is not just an engineering optimization problem; it is fundamentally an **order policy / generation order learning problem**.

---

## 2. Initial Key Empirical Observations

### 2.1 AR vs Random loss curves

The earliest result showed:

- **AR / l2r loss is much smoother**
- **Random loss drops in a staircase-like pattern**
- AR drops faster in the beginning and reaches a plateau earlier
- Random stays higher for much longer, but can become lower later

This suggests:

> The difference between AR and Random is not just “which one is lower,” but **how the loss trajectory evolves over time**.

More precisely:

- **AR advantage**: useful conditioning information is organized earlier and more smoothly
- **Random behavior**: useful information is released later, improvement is delayed, and the remaining prediction problem becomes easier only in later stages

---

### 2.2 True-token probability and predictive entropy

Later, two additional plots were analyzed:

1. Mean probability assigned to the true token at each step
2. Predictive entropy at each step

These provided the following insights:

- AR reaches a “moderately confident and stable” regime much earlier
- Random achieves higher true-token probability and lower entropy **only much later**
- The low loss / high confidence / low entropy of Random in later steps likely reflects:
  - more tokens having already been revealed
  - the remaining targets becoming easier conditional recovery problems

Therefore:

> We cannot directly treat lower future loss, lower entropy, or higher true-token probability under Random as a “good order” signal.  
> Much of it may simply reflect that the task becomes easier in later phases, not that the order itself is better.

---

## 3. The Initial Order Head Direction and the Main Problem

### 3.1 The initial approach

The original idea was:

- Construct a utility signal directly from Random trajectories
- For example:
  - future loss drop
  - how much future loss decreases after revealing a token/block
- Use this utility as supervision for an order head
- Train the order head with a listwise ranking loss
- Then use the learned prefix order to train the backbone

This naturally led to the 4-stage pipeline:

1. Random backbone pretraining  
2. Order head pretraining  
3. Policy-backbone training  
4. Weak joint / two-pass joint training

---

### 3.2 The problem discovered later

As experiments progressed, a key issue became clear:

> “Future loss decreases under Random” does **not** necessarily mean “this is a good order signal.”

Because from the earlier plots:

- Random naturally becomes easier later
- Future loss being lower later does not necessarily mean the order is better
- It may simply reflect that more of the sequence has been revealed and the remaining problem is easier

So if we directly reward:

- lower future mean loss
- lower future entropy

then the learned signal will be biased toward:

> **tokens/blocks that are naturally easier in late stages**  
> rather than  
> **tokens/blocks that should be revealed early to build useful structure**

This became the central issue in later signal design.

---

## 4. Clarified Research Principles

Through discussion, the guiding principles became much clearer.

### What we do **not** want
- We do **not** want to directly use l2r ranks as supervision targets
- We do **not** want to use Kendall distance as a training loss
- We do **not** want to use AR as a critic / teacher / baseline
- We do **not** want the loss to explicitly say “move toward l2r”

### What we **do** want
- We may observe the differences between AR and Random
- But we only want to abstract them into **trajectory-level statistical properties**
- Then define a “good order” using those properties
- If the model eventually moves closer to l2r, that should be an **emergent result**, not an **explicit prior**

In one sentence:

> We do not want to fit l2r directly; we want to learn the properties of a good trajectory.

---

## 5. Moving from token-level to block-level

Since token-level order learning was too noisy and the search space was too large, the problem was reformulated at the block level:

- Total sequence length remains 256
- Split into 16 blocks
- Each block contains 16 tokens
- **Within each block, order is fixed as l2r**
- Only the order among the 16 blocks is learned

### Motivation for the block-level setup
1. The order search space shrinks from 256! to 16!  
2. A block carries much richer contextual information than a single token  
3. Block-level trajectories are much more stable and easier to analyze  
4. This is a better first-stage experimental setup

---

## 6. First block-level signal design

### 6.1 Initial design
At the block level, the first signal design used:

1. Future mean loss
2. Future variance

But later it became clear that:

- Using raw sliding-window absolute values still biases the signal toward later steps
- Because Random trajectories are naturally easier later

---

### 6.2 Residualized signal design

So the signal was revised into a:

> **step-wise residual signal**

Concretely:

- Maintain a baseline for each reveal step
- Compute:
  - future mean loss residual
  - future variance residual
- Combine them into a utility

In simplified form:

\[
U_t = -\alpha \tilde{A}_t - \beta \tilde{V}_t
\]

where:

- \(\tilde{A}_t\): residual of future mean loss relative to the baseline at step \(t\)
- \(\tilde{V}_t\): residual of future variance relative to the baseline at step \(t\)

This removes the average “later steps are naturally easier” step bias.

---

## 7. What happened after training the block-level order head

After training the stage-2 block-level order head with the residual signal, the following was observed:

- The prefix mean index was still high
- The maximum selected block index could still be as large as 15
- This means the order head was still placing later blocks into the prefix

This suggests:

> Even after residualization, the current signal still tends to favor blocks that are naturally easier or more stable in the later part of Random trajectories,  
> instead of blocks that should be moved forward to establish useful structure earlier.

The measured residuals were roughly:

- mean residual ≈ -0.05
- variance residual ≈ 0.1

This suggests:

- baseline subtraction removes part of the trivial late-stage trend
- but it is still not sufficient to produce a true “early structural usefulness” signal

---

## 8. Pair enumeration experiment: diagnosing local order sensitivity

To determine whether the main issue lies in the backbone or in the signal, a key diagnostic experiment was designed.

### Experiment design
Fix a Random backbone checkpoint, without training an order head.  
Enumerate all ordered pairs of the first two blocks:

\[
16 \times 15 = 240
\]

That is:

- fix the first two blocks as `(i, j)`
- fill the remaining 14 blocks using a fixed rule
- measure:
  - `prefix2_auc`
  - `prefix4_auc`
  - `full_loss`

---

### Main findings

1. **The backbone already has clear local order sensitivity**
   - The gap between the best and worst pairs is substantial

2. **Local directionality already exists**
   - `(0,1)` is clearly better than `(1,0)`
   - `(1,2)` is clearly better than `(2,1)`
   - `(7,8)` is clearly better than `(8,7)`
   - etc.

3. **However, the best pairs are often not from the early part of the sequence, but from the middle or later blocks**
   - e.g. `(11,12)`, `(12,13)`, `(8,9)`, `(7,8)`

This strongly suggests:

> The Random backbone has already learned **local continuity and local directionality**,  
> but it has **not** naturally learned **frontness**.

This also explains why the current order head tends to put later blocks into the prefix.

---

## 9. Local swap experiment: diagnosing local order repair ability

A `local_swap_eval` experiment was then introduced:

- Start from a random block order
- Allow only **adjacent swaps**
- Use `prefix_auc_2` as the quality function
- Perform 1-step, 3-step, and later 10-step greedy local swaps

---

### Main findings

#### 9.1 Some samples cannot be improved at all
For many samples:

- `swap_idx = -1`
- `improved = 0`

already after the first step

This means:

> Under the current search neighborhood (“adjacent swaps only”) and evaluation metric (`prefix_auc_2` only), the current order is already a local optimum or a flat region.

---

#### 9.2 Some samples can improve by one step
Some samples improve substantially with a single adjacent swap, then stop.

This means:

> The model can already repair some obvious local order mistakes.

---

#### 9.3 A small subset can improve for multiple steps
For some samples, `prefix_auc_2` improves over 2–5 greedy swap steps.

This means:

> The Random backbone checkpoint does contain real local order repairability.

---

### Key conclusion from local swap

The local swap experiment suggests:

1. The problem is **not** that the model lacks any order information  
2. The problem is **not** simply that the number of search steps is too small  
3. The real issue is more likely:

> **the search is getting trapped in local optima**

Possible reasons:

- the action space is too weak (adjacent swaps only)
- the evaluation metric is too short-sighted (`prefix_auc_2` only)

---

## 10. What is now clearly understood

At this stage, several things are already clear:

### Facts we now know
1. The Random backbone already has local order sensitivity  
2. It can already detect:
   - local continuity
   - local directionality
3. But it will **not** naturally converge to global l2r on its own  
4. The current block-level residual future mean / variance signal still favors late easy blocks  
5. Greedy adjacent swap search also gets trapped in local optima easily

---

## 11. The current main motivation

This leads to the current primary research motivation:

> Before training a better order head, first use the swap / search framework to identify a signal that genuinely pushes orders toward a **front-loaded, continuous, l2r-like structure**.

In other words:

### The current most important task is **not**
- blindly tuning the order head architecture
- or blindly continuing stage-2 training

### The current most important task **is**
- to use `local_swap_eval` as a **signal benchmark**
- test which quality / signal actually pushes random orders toward a better structure
- and only then propagate that signal into order head training

---

## 12. The most reasonable next-step plan

### Step 1: fix the checkpoint, do not train
Use the swap/search framework to test different signals.

Keep fixed:
- checkpoint
- samples
- initial random block orders
- search actions

Only vary:
- the quality / signal

---

### Step 2: compare different signals
Check whether a signal pushes the search results toward:

- better prefix AUC
- less late-block bias
- more local contiguous segments
- more front-loaded structure
- improved Kendall-to-l2r (for analysis only)

---

### Step 3: select the best signal
The signal does **not** need to explicitly look like l2r, but it should favor:

- frontness
- continuity
- stability
- resistance to being dominated by late easy blocks

---

### Step 4: use the selected signal for training
Once a signal is validated in local search, it can be used as:

- the utility target for order head training
- the reward for a future local policy
- the criterion for selecting better orders during training

---

## 13. One-sentence project summary

> We have already shown that a Random backbone contains local order discrimination and local order repair signals; the main bottleneck is no longer “whether order information exists,” but rather “what kind of signal can organize these local order cues into a more front-loaded, continuous, and approximately l2r-like global order.” Therefore, the most reasonable next step is to use swap/search experiments to benchmark signals first, and only then propagate the selected signal into order head training.

---

## 14. New Progress: Block32, AR-likeness, and candidate-pool ranking experiments

Under the 16-block setting, many qualitative conclusions were already clear, but one important concern remained:

> Is the current block granularity too coarse, making some trajectory signals too easily confounded by task simplification?

To address this, the setup was extended to a **32-block setting**:

- sequence length remains 256
- `block_len = 8`
- total number of blocks = `32`
- within-block order is still fixed as l2r

The motivation was:

1. to make the order problem more fine-grained than the 16-block case  
2. while preserving the stability of block-level analysis  
3. and to test whether AR / Random differences become more visible at finer granularity  

---

### 14.1 AR-likeness score: comparing AR mode and Random mode inside the same random checkpoint

A new diagnostic question was introduced:

> Inside the **same random checkpoint**, if we switch only `mode=AR` versus `mode=Random`, can trajectory-level statistics reliably distinguish the two?

To do this, a family of early-trajectory components was defined:

1. `area`
   - mean reveal loss over the early steps
   - captures whether the early trajectory is low overall
2. `slope`
   - `L_1 - L_K`
   - captures how fast the early trajectory drops
3. `variance`
   - variance of the early losses themselves
   - captures how dispersed the early points are around their mean
4. `total variation`
   - sum of absolute adjacent-step loss differences
   - captures smoothness versus staircase-like behavior
5. `late_drop`
   - how much further loss still drops after the early window
   - captures whether improvement is delayed into later stages

Under the `block32`, `200-batch` setting, AR-likeness becomes much stronger than in the 16-block case:

- on the same random-only checkpoint, `AR mode` and `Random mode` can now be separated more reliably
- the most useful components are no longer centered on standalone `variance`
- stronger combinations now involve:
  - `area`
  - `slope`
  - especially `area + slope`

This suggests:

> Once block granularity becomes finer, the AR advantage of organizing useful context earlier becomes more visible even inside the same random backbone.

At the same time, later search benchmarks also showed:

> Being able to distinguish AR trajectories from Random trajectories does **not** automatically mean that the same score can directly recover an approximately l2r order from a trained random backbone.

So AR-likeness is useful as a diagnostic quantity, but not yet sufficient as a direct search signal.

---

### 14.2 Candidate-pool ranking experiment: do these trajectory metrics actually prefer l2r?

To answer more directly whether these metrics truly favor l2r, a stricter candidate-pool experiment was introduced:

- fix a `block32 random checkpoint`
- build a candidate pool consisting of:
  - many random block orders
  - plus one explicit `l2r` candidate
- evaluate each candidate by averaging over `200 batches`
- then inspect the rank of `l2r` under each metric

The most important setup used:

- `block32`
- `batch_size = 64`
- `num_batches = 200`
- `256 random candidates + 1 l2r candidate`

The result is extremely important:

- `l2r` ranks 1st under `area`
- `l2r` ranks 12th under `slope`
- `l2r` ranks 216th under `variance`
- `l2r` ranks 1st under `total_variation`
- `l2r` ranks 4th under `late_drop`
- `l2r` ranks 1st under `area_plus_slope`
- `l2r` ranks 1st under `area_plus_tv`

This implies two major conclusions:

1. **These metrics are not unrelated to l2r**
   - on the contrary, in a much larger candidate pool, `l2r` still ranks at or near the top under several key trajectory metrics

2. **Different “variance-like” metrics mean very different things**
   - poor rank under `variance` does not mean AR is unstable
   - it only means AR drops much faster early, so the early points are more spread around their mean
   - strong rank under `total_variation` means that although AR drops a lot, it does so more smoothly and continuously across adjacent reveal steps

Therefore we can now state more clearly:

> `variance` is not a good primary signal;  
> what better captures the actual AR advantage is:
> - `area`
> - `total_variation`
> - `late_drop`
> - and especially the combined metric `area_plus_tv`

In other words:

> The AR advantage is better described as “lower earlier, smoother earlier, and not delaying improvement into late stages,”  
> rather than simply “having lower variance.”

---

### 14.2 Supplement: full-order scoring

The same candidate-pool experiment was then extended to compute both:

- `early_*` metrics: computed only over the first `25%` of block reveal steps
- `full_*` metrics: computed over the full `32`-step block trajectory

The setup remained the same:

- `block32`
- `256 random candidates + 1 l2r candidate`
- `200 batches`
- `batch_size = 64`

Under the full-order metrics, the rank of `l2r` becomes:

- `full_area`: rank `1`
- `full_slope`: rank `248`
- `full_variance`: rank `1`
- `full_total_variation`: rank `1`
- `full_max_step_jump`: rank `1`
- `full_area_plus_tv`: rank `1`
- `full_area_plus_slope`: rank `234`

This is important because it shows:

1. the l2r advantage is not limited to the first `25%` early window  
2. over the full trajectory, l2r is still strongly preferred by the “low + smooth” metric family  
3. the unstable and misleading family remains the `slope`-style metrics

However, two distinctions are important here.

#### (1) What `full_area` really means

`full_area` ranking first does **not** mean that AR is uniformly better at every stage in the same way, and it does **not** mean that the mechanism is identical to `early_area`.

More precisely:

- low `early_area` mainly means AR organizes useful context earlier, so the first few steps have lower loss
- low `full_area` means that **after averaging the entire trajectory**, AR still achieves lower overall loss

One obvious reason is:

- Random is much worse in the early phase, and that strongly enlarges the full trajectory area

So:

> `full_area` should be read as “better average quality across the whole trajectory,”  
> not as “the same mechanism that explains `early_area` simply persists unchanged everywhere.”

In other words, the result supports:

- the early AR advantage is strong enough
- strong enough that even if Random can become lower later, AR still wins on the full-trajectory average

#### (2) Why both `early_total_variation` and `full_total_variation` are low, but for different reasons

This distinction is also important.

Low `early_total_variation` mainly means:

- AR drops quickly in the beginning
- but the change across adjacent reveal steps is smoother and more continuous
- without the staircase / delayed-release behavior of Random

So low early-TV is mainly about:

> **a smoother early organization process**

Low `full_total_variation`, however, includes that effect but also reflects something additional:

- AR reaches its plateau earlier
- many later steps change very little
- so the accumulated total variation over the full trajectory also becomes smaller

So low full-TV is better interpreted as:

> **smooth early shaping + earlier arrival at a plateau**

That is:

- `early_tv` mainly characterizes *how the trajectory descends early*
- `full_tv` characterizes both *how it descends early* and *how early it settles into a stable regime*

This gives a more precise statement of the AR advantage:

> AR is not simply “lower” or “more stable” in a generic sense.  
> Its advantage is that it:
> - builds useful context earlier
> - descends more smoothly with less staircase behavior in the early phase
> - and reaches a stable plateau earlier

---

### 14.3 The research conclusion is now more sharply focused

After the block32 and large candidate-pool experiments, it is now useful to separate two types of conclusions.

#### What is now largely established

1. Trajectory-level statistics extracted only from AR / Random differences **do contain signals that favor l2r**
2. These signals do not require direct l2r labels, yet in large-sample averages they still rank l2r very highly
3. The most credible signal family is converging toward:
   - early area
   - total variation
   - late-drop penalties
   - and more generally `area + stability` style combinations

#### What is not yet fully established

1. Whether these signals are already sufficient to recover an approximately l2r order **after** a random backbone has been trained
2. How these signals should be propagated during training so that the backbone itself gradually develops an l2r-favoring order bias

So the most accurate current statement is:

> We have largely established signal validity,  
> but training-time propagation still needs to be validated.

---

## 15. Redefining the role of the order head: from utility ranker to preference module

These results also motivate an important shift in how the `order head` should be interpreted:

> The order head should no longer be thought of as a module that directly reconstructs the final global order;  
> it is more naturally viewed as an **order preference module**.

Why?

- directly regressing noisy residual block utilities is easily biased toward late easy blocks
- directly outputting a near-l2r global order is too difficult a task
- but learning whether candidate order A is better than candidate order B is much more aligned with the current evidence

Accordingly, the training code is now moving toward the following procedure:

1. sample multiple candidate orders for the same example
2. compare them using an early-shaping quality
3. select a `preferred order` and an `other order`
4. train the order head so that the preferred prefix receives higher probability than the other one

This means:

- the order head is no longer a utility regressor
- it becomes a **pairwise order preference learner**
- its role is closer to:
  - an order sampler
  - a proposal-distribution learner
  - a curriculum controller

In one sentence:

> We do not want to remove the order head;  
> we want to turn it from a “final order reconstructor” into a “preference module that gradually shapes the order distribution during training.”

---

## 16. Current overall conclusion

At this point, the project has converged to a much clearer picture:

> We do not need to use l2r as a direct teacher label.  
> Once we extract trajectory properties such as “lower earlier, smoother earlier, and less delayed improvement” from the AR / Random contrast, those properties themselves are already sufficient to strongly favor l2r in large-sample averages.  
> Therefore, the key question of the next stage is no longer “whether such a signal exists,” but rather “how to propagate this signal during training, in a preference-style or curriculum-style way, so that the model gradually forms an order policy biased toward l2r.”

---
