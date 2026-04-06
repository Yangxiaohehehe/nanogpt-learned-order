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