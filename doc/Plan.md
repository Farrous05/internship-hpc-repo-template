# Implementation study — normal (offline) vs online (DAgger) fine-tuning

**Scope:** 4-week project. **Start with offline SFT to confirm the model can
learn `<end>` + topic switch at all; only then move to DAgger.** Online DAgger is
the target method; normal offline fine-tuning is the baseline we compare against.
**Hardware (DAIS, real specs):** single H200 (141 GB) via `gpu1`; **24h max
wall-clock**; **no internet on compute nodes**; apptainer containers.

---

## Shared foundation (both methods)

- **Model:** gemma-4-31b, bf16 ≈ 62 GB → headroom for vLLM + LoRA.
- **`<end>` token:** add to vocab; train embedding + unembedding rows.
- **Generation:** vLLM. Self-conversations of **8–15 turns** (collapse typically
  onsets ~turn 10 based on babel-ai records).
- **Detector extraction:** we cannot run on top of babel-ai. In week 1 we
  **extract** the collapse detector, recovery criteria, and metric computation
  into standalone modules on the cluster, and **swap the OpenAI embedder for
  local SBERT** (compute nodes are offline). This is real porting work, not
  copy-paste.
- **Negatives source:** our setup rarely produces non-collapse, so healthy
  (`<end>`-negative) contexts are scarce. We **mix in non-collapse trajectories
  from outside the collapse setup** (clean dialogue turns). Required so the model
  doesn't learn to always fire `<end>`.
- **Topics source:** ready in the repo — each `<end>` is followed by a topic
  pulled from that bank.

### Metrics (both methods)
- **Recovery rate** (primary): does it escape the loop and hold — babel-ai
  recovery criteria. This recovery rate measures collapse also.
- **Capability retention** (primary): no degradation on normal text.
- `P(<end> | collapsed)` (secondary diagnostic): sanity check that the token
  fires in the right place. Not the headline claim.

---

## Method A — Normal offline fine-tuning (baseline)

### Pipeline
1. From the collapsed runs we currently have in /results in babel-ai: at each
   onset insert `<end>` + a topic-switch continuation.
2. Add **negative examples**: healthy contexts where `<end>` must stay low
   (prevents the token firing everywhere).
3. Format: `(collapsing ctx → <end> → new topic)` + `(healthy ctx → normal token)`.
4. LoRA SFT (TRL `SFTTrainer`), 1–3 epochs.
5. Eval on held-out collapsed conversations.

### Data
Fixed, built once. Sourced from existing runs — *other models'* collapse.
Teaches the token mechanics, not gemma's own collapse signature.

### Training time
LoRA SFT on 31B, a few thousand short examples → **~2–6 h, one job**, far inside
the 24h cap.

### Strength / weakness
Simple, reproducible, guaranteed to finish. Static and off-policy: cannot show
the model recovering from *its own* loops.

---

## Method B — Online fine-tuning (DAgger)

### How DAgger works here (one round)
1. **Generate.** gemma produces a batch of ~100 self-conversations, 8–15 turns.
   vLLM batches all conversations *at each turn*, so it's ~12 batched calls, not
   1200 — roughly **~1 min of generation per round**.
2. **Label (auto).** The extracted detector embeds turns (SBERT), marks collapse
   onsets as `<end>`-positive and healthy turns as `<end>`-negative. **Seconds.**
3. **Filter (auto).** Keep only confident labels: margin check (score clearly
   past the 0.40 cutoff, not on it), persistence check (K consecutive rounds,
   not a blip), recovery-consistency check (drop if detector and recovery
   criteria disagree). No per-round human review — the detector is validated
   **once** up front against ~50 hand-checked labels, then trusted.
4. **Train.** LoRA SFT on this round's filtered labels. **~20–40 min.**
5. **Repeat.** The improved model generates the next batch — its collapses shift,
   so new labels come from states the *current* model actually reaches. That
   feedback is the whole point of DAgger. **~4–6 rounds** to see movement.

Warm-started from the Method A model so early rollouts aren't garbage.

### Why DAgger over GRPO
We know the **correct action** (`<end>` here, from the detector), so supervised
on-policy is simpler and more stable: same SFT code, no reward shaping, no PPO,
no credit-assignment problem. GRPO only wins when you have a *score* but not a
correct token — not our case. **GRPO = fallback** if labeling underperforms.

### Do we have the time? (honest)
**Compute is not the bottleneck.** ~1 h/round × ~5 rounds ≈ **5–8 GPU-hours
total** — trivial, fits comfortably in chained 24h jobs.
**Setup is the bottleneck.** The risk is entirely in week 1–2: container,
extracting the detector, the SBERT swap, and plumbing the generation loop. If
setup lands by end of week 2, the DAgger rounds in week 3 are comfortable.
**Verdict: feasible in 4 weeks — but only if setup doesn't slip.** Method A stays
the fallback if it does.

### Strength / weakness
On-policy: learns to recognize its *own* collapse — the real contribution.
Costlier than A, but far simpler than GRPO. Main risks: setup slippage, not
training compute.

---

## Side-by-side

| | Normal (A) | Online DAgger (B) |
|---|---|---|
| Data | fixed, off-policy | model's own rollouts, on-policy |
| Update | cross-entropy (SFT) | cross-entropy (SFT) on fresh labels |
| Signal | correct token (label) | correct token (auto-label + confidence filter) |
| New code | dataset builder | generation loop + auto-labeler |
| Compute | ~2–6 h, one job | ~5–8 GPU-h, chained jobs |
| Main risk | fake collapse ≠ own collapse | setup slippage |
| Claim strength | teaches the token | learns its own collapse (stronger) |

---

## 4-week timeline

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Cluster + apptainer + model download + **extract detector/recovery/metrics from babel-ai** + **swap OpenAI→SBERT** + test balloon | Working pipeline proven |
| 2 | Build Method A dataset (incl. mixed-in negatives + repo topics) + run LoRA SFT baseline + build generation loop + auto-labeler + filters | Baseline done; loop works |
| 3 | Real DAgger rounds (~4–6, chained jobs); tune | Trained online model |
| 4 | Eval both (A vs B): recovery, retention, `P(<end>\|collapsed)`; write-up | Results + comparison |

**Buffer:** 3 days–1 week of slack budgeted across weeks 1–2 for setup friction
(new stack, breakage). Realistic expectation: DAgger reached by **end of week 3**,
not sooner. If it slips further, Method A alone is a complete result.

---

## Must-do setup (both)
1. apptainer container from MPCDF `ai_containers` — don't hand-build.
2. Extract detector/recovery/metric modules from babel-ai onto the cluster.
3. Replace OpenAI embedder with local SBERT.
4. Test balloon before any real run (container → model → LoRA → vLLM → eval).
5. Debug on `gpudev` (15 min); real runs on `gpu1` (single H200).

---

## Open decision (affects labeling)
What does `<end>` physically do at inference — clear context, or inject a
new-topic prompt? Pin this before building B.