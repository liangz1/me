---
title: "Where instruction tuning suppresses demonstration copying: a layer-transplant localization"
date: 2026-07-01
categories:
  - interpretability
tags:
  - mechanistic-interpretability
  - instruction-tuning
  - in-context-learning
  - layer-transplant
toc: true
toc_label: "Contents"
toc_sticky: true
---

Base language models, given a few-shot prompt, often copy the surface form of the demonstrations instead of reasoning about the query, and instruction tuning largely removes this. This note asks two questions about that copying. Does the base model fail because it never computes the query-appropriate answer, or because it computes it but then lets the copied word win? And where in the network does instruction tuning intervene to change the outcome?

The answers point in a consistent direction. The base model does compute the query-appropriate answer: that signal (the competitor) stays present in the residual stream throughout, and the copied word only overtakes it in the upper layers. So the copying is executed late. But a layer transplant locates the decision much earlier: replacing just the first two layers is enough to flip whether the copying happens at all. The change is concentrated in the earliest layers, while the copying itself plays out near the end. These are two separate findings, a late execution and an early decision; how the early state leads to the late outcome is something this work locates but does not yet explain.

Why this might interest readers working on interpretability: copying versus reasoning is a familiar topic, usually approached behaviorally, by varying the prompt and reading the output. This post takes a mechanistic angle instead. A cross-model layer transplant lets me ask not just whether instruction tuning removes the copying, but where inside the network the change is located, and whether it looks like a new mechanism or a switch flipped on an existing one. Treating the base and instruction-tuned models as two halves to be recombined turns a behavioral difference into a question about specific layers, which is a small step toward saying what instruction tuning actually does to a model internally.

> **Note on AI assistance.** The experiments, analysis, and conclusions in this post are my own. I used an AI assistant as a discussion partner while working through the results, to help check the related-work citations, and to edit the prose for clarity. The claims, the interpretation of the data, and the decisions about what to include or cut are mine, including the negative steering result and the points where I argue against my own earlier interpretations.

## Summary of findings

**Behavioral phenomenon.** On a belief-attribution task, base models copy words out of the few-shot demonstrations into their reasoning (content overlay) on a large fraction of items; instruction-tuned models do this rarely or not at all. The contrast holds across model size.

**Localization.** A cross-model layer transplant flips the behavior from the earliest layers. Transplanting the first two instruction-tuned layers into base suppresses the overlay; the reverse takes the first nine base layers. A fifty-item check moves the group overlay rate in the same direction as the single-item result.

**Mechanism clue.** Two views of the model's internals, the logit lens and the attention pattern, both place the actual copying in the upper layers, not the early ones the transplant points to. An attempt to turn the early-layer difference into a steering vector failed a control and is reported as a negative result.

## Task definition

The task is belief attribution. Each item states what someone believed and what was in fact true, and asks whether the belief was correct. The model is prompted with a few-shot template that reasons step by step and puts the yes/no answer last. Here is one demonstration from the prompt:

```
Context: She thought the milk was fresh. The milk had gone sour.
Question: Was her belief correct?
Thinking: First, I need to understand her belief. She believed the milk was still fresh. ...
Answer: No.
```

The query that follows the demonstrations gives only a context and a question:

```
Context: He assumed the kitchen was clean. The kitchen was spotless.
Question: Was his assumption correct?
```

Here the belief (`clean`) agrees with the reality (`spotless`), so a correct response restates both, compares them, and answers yes:

```
Thinking: First, I need to understand his assumption. He assumed the kitchen was clean. ...
Answer: Yes.
```

This is what the model should produce. What the base model actually produces is different, and the difference is the subject of this note.

The dataset has 50 items in total, 25 where the belief matches reality (answer yes) and 25 where it does not (answer no). The example shown here, and the one used in the single-item experiments later, is item 46 from this set.

### Why this task separates copying from reasoning

This task separates the two because part of the answer cannot be copied. When the model restates the belief and reality, it must produce query-specific words like `clean` and `spotless` that no demonstration contains. So if a demonstration word shows up there, the model is copying, and you can see it directly in the raw output. This property defines an already-studied class of tasks, abstractive in-context learning, where the answer content words do not appear in the demonstrations. Belief attribution is just one task in this class, not a special one.

A multiple-choice task like MMLU cannot separate the two. Its answer space is a set of shared labels (A, B, C, D) that the demonstrations already display, so copying and reasoning both produce an answer from that same demonstrated answer space, and the output alone does not reveal which one happened.

Strictly speaking, this task does not prove deep reasoning. What it gives me is a clean behavioral separation between copying demonstration content and producing query-grounded content.

## Overlay: copying made visible

Here is what the base model actually produces on the kitchen item, instead of the correct response above:

```
... Next, I need to compare his belief with the reality. The belief of the milk being fresh did not match the reality that the milk was sour. So my answer should be no.
```

The comparison sentence is about `milk`, not the `kitchen`. The word `milk` could not have come from this query; it was lifted from a demonstration. I call this content overlay, and the word the model should have produced instead (here, `kitchen`) the competitor. Overlay is the copy made visible, and the rest of this note tracks where it comes from.

Formal definition: I count an output as overlay if a content word from the demonstrations appears in a slot where the query's own context supplies a different content word and does not license the demonstrated word.

### Why a highly scaffolded template?

The prompt uses a fixed, heavily structured reasoning template that walks through four steps in the same words every time (understand the belief, check the fact, compare, then answer), and crucially puts the yes/no answer last:

```
First, I need to understand the belief. ...
Then, I need to check the fact. ...
Next, I need to compare belief with reality. ... did / did not match.
So my answer should be yes / no.
```

Two reasons for this choice. First, putting the verdict at the end makes the intermediate reasoning an uncontaminated probe: we can watch what the model does with belief, reality, and the comparison before it commits to an answer, and overlay surfaces in that reasoning region. Second, the rigid scaffold gives the model a fixed slot structure to fill, which makes the copying versus reasoning distinction sharp: in the comparison slot, a reasoning model writes about the query, a copying model writes about a demonstration. A looser prompt would blur where the overlay sits.

This scaffold may amplify slot-filling/copying, so I do not claim the exact overlay rates are prompt-invariant. I use it because it makes the phenomenon measurable; a later supplementary check suggests the late-rise pattern is not unique to this scaffold.

## Preliminary experiment: copying is a stable base-model behavior

Before localizing the copying, it is worth checking how widespread it is. On a fixed four-shot prompt (two demonstrations answered yes, then two answered no), the overlay rate over the 50-item set is:

| model | overlay rate |
| --- | --- |
| 2b base | 46% |
| 2b instruction-tuned | 2% |
| 9b base | 58% |
| 9b instruction-tuned | 0% |

These rates use greedy decoding (temperature 0), so each prompt gives a single deterministic answer block. To detect overlay I first flagged answers containing any demonstration content word, using a fixed keyword list of those words, then read through the outputs by hand under a fixed criterion: a demonstration content word that the item's own context does not license. I did this manual pass for all four model conditions, over every item rather than only the flagged ones, so it corrects both false positives from the keyword match and cases it missed. The reported rates are after this check.

Base models copy on a large fraction of items, while instruction tuning almost eliminates it. The contrast holds across model size, and if anything the larger pair makes it sharper: 9b base copies more than 2b base, so in this setup, scaling from 2b to 9b does not fix copying, while 9b instruction-tuned is the cleanest of the four.

The copying is also not an artifact of one demonstration ordering. Reordering the examples changes which words get copied, but the overlay rate stays similar. (I have run a fuller order ablation on this; the write-up is still pending.)

So copying is the stable behavior of the base models, robust to both scale and demonstration order. That is what makes it worth asking where in the network it comes from. The mechanistic experiments below use the 2b pair, the smaller and simpler case; the 9b pair serves mainly to show the contrast is not specific to one model size.

## The transplant experiment

The question for this note is: where is overlay decided? The base model emits `milk` (copy) instead of `kitchen` (the query-appropriate word); the instruction-tuned model emits `kitchen`. Somewhere between the two models, something changes. To locate it we run a cross-model layer transplant.

The setup is a single forward pass on the prompt, read out at the position where the model is about to emit the overlay word (right after "The belief of the"). We measure two numbers: the softcapped logit of `milk` (the overlay word) and of `kitchen` (the competitor), as recovered by logit lens from the residual stream. Gemma applies logit softcapping, so I compare the softcapped logits rather than raw unbounded logits. In the pure base model, `milk` wins (the overlay); in the pure instruction-tuned model, `kitchen` wins.

A transplant at layer k means: run the first k layers using one model (the source), then hand the residual stream to the other model (the target) and run the remaining layers with it. Gemma-2-2B has 26 transformer blocks. Sweeping k from 0 (entirely the target model) to 26 (entirely the source model) traces a path between the two models. We do this in both directions: transplanting the early instruction-tuned layers into base (IT to base), and the early base layers into the instruction-tuned model (base to IT). A control where a model is patched with its own activations reproduces the unpatched output exactly, confirming the plumbing is correct.

### Result: a flip, and it is early and asymmetric

![Flip curves: final-layer softcapped logit of milk vs kitchen against transplant point k, for both transplant directions](/assets/images/transplant-fig1-flip-curves.png)

*Figure 1: flip curves. x axis = transplant point k; y axis = final-layer softcapped logit of `milk` vs `kitchen`; left panel IT to base, right panel base to IT.*

Reading the left panel (IT to base): with zero or one instruction-tuned layer in front, base still emits the overlay (`milk` wins). But as soon as the first two layers are instruction-tuned, the output flips: `kitchen` overtakes `milk`, and stays ahead for every later transplant point. Transplanting just the first two instruction-tuned layers into base is enough to suppress the overlay on this item.

Reading the right panel (base to IT): the mirror image, but not symmetric. Replacing the first six instruction-tuned layers with base layers barely moves anything; the instruction-tuned model still suppresses the overlay. Only when the first nine layers are base does the overlay return. The flip is at k = 9 here, against k = 2 in the other direction.

So the location is early, in the first handful of layers, and the two directions are asymmetric: entering the overlay-suppressed state takes only about two early instruction-tuned layers, while leaving it takes about nine. In this transplant setup, the suppression is easy to switch on and hard to switch off.

### A switch, not a dial

I do not mean a literal discrete mechanism has been identified. The claim is descriptive: under this sweep, the trajectories cluster into two families rather than interpolating smoothly.

The per-transplant-point trajectories make the shape of the transition clearer.

![Per-transplant-point residual trajectories over layers 16 to 25, milk vs kitchen, for each transplant point k in both directions](/assets/images/transplant-fig2-trajectory-panels.png)

*Figure 2: residual trajectories over layers 16 to 25 for each transplant point k, `milk` vs `kitchen`, with the pure-target trajectory drawn faint for reference. Left column IT to base, right column base to IT.*

Two things stand out. First, the trajectories fall into two families, one looking like pure base and one like pure instruction-tuned, rather than spreading evenly between them. On the left (IT to base), k = 0 and k = 1 sit in the base family and k = 3 onward in the instruction-tuned family; on the right (base to IT), k = 0 through k = 7 are instruction-tuned and k = 9 onward are base. Second, each direction has exactly one transitional trajectory at the boundary, k = 2 on the left and k = 8 on the right, that does not sit cleanly in either family. But even these lean toward the family they are about to join, and the lean matches their final output: the k = 2 curve already tilts toward instruction-tuned, which is the side that wins its output, and k = 8 toward base. So the transition is not perfectly sharp, but there is only a single in-between point on each side, and it is already committed to one side. That is the shape of a switch between two states, not a dial where each added layer would shift the curves a little further.

### Does this hold across items?

The single-item result is a localization on one example. To check it holds beyond this one item, I ran the two flip-point transplants (k = 2 for IT to base, k = 9 for base to IT) on all fifty items and measured overlay as a group rate rather than a single-token logit. For each item I generated the first answer block and checked whether any strong demonstration content word (milk, fresh, sour, package, doorstep, keys, counter, road, open, closed, passable, construction) appeared where the item's own context did not license it.

| condition | overlay items | rate |
| --- | --- | --- |
| base | 23/50 | 46% |
| instruction-tuned | 1/50 | 2% |
| transplant k=2 (IT into base) | 9/50 | 18% |
| transplant k=9 (base into IT) | 34/50 | 68% |

The ordering matches the single-item picture. Transplanting the first two instruction-tuned layers into base drops the overlay rate from 46% to 18%, much closer to the instruction-tuned level than to base, though not all the way to the instruction-tuned 2%. Transplanting the first nine base layers into instruction-tuned raises it to 68%. The group statistic flips in the same direction as the single token, so the localization is not an artifact of one item.

Two caveats. The k=2 residual (18%) is not zero, but that is the boundary behavior expected at a flip point: k=2 is the minimal transplant that just crosses over, and the leftover cases are single-word leaks (passable) and a few full milk-fresh-sour copies. The k=9 rate (68%) is higher than pure base (46%), and on several items where base itself does not overlay, the base-into-instruction-tuned transplant does. So this direction is not a clean reversion to base behavior: transplanting base's early layers into instruction-tuned produces a hybrid that copies more than either pure model, likely because the later instruction-tuned layers receive an input distribution they were not trained on. The IT-to-base direction, where overlay is suppressed, is the cleaner half of the evidence.

## Decided early, carried out late: attention and logit evidence

The transplant locates where the copying is decided: the first couple of layers. But those early layers are not where the copying is carried out. Two views of the model's internals, the logit lens and the attention pattern, both place the actual copying much later, around layers 16 to 22. The early layers set a state; the later layers act on it.

Start with the logit lens. Reading the overlay and competitor logits layer by layer, across all of these transplants on this item, the competitor word keeps a substantial logit at every layer; it is never strongly suppressed on its own. What moves is the overlay word. Through the middle layers the competitor leads, and only near the end, not before about layer 22, does the overlay word rise to overtake it, and only in the runs that end up copying. In the runs that suppress the overlay, the competitor simply keeps its lead to the end.

<details>
<summary><strong>Supplementary: the same pattern holds under a different prompt template (7 items)</strong></summary>

<p>The same competitor-stays-high, overlay-rises-late pattern shows up under a different prompt template. An earlier exploratory run used the same belief items but a five-shot, less heavily scaffolded template, and tracked the overlay and competitor logits layer by layer for pure base and pure instruction-tuned (no transplant).</p>

<img src="/assets/images/transplant-fig3-seven-item-supplement.png" alt="Overlay vs competitor trajectories for seven items, base vs instruction-tuned, under a different prompt template">

<p><em>Figure 3: overlay vs competitor trajectories for seven items, base vs instruction-tuned.</em></p>

<p>Across these seven items the competitor keeps a high logit in both models, and in base the overlay word rises to meet or overtake it only in the last few layers, while in the instruction-tuned model the overlay stays low throughout, the same shape seen on the main item. Because these are the same items under a different template, the pattern does not seem tied to the particular scaffolded prompt used in the rest of this post. This run is not an apples-to-apples comparison (five-shot, less scaffolded, and a direct base-versus-instruction-tuned readout rather than a transplant), so I treat it as a side observation rather than part of the main result. The competitor words here (charged, already, submitted) are also cleaner than kitchen, since they never appear in their item's context.</p>

</details>

So the logit lens places the copying late: the overlay word only wins in the final layers, on top of a competitor signal that was there all along. The attention pattern tells the same story from the other side.

Now the attention. At the position where the model is about to emit the overlay word, I measured how much each head attends to the demonstration's `milk` tokens (the copy source) and to the query's `kitchen` tokens (the semantic target), for base and for instruction-tuned, layer by layer.

![Attention mass at the overlay position, summed over heads, across layers, for base and instruction-tuned to demo-milk and query-kitchen](/assets/images/transplant-fig4-attention-mass.png)

*Figure 4: attention mass at the overlay position (summed over heads).*

What matters is not when base attends to `milk`, but when the two models attend differently. Through the early and middle layers they are not cleanly separated: both barely attend to `milk` in layers 0 to 5, both spike on it at layer 6, and at layer 14 the instruction-tuned model still attends to `milk` more than to the query's `kitchen`. No stable gap yet.

The separation begins at layer 16, and the change is on the instruction-tuned side. From layer 16 through 23, instruction-tuned attention to `milk` collapses and stays low while its attention to `kitchen` is high; base attends to both and drops neither. So the difference is not that base seeks `milk` and instruction-tuned seeks `kitchen`. Base attends indiscriminately to the copy source and the query alike; instruction tuning has taught the model to drop the demonstration and attend only to the query. The copying is what base fails to suppress, not something it actively seeks.

This band lines up with the logit lens: the overlay word overtakes the competitor around layer 22, inside the same window. Both readings put the copying in the upper layers, while the transplant puts the decision in the first two. The early layers do not copy; they set a state, and fifteen layers later the attention either still includes the demonstration's `milk` (base) or has dropped it for the query (instruction-tuned).

## A direction, but not a steerable one

Back to the early layers. The transplant says they write a state that decides the outcome; is that state a single direction one could add to base to switch off copying? I tried this, and the answer is a clean negative that I think is worth reporting.

I took, for twenty-seven items where base produces overlay under a looser per-item criterion (the strong-overlay rate reported earlier counts 23), the difference between the instruction-tuned and base residual at the layer-2 output, at the overlay position. These difference vectors are highly consistent in direction (mean pairwise cosine 0.74, and on average 87% of each item's difference lies along a single shared direction). Adding this shared direction back into base's layer-2 residual, at a suitable magnitude, did remove the overlay on the test item: base stopped writing `milk` and wrote `kitchen` instead.

But a control undercuts the obvious interpretation. Adding random directions of the same magnitude removed the overlay too, on three of five random directions, even though those directions are nearly orthogonal to the shared one. The magnitude needed to flip the overlay is in a range where many different perturbations flip it, so this test cannot isolate the shared direction as causally special, even if it does not tell us what those perturbations are doing.

One detail I found surprising, and would like others' thoughts on: the random perturbations did not produce nonsense text. They produced either overlay or clean restatement, two recognizable states, not broken text. It is as if the medium-magnitude perturbation moves the model between two attractors rather than off the data manifold entirely. I do not have a hypothesis for why, and I am flagging it as an open question rather than something I can resolve here.

A separate caveat on the consistency number: most of the twenty-seven items overlay with the same words (milk, fresh, sour), so the 87% shared-direction figure is measured on a set whose copied content is fairly homogeneous, and may partly reflect that homogeneity rather than a fully general anti-copy direction.

## Alternative explanations I considered

Two objections to the localization are worth addressing directly, since both are the first things a careful reader will raise.

Is the transplant flip just an artifact of mismatched scales between the two models? When you run the first layers of one model and the rest of another, there is no guarantee the two operate at the same numerical scale; if base activations live at one magnitude and instruction-tuned at another, the combined run could land in a range neither model was built for, and the flip might reflect that mismatch rather than anything meaningful. As a check, I tracked the norm (the length) of the final-layer residual at the readout position for every transplant point.

![Residual norm at the readout position against transplant point k, for both directions, with the two flip points marked inside the stable region](/assets/images/transplant-fig5-norm-sweep.png)

*Figure 5: norm sweep.*

The norm is flat across most of the sweep and only moves near the pure-source end, where almost all layers come from one model. The two directions settle at different stable levels (around 190 for IT-to-base, around 75 for base-to-IT), which is expected since they end in different models. What matters is that both flip points fall inside the flat region, not where the norm moves: the IT-to-base flip at k = 2 has norm 183, and the base-to-IT flip at k = 9 has norm 77, each in line with its neighbors. The flip happens where the scale is normal for that direction; the large norm changes are elsewhere. The mismatch between the two models is real, but it is not what produces the flip.

A second objection: if the first two layers barely differ in their attention between base and instruction-tuned, how can those layers be where the outcome is decided? Where is the difference, if not in attention?

The difference is there in the residual stream, just not in the attention pattern. The transplant swaps the entire computation of those layers: the attention pattern, the attention output projection, and the MLP. The pattern is largely alike between the two models, but the other two are not, so heads that attend to the same positions still write different vectors into the residual stream. And they do: measured directly, the full residual difference between base and instruction-tuned at the overlay position is already over a fifth of the residual norm by the end of the first two layers.

## Limitations and open questions

- Single item for the fine-grained results. The exact flip locations (k = 2 versus k = 9), the switch-like trajectory shape, and the attention story are all from one item (kitchen-clean, overlay = `milk`, competitor = `kitchen`). The fifty-item run checks only the group-level direction of the transplant effect, not the fine-grained picture. Reproducing the detailed results across items is the obvious next step.

- The competitor word in the single-item study, `kitchen`, has one weakness as a test case: besides appearing in its own context, it also shows up in one of the demonstrations. So base producing `kitchen` could partly be demonstration copying rather than purely semantic computation, which blurs the overlay-versus-competitor contrast. The supplementary seven-item check is cleaner on this point: several of its competitors (charged, already, submitted) appear only in their own contexts and not in any demonstration, yet show the same competitor-high, overlay-late pattern.

- The y axis throughout is a logit-lens estimate. The model only produces logits at the final layer, by passing the residual stream through the final norm and the unembedding. The logit lens applies that same final-layer mapping to the residual stream at earlier layers, as if the model stopped there, and softcaps the result. This gives a reading on the same scale as the output, but the model does not actually compute logits at those intermediate layers; the layer-by-layer numbers are a projection we impose, not a quantity the model represents internally.

- All mechanistic experiments (transplant, attention, steering) are on the 2b base and instruction-tuned model pair. The 9b results establish only that the overlay phenomenon itself is present, and in fact sharper, at larger scale; they do not show that the same early-layer localization holds there. Whether the mechanism is the same across model size is untested.

- The base-to-IT direction shows a hybrid effect, copying more than either pure model on some items.

- The steering direction is real and consistent across items but not shown to be causally special, since same-magnitude random directions remove the overlay too. And the early-layer state that the transplant points to as decisive has no token-level semantics under the logit lens, so what it actually encodes remains unresolved.

## Related work

The questions and methods here connect closely to several existing lines of work. That overlap is worth stating up front: the problem of copying versus reasoning in in-context learning is one many people have studied, the methods (cross-model patching, the logit lens, attention analysis) are established tools, and the patterns found here, an early-to-late split and copying executed by late attention heads, converge with what others have reported. Against that shared background, this work also differs from each of these studies in ways that are non-trivial. The rest of this section places it next to the closest of them and is explicit about both the overlap and the differences.

### Prakash et al. (2024), "Fine-Tuning Enhances Existing Mechanisms"

What they did. They argue that fine-tuning enhances mechanisms already present in the base model rather than installing new ones. In their case study, a model fine-tuned on arithmetic uses the same entity-tracking circuit as its base, with fine-tuning only strengthening one sub-component. They show this with cross-model activation patching, a close relative of the layer transplant used here. ([arXiv](https://arxiv.org/abs/2402.14811))

Alignment. The method is closely related: they use cross-model activation patching, patching activations from one model into another, which is the same family as the layer transplant here. The findings align too. The competitor (the query-appropriate answer) is already represented in the base model, with a high logit throughout the residual stream; instruction tuning is what makes it win consistently rather than be overridden by the copy. This supports their framing from a different task.

Difference. In their case fine-tuning enhances the target mechanism by making a sub-component compute better. Here, instruction tuning improves the semantic outcome by suppressing a competing copy signal, not by strengthening the semantic signal itself, which is already strong in the base model.

### Halawi et al. (2024), "Overthinking the Truth"

What they did. They study how models process false few-shot demonstrations (for example, sentiment examples with their labels flipped). Decoding intermediate layers, they find that correct and incorrect demonstrations behave alike in early layers but diverge at a "critical layer", after which accuracy on the incorrect demonstrations falls. The name "overthinking" captures this: if the model stopped at an early layer it would be more accurate, but it keeps computing through later layers, gets pulled toward the false demonstrations, and ends up worse. They trace this to late-layer heads that attend back to and copy from the demonstrations. ([arXiv](https://arxiv.org/abs/2307.09476))

Alignment. Both the early-alike, late-diverging shape and the late copying heads parallel what the logit lens and attention show here. Their semantic-prior-versus-copied-label competition corresponds to the competitor-versus-overlay competition in this work: the model's own semantic answer leads early, and the copied demonstration content takes over late.

Difference. Instruction tuning suppresses the content overlay here, but Halawi et al. find it does not remove overthinking. I suspect the reason is that their prompts are ambiguous about the task. A demonstration like "Review: [negative text] / Answer: Positive" admits two readings: that the labels are wrong and should be corrected, or that the examples define a labeling convention to follow. Halawi's overthinking is the model taking the second reading. An instruction-tuned model has no instruction telling it which reading to take, so it does not reliably suppress the copying. The task here removes that ambiguity with an explicit instruction to reason about the actual content.

### Wei et al. (2023), "Larger language models do in-context learning differently"

What they did. They study how scale and instruction tuning shift the balance between a model's semantic priors and the input-label mappings given in context, using flipped-label and semantically-unrelated-label setups across several model families. Two findings are relevant here: larger models override their semantic priors more readily, following the in-context labels even when those contradict the priors; and instruction tuning strengthens both the use of semantic priors and the use of in-context mappings, but more the former. ([arXiv](https://arxiv.org/abs/2303.03846))

Alignment. Both findings line up with the four-way comparison in the preliminary experiment here. Larger models leaning harder on the in-context demonstrations matches 9b base copying more than 2b base. Instruction tuning strengthening the use of semantic priors matches the instruction-tuned models relying on the query's meaning rather than copying the demonstrations.

Difference. Their evidence is behavioral, measured as accuracy across model families; the localization here is mechanistic, by layer transplant on a single base and instruction-tuned pair. There is also a reversal in framing: what Wei et al. describe as a capability, larger models following in-context evidence over their priors, shows up here as the failure mode, larger base models copying the demonstrations more. The underlying tendency, leaning on the demonstrations, is the same; whether it helps or hurts depends on whether the demonstrations should be followed.

### Other related work

Beyond the closest work, the task and its phenomena rest on a few established ideas. The copying that produces the overlay is the behavior of induction heads, attention heads that match a pattern earlier in the context and copy its continuation ([Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)); that induction heads underlie this kind of copying is well established, so I take it as given. The setup itself, where the answer words are absent from the demonstrations, is the abstractive setting of in-context learning, the abstractive-versus-extractive distinction introduced by [Todd et al. (2024)](https://arxiv.org/abs/2310.15213). And the idea that suppressing copying can help abstractive tasks is the aim of Hapax ([Sahin et al., 2025](https://arxiv.org/abs/2511.05743)), which trains models to reduce inductive copying while preserving abstractive performance, building on Todd et al.'s task set.

## Next steps

The most direct next step is at the circuit level. The transplant localizes where the copying is decided, but not which heads carry it out. Existing tools fit this directly: copy-suppression and anti-induction heads have been characterized ([McDougall et al., 2023](https://arxiv.org/abs/2310.04625)), and the strength of induction versus anti-induction circuits, and how fine-tuning shifts it, has been measured ([Jobanputra et al., 2025](https://arxiv.org/abs/2505.21785)). Applying those measurements across the transplant would test whether instruction tuning suppresses the overlay by weakening the induction heads that copy, by strengthening the anti-induction or copy-suppression heads that resist it, or both.

A second step tests the nature of the suppression rather than its location. The idea is that instruction tuning suppresses copying when an explicit instruction calls for it, not in every case. Halawi et al.'s flipped-label prompts can be interpreted in two valid ways: that the example labels are wrong and should be corrected, or that the examples define a labeling convention to apply. We can verify explicitly how an instruction-tuned model responds to each interpretation by adding an instruction that selects it: one version asking for the true sentiment regardless of the example labels, the other saying the examples define a convention to apply. If instruction tuning suppresses copying under the first but not the second, while the base model ignores the instruction in both, that supports reading the suppression as instruction-following rather than an unconditional anti-copy effect.
