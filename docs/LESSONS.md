# Lessons

What a month of debugging this taught, written down because the technical
findings in [FINDINGS.md](FINDINGS.md) are specific to one dataset and these are
not.

---

## Audit what you import to the same standard as what you write

The defect that cost the most — a supervision target wrong on exactly the frames
that decide closed-loop behaviour — was not in code written here. Our own label
builder derives the target from realized poses, refuses each route's tail and
thins standstills, and measures clean at 0.0000 m.

We audited that pipeline carefully and then adopted an external dataset without
applying the same checks to it. Inputs get audited; targets are assumed correct
because they arrive labelled "ground truth".

**The check is cheap.** Compare the stored target against what the system
actually did next. It took twenty minutes and could have run in week one.

---

## A metric scored against a label cannot see a wrong label

Open-loop waypoint error measures distance to the *target*, not to correct
behaviour. If the target is wrong on some frames, the metric does not merely fail
to reveal it — it **rewards** the policy that reproduces the error more
faithfully.

This is why the model looked healthy at 0.056 m while scoring DS 11.56, and it is
why closed-loop evaluation is not simply a more expensive confirmation of
open-loop. Against this class of error it is the only valid measurement, because
its score comes from behaviour in an environment rather than from agreement with
a file.

---

## Measure the mechanism before spending on a treatment

Three interventions were run before the cause was known: creeping, aggressive
creeping, and a full 25-epoch run with speed dropout. All three treated symptoms.
The measurement that found the actual cause took twenty minutes.

When the feedback loop is eleven hours long, guessing is very expensive and
measuring is very cheap **relative to it**. The ratio should drive the order of
work, and it did not.

---

## Verify conventions against an independent signal, never by reading code

Every coordinate and packing convention that was *read* turned out wrong or
ambiguous. Every one that was *measured against something outside the annotation*
was settled:

- BEV crop geometry — against LiDAR ground and tall returns; the correct
  arrangement scored 0.434 where the runner-up scored 0.361
- the ego frame's axis meaning — by grouping the fourth waypoint by navigation
  command and seeing which component moved
- bit-plane semantics — by measuring each layer's occupancy against the
  documented meaning before relying on it

And the counter-example that proves the rule: three attempts to settle the
`label_raw` box convention statistically all failed to separate the candidates,
because the vehicles with enough returns to measure sat straight ahead where the
distinguishing sign flip has no effect. That test was **abandoned rather than
resolved by argument**, and a path with already-verified geometry was used.

When a measurement cannot separate the hypotheses, that is information. It is not
permission to pick the most plausible one.

---

## Infrastructure failures disguise themselves as results

Three separate times, something that was not a driving result was recorded as
one:

- a missing agent config — the evaluator caught the exception, wrote "Registering
  the route statistics", and produced 36 routes at DS 0.00
- a simulator sensor timeout on a shared GPU — recorded as a finished route with
  DS 0.00 and no infractions, indistinguishable from a policy that never moved
- a validation metric that scored each frame against another frame's future,
  reporting a constant 159.56 m for eleven hours while the training curve fell
  normally

None of these looks like an error. Each looks like a number.

**Guards that now exist, and the shape they share:** refuse to start when a
precondition is unverifiable (no agent config, model will not build), and refuse
to accept a result whose *shape* is impossible (every route at zero, a validation
error two orders of magnitude off). Both are cheap. Both would have saved a day.

---

## Know the noise floor before designing the experiment

Route-to-route DS spread over one full run is 6.25 against a mean of 4.10. Twelve
routes estimate the mean to about ±2.4 at 90%; six routes to ±4.5.

Pairing does not rescue it. The route-level DS correlation between two different
runs is **−0.09** — a route's score is decided by where the policy happens to
fail, not by the route's difficulty — so scoring the same routes for both arms
buys nothing. RC does correlate at +0.42, but DS is what is reported.

The consequence is liberating rather than limiting: screen for effects above
roughly 3 DS and ignore smaller ones, because a smaller effect was never going to
be demonstrable within this budget anyway.

---

## Report the confound that favours you

Four confounds separate this work from the published numbers it sits beside.
Three hurt it — simulator version, an unseen town covering 17% of the benchmark,
baselines quoted rather than re-run. One **helps** it: the supervision target was
rebuilt here and not for the references.

That fourth one is written into the limitations with its direction stated
explicitly. A reader who finds an advantage the author did not disclose stops
trusting the three that were disclosed.

---

## Withdraw claims rather than soften them

The reliability gate did not do what it was designed to do. The response was to
report it as a negative result and remove the interpretability claim, not to
restate it more carefully.

Two things came out of that which a softened version would not have produced: a
mechanistic explanation (the gate reads pooling statistics diluted by token
count, not sensor quality), and an ablation row that had been an annoyance —
`no_gate` scoring identically — turning into confirmation.

A thesis that can explain why a component failed is stronger than one that
reports a number it cannot account for.
