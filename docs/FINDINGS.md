# Findings

Every number here came from a measurement in this repository, and the tool that
produced it is named. Nothing is quoted from a paper and nothing is an estimate.

The order is the order the work happened in, because each finding is what made
the next one visible.

---

## 1. The harness reproduces the infraction score but not route completion

`tools/run_autopilot_val.sh` drives TransFuser's own privileged expert through
our stack, so the route definitions, scenario triggers, traffic, infraction
detectors and DS arithmetic are checked against an external reference rather than
against our own expectations.

| | ours | published |
|---|---|---|
| infraction score IS | 0.884 | 0.89 |
| route completion RC | 73.38 | 82.71 |
| driving score DS | 64.03 | 74.49 |

The infraction half — collision and violation detectors, penalty multipliers,
the score arithmetic — reproduces to 0.006. Route completion does not, and the
cause is the simulator version: the reference documents CARLA 0.9.10.1 and this
work runs 0.9.14, whose traffic manager was rewritten and deadlocks more often in
Longest6's dense traffic. Eighteen of thirty-six routes ended with "agent
blocked" or "agent timed out" for the *expert*.

**Consequence for reporting.** Every agent evaluated in this chain carries the
same handicap, so scores are reported next to this expert rather than next to
published figures. A perfect policy would still lose about 11% of RC here.

An earlier version of this check used the stock leaderboard evaluator and
produced RC 87.94, which looked like a pass and was not: the stock evaluator runs
lighter traffic and penalises stop-sign infractions, which Longest6 does not.
Longest6 must run through the `*_local.py` evaluator. Results produced the other
way are comparable with nothing and were discarded.

---

## 2. Four train/deploy convention mismatches, none visible in the loss

After switching to the published dataset, the agent built its inputs with
conventions that no longer matched it. Each was found by comparing the
distribution of what the agent builds against the same channel over the training
set (`tools/check_closed_loop_inputs.py`, `tools/compare_input_dist.py`), not by
reading code.

- **LiDAR offset sign** — the forward shift was applied as −1.3 m where the data
  wanted +1.3 m, a 2.6 m error in where the world sits relative to the car.
- **Camera rig** — three cameras at ∓60° stitched to 960×160 and centre-cropped
  to 704×160; the agent had been building a different strip.
- **Goal distance** — the look-ahead goal was 3–6× beyond its training range
  while the open-loop waypoint error stayed at 0.137 m.
- **BEV raster** — their top-down PNG is 500×500 at 5 px/m and ego-centred; ours
  is a 32 m × 32 m ego-bottom window. Resizing one onto the other, which is what
  the adapter did, trains the auxiliary head against a map bearing no geometric
  relation to the frame the network reasons in.

The BEV geometry was settled by measurement rather than by reading their
renderer, whose own comments call one shift "weird" and carry two FIXMEs: ground
returns must land on drivable area and tall returns must not. The correct
arrangement scores 0.434 over 398 frames, the plain resize 0.081, the runner-up
0.361 (`tools/check_bev_alignment.py`).

**None of these moved the open-loop error.** That is the point: a metric scored
against a stored label cannot see an input the deployment builds differently.

---

## 3. The supervision target is wrong on exactly the frames that decide closed-loop behaviour

This is the finding that mattered most, and it took a month to look for because
inputs get audited and targets are assumed correct.

The published dataset ships a waypoint block with each frame. Comparing it with
the pose the vehicle actually reached at the same future times, per route, at the
measured 0.5 s frame stride:

| frames | n | agreement | label 2 s travel | real 2 s travel |
|---|---|---|---|---|
| moving | 5936 | **0.080 m** | 8.03 m | 7.81 m |
| stopped | 213 | 0.589 m | 0.77 m | 2.80 m |
| stopped, then pulled away | 126 | — | **0.82 m** | **4.56 m** |

Across Town05, 14.1% of stopped frames are pull-aways and only 10% of those carry
a label showing any motion. The raw data shows it plainly: three consecutive
stationary frames hold byte-identical waypoints describing a slow creep, and the
next frame is at 3.12 m/s.

So the labels are right exactly where the car is already moving and wrong exactly
where it has to start. A network fitted to them reproduces them to 0.056 m
open-loop and learns *when stopped, creep* — which is the 0.30 m/s it predicts at
zero speed, the 89% brake rate, and the route timeout.

**Open loop cannot detect this**, because it is scored against the same wrong
target. A policy that reproduces the error more faithfully scores *better*.

### The repair

`data.tf_relabel` rebuilds the target from realized poses. All frames or none:
switching definition at a speed threshold would leave two meanings of "waypoint"
in one training set. Frames whose route ends before the horizon are refused
rather than padded, because padding by holding the last pose is what manufactured
a standstill target in the first place.

Verified in both directions (`tools/check_relabel.py`), since a fix that quietly
moved the moving frames would trade a known failure for an unmeasured one:

| | stored | relabelled |
|---|---|---|
| moving | 7.28 m | 7.06 m — unchanged, as it should be |
| pull-away | 0.70 m | **3.72 m** — the repair |
| stays stopped | 0.10 m | 0.05 m — still stops at red lights |

And on the trained policy, over 164 stopped frames where the expert pulls away:

| | expert | before | after |
|---|---|---|---|
| pull-away frames | 2.56 m/s | 0.34 m/s | **1.64 m/s** |
| stays-stopped frames | 0.03 m/s | 0.07 m/s | 0.12 m/s |

A 4.8× increase where it was needed, with the red-light behaviour intact.

### Our own collection does not have this defect

Checked on disk over 8438 labelled frames of 25 routes, not just in the code:
label agrees with realized pose to **0.0000 m**, pull-away frames carry 3.73 m of
travel, and only 15.8% of frames are stopped against their 30%. `build_labels.py`
derives the target from realized poses, refuses each route's tail, and thins the
interior of long standstills — so 35.8% of our stopped frames are pull-aways
against their 14.1%.

The defect was in what we **imported**. We audited our own pipeline and never
applied the same standard to the one we adopted.

---

## 4. The reliability gate does not do what it was designed to do

Eq. 4.9 is `z = g·z_camera + (1−g)·z_lidar`, checked rather than assumed, so `g`
is the camera weight and higher means more camera. If it tracked sensor quality,
degrading the camera would push it down. It does not
(`tools/check_gate_response.py`, three independent seeds):

| | s0 | s1 | s2 |
|---|---|---|---|
| clean | 0.811 | 0.794 | 0.846 |
| darkened | +0.002 | −0.006 | −0.003 |
| heavy noise | **+0.160** | **+0.195** | **+0.146** |
| camera removed entirely | **+0.149** | **+0.178** | **+0.089** |
| LiDAR removed entirely | +0.002 | +0.000 | +0.003 |

Take the camera away — the exact condition sensor dropout trains for — and the
network leans on it harder.

**The mechanism was already written down in `fusion.py` and never connected to
this.** The gate reads mean-pooled summaries of 440 camera tokens against 4096
LiDAR pillars, so LiDAR arrives ten times diluted by token count alone. Drop the
camera and its tokens become the absent token repeated, whose pooled summary is
cleaner and lower-variance than the diluted LiDAR one — so the gate reads
"confident" and rises. It responds to pooling statistics, not to sensors.

This also explains the ablation beside it: `no_gate` scores 0.136 against 0.135
for the full model, inside the 0.003 seed spread. A gate carrying no signal is
exactly what that row should look like.

Two repairs follow from the diagnosis, and both are now implemented and
switched on in `configs/egca.yaml` (untrained as of this writing): attention
pooling (`pooling: attention`), so neither branch's contribution depends on how
many tokens it emits, and direct gate supervision (`gate_supervision: 0.3`) —
sensor dropout already manufactures frames whose reliability is known by
construction, so on exactly those frames the gate is trained toward the
modality that still carries information, instead of hoping the meaning
emerges. The gate is also per-dimension now (`gate_form: vector`): a scalar
convex mix is strictly less expressive than the `concat` ablation's linear
merge, which is the other half of why `no_gate` tied.

---

## 5. Open-loop waypoint error does not separate fusion strategies

Best validation waypoint L1 after 20 epochs, all on the same collection and
budget:

| config | error | |
|---|---|---|
| egca s0 / s1 / s2 | 0.137 / 0.134 / 0.135 | seed spread **0.003** |
| concat | 0.137 | inside the spread |
| late | 0.135 | inside the spread |
| no_gate | 0.136 | inside the spread |
| no_dropout | 0.136 | inside the spread |
| camera_only | 0.138 | inside the spread |
| no_aux | 0.140 | barely outside |
| **full_attn** | **0.123** | four times the spread, and *better* |
| lidar_only | 0.203 | clearly worse |

Two conclusions, both written into the thesis:

1. Any claim that cross-attention beats concatenation or late fusion has to rest
   on closed-loop score, not on waypoint error.
2. The one place the metric *does* separate is the place equivalence had been
   claimed. Linear attention removes 92% of the fusion operator's
   multiply-accumulates (7.38 → 0.59 GMAC) at a measured cost in waypoint
   accuracy. The thesis now states a quantified trade-off, not equivalence.

---

## 6. What the policy does now, and why it fails

Closed-loop Longest6 in our chain. Routes whose status says the simulator crashed
are excluded — those are infrastructure failures recorded as DS 0.00 driving
results, and including them understated both runs.

| run | routes | DS | RC | IS |
|---|---|---|---|---|
| privileged expert | 36 | 64.03 | 73.38 | 0.884 |
| `tf_base_rl` (relabelled) | 34 | 4.10 | 29.25 | 0.250 |
| `concat` | 30 | 2.54 | 21.16 | 0.324 |

The last two are **not comparable** — different dataset, different epoch budget —
and the thesis says so.

The month-long standstill is gone. The car drives: two routes completed outright,
six above 50%, median RC 17.5. What it does instead is collide.

| infraction | total | per route |
|---|---|---|
| vehicle collision | 141 | **3.92** |
| layout collision | 37 | 1.03 |
| blocked | 26 | 0.72 |
| outside route lanes | 24 | 0.67 |

Route outcomes: 26 blocked, 6 timed out, 2 simulator crashes, **2 completed**.

Collisions are not only an infraction-score problem. The trace shows what
"blocked" is: the car touches something, wedges against it, and the controller
holds throttle at 0.749 with the brake off while the speed stays at 0.00 for the
180 s the evaluator waits. **Route completion dies from the same event that costs
the infraction**, which makes the collision rate the one lever that moves both
terms of DS.

### Why it collides: the auxiliary target had no agents

The BEV target carried road and lane and nothing else, while the vehicle and
pedestrian layers sat unread in the same PNG. Their encoder documents the
packing:

```
channels  0-4  -> plane 0   road, lane, lights
channels  5-9  -> plane 1   vehicle, pedestrian
channels 10-14 -> plane 2   future vehicles
```

cv2 writes plane 0 to blue and plane 1 to green, so vehicles are bit 7 of green.
Their own loader appears to keep planes 10–11 only because it reads the file as
RGB, where red and blue are swapped against ours — both arrive at road and lane.
Occupancy over 120 frames before relying on it: road 0.364, lane 0.045, lights
0.0001, vehicle 0.0038, pedestrian 0.0001, matching their 22×9 px vehicle
template at 5 px/m.

`bev_classes: 5` emits them now. **Implemented and committed; not yet trained or
evaluated.**

A first attempt rasterised the `label_raw` bounding boxes instead, and was
abandoned: three statistical tests could not separate the candidate coordinate
conventions (best 0.361 against 0.351) because the vehicles carrying enough LiDAR
returns to measure sit almost entirely straight ahead, where the lateral sign
flip that distinguishes the hypotheses changes nothing. The top-down raster has
no such problem — its geometry was already established against the LiDAR.

---

## 7. The auxiliary BEV target was vertically mirrored against the token grid

Found by an external review of the index arithmetic rather than by a
measurement, and then checked against one: the two conventions are both
written in this repository, three files apart.

The LiDAR pillar canvas indexes rows by `ix = (x - x0) / pillar_size` — row 0
is the cell the ego stands on, the last row is 32 m ahead. Both BEV targets
are written the other way up: `collect_data._bev_px` computes
`row = n - (x - x0)/(x1 - x0) * n`, and `decode_topdown`'s crop (the one
measured at 0.434 against LiDAR) puts x = 32 m at row 0 and the ego on the
bottom edge. Same columns, opposite rows.

`BEVSegHead` is a deconvolution — translation-equivariant, unable to express
a global flip — and the LiDAR branch has no intra-modal attention to route
information across the grid. So the head was asked to predict, at each token,
the class of the cell mirrored fore-aft about the grid centre. Roads are
often fore-aft symmetric over 32 m, which is why the seg loss still fell; the
vehicle class, the entire point of finding 6's repair, is not.

This retroactively explains a row that had been left unexplained: `no_aux`
scores 0.140 against the full model's 0.137, inside noise, where TransFuser
measured the same auxiliary supervision as worth 14 points of route
completion. An auxiliary target that is geometrically inconsistent with the
tokens it supervises carries almost no usable signal — and what does leak
through cross-attention teaches the tokens to describe the wrong end of the
road.

**The repair is one flip in one place**: `BEVSegHead.forward` reverses the
token rows before the deconvolution, so input and target align cell-for-cell
and the head's output stays in the target's own convention — which is what
the agent's `_bev_vehicle_distance` already assumes when converting rows back
to metres. The on-disk formats are untouched.

---

## 8. The evaluation agent kept the old camera yaws after the rig was corrected

When the camera rig was corrected for the published dataset (finding 2), the
yaws moved from ±55° to ±60° in `sensors.CAMERAS` — but
`leaderboard_agent.sensors()` carried its own copy of the list, and that copy
kept ±55°. Every closed-loop score so far, including the DS 4.10 headline,
was therefore driven with 5° of duplicated seam per side and both side
segments rotated 5° toward the centre: a projection the network was never
trained on, on every frame, invisible to open-loop error for the usual
reason.

The agent now reads the yaws from `sensors.CAMERAS`. One list, one rig. The
lesson is finding 2's, repeated because it repeated: a convention that lives
in two places will eventually be two conventions.

---

## 9. Two dead ends, recorded so they are not retried

**Aggressive creeping.** A 3 s threshold instead of 55 s: 21 paired routes gave
RC 18.6 against 20.7. The hazard interlock correctly refuses to creep into dense
traffic, where something is nearly always inside the box in front.

**Speed dropout.** Withholding the speedometer on 50% of training samples with a
learned "unknown" token, to break the network's habit of echoing it
(`tools/check_speed_reliance.py` sweeps the speed input on a fixed scene):

| | reliance |
|---|---|
| no treatment | 0.85 |
| dropout, epoch 11 | 0.77 |
| dropout, epoch 19 of 25 | **0.76** |

A full 25-epoch run moved it by 0.09 and left the standstill intact — at speed 0
it still predicted 0.30 m/s. Both were treating a symptom; the cause was finding
3.

`MeasurementEncoder.force_absent` survives from this work and feeds the learned
unknown token at inference. It is harmless and it is not a fix.
