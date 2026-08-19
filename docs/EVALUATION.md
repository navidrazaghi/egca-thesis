# Evaluation protocol

How to run the closed-loop benchmark, and the traps that produce numbers which
look valid and are not.

---

## Longest6 is not the stock leaderboard protocol

Two differences push the score in opposite directions, so using the wrong
evaluator does not simply add noise — it produces a figure comparable with
nothing:

- Longest6 runs **dense traffic** (500 spawn points) where the stock protocol
  runs light traffic
- Longest6 applies **no stop-sign penalty**

It must run through the vendored `leaderboard_evaluator_local.py`, with
`DATAGEN=0`. Both are set in `tools/eval_env.sh` with the measurement that
justifies them. A first pass here used the stock evaluator and reported RC 87.94
against the reference's published 82.71, which read as a pass and was a harness
that had made the benchmark easier.

## Score arithmetic

`DS = mean over routes of (RC × IS)` — the mean of the products, not the product
of the means. Infraction multipliers: pedestrian 0.50, vehicle 0.60, static 0.65,
red light 0.70, stop 0.80.

Route termination: **blocked** is speed under 0.1 m/s for 180 s; **timeout** is
0.8 s per metre plus 5 s.

---

## Running it

```bash
# full protocol, 36 routes, 4 parallel simulators, about 11 h
OUT=$HOME/thesis/code/results/eval_myrun \
AGENT=$HOME/thesis/code/egca/carla_sim/leaderboard_agent.py \
AGENT_CONFIG=$HOME/thesis/code/configs/agent/myrun.json \
TRACK=SENSORS SLOTS=4 PORT_BASE=3000 TM_BASE=8200 \
  bash tools/run_autopilot_val.sh
```

`tools/run_after_ladder.sh` wraps this: it waits for training to release the
card, writes each agent config, verifies the model and controller build from it,
then evaluates. Use it rather than calling the runner directly — the guards live
there.

### Ports

`PORT_BASE` and `TM_BASE` are overridable because the box is shared. `boot()`
refuses a port already served, because docker will happily start a container that
cannot bind and the slot then spends its whole run talking to nothing. Six routes
were lost that way once.

### Controller settings must be identical across compared runs

Write them explicitly into every agent config rather than taking them from each
checkpoint. Checkpoints trained before a control feature existed do not carry its
keys at all, which both crashes the controller and — worse — would drive one
model with a standstill-escape heuristic and its comparison without one, then
attribute the difference to the architecture.

---

## Screening: a cheaper answer to "does this idea help at all?"

Eleven hours per evaluation and twenty-one per training is too expensive for
exploration. Two switches make a fast pass possible.

**`data.tf_frame_stride`** keeps every Nth frame. Stride 4 leaves 50,783 training
frames. Striding across all towns rather than dropping towns preserves the layout
diversity a held-out-town split exists to test, and it recovers the I/O as well:
training is bound by random reads over 234 GB against roughly 60 GB of page
cache, so the ~59 GB working set stays cached after the first epoch and the run
is faster than the frame ratio alone predicts.

**`ROUTE_IDS`** drives a subset. A stratified twelve (two per town):

```bash
ROUTE_IDS="0 1 6 7 12 13 18 19 24 25 30 31"
```

### What a subset costs, measured

Over one full run the route-to-route DS spread is 6.25.

| routes | 90% of estimates within |
|---|---|
| 6 | ±4.5 |
| 9 | ±3.4 |
| 12 | ±2.4 |
| 18 | ±1.8 |
| 24 | ±1.2 |

**Pairing does not help.** The route-level DS correlation between two runs is
−0.09: a route's score is decided by where the policy happens to fail rather than
by the route's difficulty. RC correlates at +0.42, but DS is what is reported.
Only the count helps.

So screening sees effects above roughly 3 DS and nothing smaller. Screening
numbers are comparable **between arms of the same screen**, never with a
36-route figure.

---

## Traps

**A missing agent config is recorded as a driving result.** The evaluator catches
the exception, registers the route statistics, and writes DS 0.00 for all 36. The
chain now refuses to start unless the config exists and the model and controller
actually build from it.

**A simulator timeout is recorded as a driving result.** "A sensor took too long
to send their data" on a contended GPU produces a finished route with DS 0.00 and
no infractions — indistinguishable from a policy that never moved, and it drags
the mean down by a whole route. `scored()` now rejects any record whose status
mentions a crash, so the slot retries on a fresh simulator.

Excluding those, `tf_base_rl` is DS 4.10 / RC 29.25 over 34 routes rather than
3.87 / 27.63.

**The exit code lies.** `echo "$(date +%T) exit=$?"` reads the status of `date`.
Capture `local rc=$?` before anything else runs.

**`best.pth` is only as good as the validation metric that chose it.** A run whose
validation was broken has a best checkpoint selected on nonsense — `tf_base_rl`
had its frozen at epoch 3. Its training targets were never affected, so
`last.pth` is a fully trained model, and the cosine schedule anneals to zero
anyway (the previous run's best and last differed by one millimetre).

**`pkill -f` matches your own command line.** Bracketing the pattern only helps
when the string appears nowhere else; if the launch command contains it, pkill
kills the shell running it. Twice.

**The box is shared.** Another user's four CARLA instances alongside four of ours
is enough to make sensors time out. Check `ps -eo comm | grep -c CarlaUE4` before
choosing `SLOTS`.
