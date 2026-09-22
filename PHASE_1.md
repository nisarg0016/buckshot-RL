# Phase 1: MDP Definition

Design decisions for the CSES Roulette RL agent, covering the game's state/action/reward
formulation and the self-play training setup. Godot remains the single source of truth
for game rules; the RL process communicates with the running game over a live bridge
(protocol design is Phase 2, deferred — see bottom of this document).

## Scope

- **Players:** locked to 2. Avoids a variable-target action space (opponent count would
  otherwise change as players die) and matches the self-play structure below.
- **Game coverage:** full game from the start — all rounds, all 5 upgrade types. No
  incremental no-upgrade-only phase.
- **Architecture:** Godot stays authoritative for rules/state. Nothing about the rules is
  reimplemented on the Python side; the bridge only carries observations, masks, and
  actions back and forth.

## Self-Play Structure

- A single policy controls both seats. Necessary because the game is multiplayer and no
  human opponent can realistically sit through the number of episodes needed for training.
- Training opponent = a pool of the policy's own past checkpoints, snapshotted
  periodically, rather than the live current copy playing itself in lockstep.
  - Opponent sampling from the pool: **uniform** initially; **weighted** (e.g. toward
    stronger/more recent checkpoints) as a later refinement.
- **Rejected alternative:** an information-asymmetric "cheating" opponent (full shell-order
  visibility) used to train an "honest" (counts-only) agent. Rejected because a
  perfect-information agent's optimal policy in the base game is trivial and deterministic
  (always self-shoot a known blank, always target-shoot a known live) — this makes it a
  near-unbeatable, low-signal training partner that teaches excessive risk-aversion rather
  than sound decision-making under the uncertainty an honest opponent actually faces.

## Observability

- Both seats see only the announced shell counts (`gameState.realCount`,
  `gameState.blanksCount`) — never the true shell order (`shotgunShells`).
- Exception: using the magnifying glass (`magGlass`) reveals the identity of the very next
  shell (`shotgunShells[0]`) for exactly one upcoming shot. That knowledge must reset to
  "unknown" the moment that shell is actually fired.
- This reset/tracking logic lives on Godot's side (it owns the rules) and is reported as
  part of the observation each step — not re-derived independently in Python.

## Action Space

Fixed, flat, discrete space of **12 actions**. Legality varies by state; size does not
(handled by action masking — see below).

| # | Action | Underlying call | Notes |
|---|--------|------------------|-------|
| 0 | `shootSelf` | `shootPlayer(caller, caller)` | |
| 1 | `shootOpponent` | `shootPlayer(caller, opponent)` | |
| 2 | `useCigarette` | `useUpgrade` → `useCigarette` | heals |
| 3 | `useBeer` | `useUpgrade` → `useBeer` | ejects current shell |
| 4 | `useMagGlass` | `useUpgrade` → `useMagGlass` | reveals next shell |
| 5 | `useHandSaw` | `useUpgrade` → `useHandSaw` | doubles next shot's damage |
| 6 | `useHandcuff` | `useUpgrade` → `useHandcuff` | skips opponent's next turn; target implicit (only 1 opponent) |
| 7 | `pickupCigarette` | `pickUpUpgrade` | takes any table instance of this type |
| 8 | `pickupBeer` | `pickUpUpgrade` | |
| 9 | `pickupMagGlass` | `pickUpUpgrade` | |
| 10 | `pickupHandSaw` | `pickUpUpgrade` | |
| 11 | `pickupHandcuff` | `pickUpUpgrade` | |

**Design rationale:**

- Upgrade pickup is indexed by **type**, not table **slot** — `generateRandomUpgrades()`
  regenerates the table's contents and count every round, so a slot-indexed action would be
  unstable across rounds. Any on-table instance of the chosen type is functionally
  equivalent.
- No explicit "pass"/"end turn" action exists. In `gameManager.gd`, `shootPlayer` and
  `pickUpUpgrade` call `endTurn()`; `useUpgrade` never does. Turn-chaining is expressed
  naturally through repeated `step()` calls by the same agent (see below), not a dedicated
  action.
- No separate target dimension for `shootOpponent`/`useHandcuff` — fixed 2-player scope
  means the opponent is always unambiguous.

Legality is fully state-dependent: round index gates which upgrade types can even appear
(`generateRandomUpgrades`), inventory gates which `use*` actions are available, `isUpgradeRound`
gates shoot-phase vs. pickup-phase actions, and `handCuffedPlayers` can zero out all legal
actions for a turn entirely. Handled via **action masking (MaskablePPO)**, not a reward
penalty for illegal actions. Godot's bridge must report this mask alongside every
observation.

## Turn vs. Step Semantics

- One `step()` call = one action from the table above.
- A single real in-game turn may span multiple consecutive `step()` calls by the same
  agent (e.g. `useCigarette` → `useMagGlass` → `shootOpponent`), since only
  `shootPlayer`/`pickUpUpgrade` end the turn.
- Consequence: most steps within a turn carry no direct win/loss signal. Credit assignment
  across the chain is handled by the return (discounted future reward via GAE), not by
  per-step terminal signals — see Reward Function.

## Observation Encoding

- Categorical/discrete fields are **one-hot encoded**, never packed into a single ordinal
  scalar — an integer encoding (e.g. blank=0, unknown=1, live=2) would falsely imply live
  is "more" than blank to the network.
  - Example: next-shell knowledge (populated after `useMagGlass`) is three binary flags —
    `knownBlank`, `unknown`, `knownLive` — exactly one set to 1.
- The full field-by-field observation vector is **not yet finalized** beyond this encoding
  principle. Revisit before implementation.

## Reward Function

Per-step reward, defined identically regardless of which action produced it:

```
reward_t = (own_hp_t − own_hp_{t−1}) − (opponent_hp_t − opponent_hp_{t−1})
```

- Damage dealt to the opponent → positive (their hp drop is subtracted, flipping sign).
- Damage taken → negative (own hp drop).
- Healing (`cigarette`) → positive (own hp rises).
- Non-damage actions (`beer`, `magGlass`, `handcuff`, all `pickup*`) → 0 direct reward. Their
  value is captured indirectly: the discounted return (GAE) propagates credit backward from
  the hp-changing actions they set up later in the trajectory. **No per-upgrade-type reward
  rule is needed** — the one formula above covers every action uniformly.

Terminal reward, added on top of the accumulated step rewards for the episode:

```
terminal_reward = +R_WIN   if this agent wins the episode
                 = −R_WIN   if this agent loses the episode
```

`R_WIN` should be set large relative to per-step hp-delta magnitudes (e.g. hp-delta worth
±0.1 per hp, `R_WIN = 10`), so the dominant optimization target stays "win the match" rather
than "accumulate favorable hp trades." Exact constants are a tuning detail, not a design
decision, and should be revisited once training is running.

**Further reading (optional):** potential-based reward shaping (Ng, Harada, Russell 1999)
is the formal treatment of adding shaping reward without altering the optimal policy —
worth a look if stricter guarantees on the shaping are wanted later.

## Evaluation Strategy

- **Floor check:** win rate against a uniform-random legal-action baseline policy.
- **Primary progress metric:** periodically freeze a checkpoint as a fixed reference;
  track newer checkpoints' win rate against that frozen reference over the course of
  training. An upward trend indicates the self-play loop is actually improving play, not
  cycling in place.
- No closed-form/known-optimal baseline exists for the full game (unlike the no-upgrade
  base case, which has a computable near-optimal probability-threshold strategy) — an
  accepted tradeoff of training on the full game from the start rather than incrementally.

## Explicitly Deferred (Phase 2+)

- Communication protocol between Godot and the RL process — message format, transport,
  triggering signal, synchronization with Godot's async/animation code.
- Training-speed practicalities: `useUpgrade`/`shootPlayer` paths run real tweens/animations
  (`await get_tree().create_timer(...)` etc.) that must be bypassed or disabled during
  training.
- Checkpoint pool mechanics beyond initial uniform sampling: snapshot frequency, pool size
  cap, weighted-sampling scheme.
- Full field-by-field observation vector.
