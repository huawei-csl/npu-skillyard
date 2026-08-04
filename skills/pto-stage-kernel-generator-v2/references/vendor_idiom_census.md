# Vendor idiom census (CANN 9.1.0-beta.3, A2/dav-c220)

Corpus: the AscendC operator kernels that ship with CANN on the eval machine --
**991 `.cpp` + 4912 `.h`** under `$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/`.

**What this method can and cannot do.** Counting how often an idiom appears tells you what
is *normal practice*. It corroborates or contradicts a rule of the form "this is how it is
done". It **cannot** settle a rule of the form "X causes corruption" -- frequency is not
correctness, and a rare idiom is not necessarily wrong. Every entry below is labelled with
which kind of claim it supports. Where a query could not distinguish two things, it is
recorded as inconclusive rather than rounded into a conclusion.

This is a cheap instrument: the entries below cost a handful of greps and closed two
defects that had been open with a wrong hypothesis.

---

## 1. CORROBORATED

| our rule | census evidence |
|---|---|
| `COOK-§6.5` event-id model (per-pipe-pair pool, 8 ids on A2, split across the 2 AIV sub-blocks) | The runtime allocator `TPipe::AllocEventID<HardEvent evt>` indexes `eventPool_ + EventToIndex(evt)` -- a pool **per pipe-pair class** -- with `QUE_MAX_EVENT = 8` for `__NPU_ARCH__ == 2201`. Exactly the model the rule assumes. |
| `mixed` (Cube+Vec) is a real archetype needing its own treatment | **223 files** contain both `ASCEND_IS_AIC` and `ASCEND_IS_AIV` code; 135 declare a MIX task type. |
| Cross-core AIC<->AIV sync is normal, not exotic | `SyncAll` in **586** files; `CrossCoreSetFlag`/`WaitFlag` in ~285. (`FftsCrossCoreSync` in only 8 -- FFTS-specific sync is the rare form.) |
| `COOK-§10.8` offering `AtomicType::AtomicAdd` for cross-work-item reduction | `SetAtomicAdd` in **258** files -- mainstream, not a last resort. |
| Our ND->NZ-on-load path is the standard one (and its ~5% cost is worth knowing) | `nd2nz` in **333** files, `Nd2NzParams` in 155. |
| All 20 `(src,dst)` flag pipe pairs are usable | Every class appears. Top: `V_M` 792, `V_MTE3` 703, `MTE2_V` 660, `MTE3_MTE2` 462. |

## 2. CONTRADICTED -- rules that were wrong and are now fixed

| our rule | census evidence | action taken |
|---|---|---|
| "deep (>=3) prefetch rings are the goal" | `BUFFER_NUM` across the ~490 kernels declaring one: depth 1 = 145, **depth 2 = 316**, depth 3 = 4, depth 4 = 9, depth 5+ = 5. **Depth >=3 is under 4%.** `deep_norm_grad` itself ships at depth 1. | `COOK-§6.5` now says prefer depth 2; >=3 is exotic and measurement-gated. Two optimizer rounds had been spent chasing depth 3. |
| `deep_norm_backward` D5: "`wait_flag(PIPE_V, PIPE_MTE2, id)` needs re-probing or a scope note" | `HardEvent::V_MTE2` is used in **395** files. It is ordinary. | D5 **withdrawn**; our deadlock was our own protocol bug. |

## 3. THE LARGEST DIVERGENCE -- and it is ours

| | `pipe_barrier(PIPE_ALL)` | scoped `PipeBarrier<pipe>` |
|---|---|---|
| vendor | **19 of ~5900 files (0.3%)** | 1198 files |
| **our generated kernels** | **105 of 111 (95%)**, 417 occurrences | rare |

A `PIPE_ALL` barrier drains **every** pipe. It is the exact opposite of the overlap that
optimizer campaigns are spent trying to create, and we emit it by default in essentially
every kernel while the vendor almost never does.

**The scoped form has always been available to us.** `pipe_barrier` takes any `pipe_t` --
`pipe_barrier(PIPE_MTE3)`, `pipe_barrier(PIPE_V)` -- and pto-isa itself uses it that way
(`comm/async_common/ccu_trigger.hpp`). We default to the big hammer out of habit, not
constraint.

**Status: measured divergence, UNMEASURED performance claim.** No one has yet shown that
replacing these barriers is faster, and there is direct evidence it is delicate:
`dequant_swiglu_requant` attempt 1 removed intra-chain barriers and **failed validation**,
and `deep_norm_backward` D6 found that a per-item `PIPE_ALL` was silently protecting the
*output* tiles against a WAR hazard that no rule mentions. So the work is
**replace each barrier with the correctly scoped flag class**, not delete barriers, and
every step needs re-validation.

Treat this as the highest-value open optimizer hypothesis for the suite, not as a result.

## 4. INCONCLUSIVE -- recorded so it is not mistaken for evidence

* **`C1` (scalar `__gm__` access).** `GetValue(`/`SetValue(` appear in 973/463 files, but the
  query cannot distinguish `GlobalTensor` (a GM scalar access, what C1 is about) from
  `LocalTensor` (a UB access, which C1 does not restrict). **This census does not test C1.**
  C1 rests on our own hardware probing, which stands.

---

## How to extend this

Add a row only when you can state, in advance, which cookbook claim the count would
corroborate or contradict, and whether the claim is about *idiom* or *correctness*. A count
that would not change any rule is not worth running.
