# BN16 scalar tail-bank CPU geometry and selected future derivatives

This independent prefix records a CPU-only scalar4-byte-word address review.
The raw and SS original BN16 candidates, their eight artifacts each, and both
manifests are frozen and unchanged. The model hashes every old file and each
old manifest before and after. Every new output path is restricted to
`.scratch/psplit_bn16_tailbanks_future38_`. No compiler, GPU, benchmark, formal
source, results or Git mutation is involved. No next iteration has started.

The real model process exited0 with empty stderr and status
`PASS_CPU_LITERAL_SCALAR_TAIL_BANK_GEOMETRY_ONLY`. The detailed lane map JSON
contains all32 scalar lane addresses, bank indices, unique words by bank, lane
activity, coordinates, product, consumer WG/CTA warp and fragment slot or U
loop round. The summary JSON and stdout separate STORE and U LOAD metrics.

## Pinned source binding and exact scope

CUTLASS HEAD remains `f7b19de32c5d1f3cedfc735c2849f12b537522ee`.
The exact C layout is at
`third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:389`:
thread mode(4,8,4), value mode(2,2,2), strides((128,1,16),(64,8,512)).
The FP16/BF16 N16 SS and RS traits all bind ThrID128 and this C layout.
The model asserts those literal declarations and binds the exact header SHA.
It does not instantiate the owning CuTe fragment or inspect emitted SASS.

For local consumer tid0..127 and slot0..7, the flattened pinned coordinate is:

```
t0 = tid % 4
t1 = (tid / 4) % 8
t2 = (tid / 32) % 4
v0 = slot % 2
v1 = (slot / 2) % 2
v2 = (slot / 4) % 2
row = t1 + 16*t2 + 8*v1
col = 2*t0 + v0 + 8*v2
word = 1024*product + 16*row + (col XOR mask(row))
bank = word % 32
```

Products P1/P2/P3 use CTA tids128..255, warps4..7; P4/P5/P6/P7 use tids256..383,
warps8..11. Local fragment tid is CTA tid modulo128. Each P has4 warps×8 slots
=32 STORE instances, each with32 active lanes and32 distinct words. Seven P
give224 STORE instances/mask. Product stride1024 words is divisible by32;
the dynamic typed tail base has128-byte alignment, so its base is bank-neutral
under this model. Both raw and SS candidates have the identical tail mapping.

All five masks are in0..15 for rows0..63 and are rowwise XOR bijections. Every
mask covers exactly all7,168 typed tail words, each once in STORE and once in U.

| Mask | STORE occupied banks/instance | Distinct words/occupied bank | U LOAD occupied banks/active instance | Distinct words/bank |
|---|---:|---:|---:|---:|
| Current `row&3` | 8 | 4 | 32 | 1 |
| `0` | 8 | 4 | 32 | 1 |
| `(row&1) OR ((row&2)<<2)` | 16 | 2 | 32 | 1 |
| `[(row&1) OR ((row&6)<<2)] AND 15` | 16 | 2 | 32 | 1 |
| Selected `((row&2)>>1) OR ((row&4)<<1)` | 32 | 1 | 32 | 1 |

Each STORE result holds for every slot, warp and product, not just the worst
sample. The current and intermediate multiplicities are distinct addresses in
the same bank, not repeated requests to one word. Same-word duplicate requests
are0 for all compared instances, so no broadcast correction changes this table.
For example, current warp0/slot0 sends words0,32,64,96 to bank0.
The middle two masks are exactly identical over all rows: AND15 removes the old
expression's contribution from row&4.

## Why the selected mask has one scalar STORE word per bank

For fixed warp/slot, the selected mask places row bit1 into column bit0 and row
bit2 into column bit3. The five bank bits become:

```
bank bit0 = v0 XOR row bit1
bank bit1 = t0 bit0
bank bit2 = t0 bit1
bank bit3 = v2 XOR row bit2
bank bit4 = row bit0
```

The32 lanes contain every combination of t0's two bits and t1's three bits.
All five bank bits are independent, so each bank appears exactly once. The mask
values for rows0..7 are0,0,1,1,8,8,9,9; it is valid for fixed N16 and does not
change the row/product strides, tail size or output coverage.

An exhaustive bounded family search additionally tested4096 periodic8 linear
XOR masks, where each of row bits0,1,2 selects an arbitrary four-bit coefficient.
On representative warp0/slot0,256 masks have worst4 distinct words/bank,2304
have worst2 and1536 have worst1. The selected coefficients are(0,1,8). Only the
selected mask's result is asserted over all warps/slots/products by this search
summary; the detailed comparison separately checks all five specified masks.
One word per bank reaches the scalar full-warp lower bound in this address
model, but the mask is not claimed unique or optimal for GPU execution.

## U LOAD activity and geometry

Unified U uses fixed CTA tids0..383 and `linear=tid+384*round<1024`. This is not
a128-thread consumer fragment loop. The exact active masks are:

| U round | Active CTA warps | Active lanes/active warp | Inactive warps |
|---|---:|---:|---:|
| 0 | 0..11, total12 | 32 | 0 |
| 1 | 0..11, total12 | 32 | 0 |
| 2 | 0..7, total8 | 32 | 8..11, total4 |

There are no partly active32-lane warps in this fixed1024-coordinate tile.
The model nevertheless stores actual active-lane lists and excludes zero-active
instances from multiplicity statistics. Each product has32 active warp/round
LOAD instances and4 zero-active records; seven products have224 active and28
zero-active records/mask. Inactive lanes contribute no word or bank request.

Each active warp covers two consecutive complete16-column rows with opposite
parity. Row parity selects opposite16-bank halves; any legal row XOR permutes
each half. Every compared mask therefore leaves the U LOAD at32 distinct words
across32 banks, one each. There are no same-word LOAD repeats or broadcasts.

## Exact one-line future derivatives

The selected future hypothesis is copied into two new files in this prefix:

- `psplit_bn16_tailbanks_future38_raw_candidate.cu`, SHA
  `015f6ca799af720a859ffdf73db2655de3f927bb31e3e0bf5751832ea4d8353d`.
- `psplit_bn16_tailbanks_future38_ss_candidate.cu`, SHA
  `ed8123e87b931ab202241f170ff32fc666678d3a7a69481193de4b90f3ad8b86`.

Each derivative changes exactly one occurrence:

```cpp
// Original frozen line:
const int xor_mask = row & (Cols/4-1);
// Derivative line:
const int xor_mask = ((row&2)>>1) | ((row&4)<<1);
```

The byte replacements preserve original indentation/newlines. Inverse replacement
restores its own original candidate byte-for-byte, bound to raw SHAab978377...
and SS SHA94c482bc.... The exact full parent/derivative hashes and line bytes
are in source_delta.json and review.json. No old candidate/protocol/manifest was
changed. Register quotas, launch bounds, source mathematics, shared capacity,
host qualification, barrier counts, graph/fallback contract and all other bytes
remain their frozen hypotheses.

These derivatives are uncompiled, unregistered selected future hypotheses only.
Their bank result does not prove compiler instruction grouping, actual shared
transaction count, broadcast handling, bank stalls, latency, GPU speedup, qfit,
spill behavior, physical residency or GPU numerical precision. A future formal
owner may choose them for first compilation in a separately approved experiment;
this review does not begin that experiment or defer current iteration38 work.
