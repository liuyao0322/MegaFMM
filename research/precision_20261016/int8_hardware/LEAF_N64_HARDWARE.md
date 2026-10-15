# P54 hardware: native N64 leaf shape control, rejected

P54 is rejected. The working four sources and executable are restored byte-exact
to retained P51; a new process passes78 original FP64 gates and79872 exact inside
exponents. Production/library/Auto remain unchanged and the goal remains active.
Scope: contiguous non-transposed1024^3,alpha1,beta0,original FP16/BF16,FP32output.

## Controlled hypothesis and own checks

Frozen LEAF_N64_DESIGN.md and CPU generator/proposal preceded source application.
Relative to frozen P53, widen native MMA from64x32 to64x64, d/r fragments from16
to32 values, grid896 to448, two49152B operand slots to two65536B slots, actual
Shared98432 to131200B. Seven fullK512 main products, all eight native scaled
cross dots, exact original inside, p11FP16/p8BF16 and the scalar recombination
kernel remain. The additional7MiB leaf workspace and all three kernels remain
within complete call timing. No allocation, host sync or input cache in run.
Generated sources matched frozen CPU hashes before compilation.

Own pre-MMA probe: six fresh synthetic configurations, seeds119101+457*trial
(FP16 additionally239), all448CTAs, four K128 sequences and both reused slots.
The verbatim P54 TMA producer plus independent generic swizzle reads compare
276824064 native uint16 values including CTA reuse;5767168 unique native
workspace values per configuration. All11010048 seven-plane output coordinates
and shared/workspace/output guards pass, zero bit errors. Generic reads release
the current slot without MMA: this is not a WGMMA synchronization certificate.
Own runtime check finds one CTA/SM for both dtypes on132SMs.

Own GPU scalar recombination: three fresh signed/scaled/canceling leaf arrays
and signed zeros, seeds120103/120564/121025. An independent CPU oracle updates
the four outputs in P51 leaf order. All3145728 output bit comparisons and5505792
unchanged input/guard values pass, no nonfinite. A deliberately changed C12 order
differs22367/22532/22487 outputs. This does not prove complete matrix bit equality.

Own build: consumer106 registers for both dtypes, recombine18, prep52BF16/48FP16,
zero stack/spill and no C7515/C7510 warnings. Standard78 original FP64 gates and
79872 exactinside exponents pass. MaxeF BF16/FP16=1.61320431617e-5/2.16394614392e-5;
each printed per-case eF equals P51, without a complete-output bitwise claim.

## Independent full-call comparisons

Each row pair comes from four independent ABBA processes with312 original gates
and319488 exactinside exponents per mode. All pass. Numerical configurations are
the same78 deterministic cases, not fresh seeds. All timing arms are retained;
unlocked clocks, no confidence intervals. Shape and retention controls use
separate processes. Ordinary includes descriptor construction, prep, seven-leaf
writes, scalar recombination and all launches; Graph includes captured GPU work.

| Control | Mode | dtype | old us | P54 us | P54 slower |
|---|---|---|---:|---:|---:|
| P53 shape | ordinary | BF16 |40.957719326|41.198839665|0.5887%|
| P53 shape | ordinary | FP16 |41.041360378|41.276760101|0.5736%|
| P53 shape | Graph | BF16 |37.548600197|37.724440098|0.4683%|
| P53 shape | Graph | FP16 |37.546919823|37.938601017|1.0432%|
| P51 retention | ordinary | BF16 |27.447799921|41.202840805|50.1135%|
| P51 retention | ordinary | FP16 |27.319679499|41.248278618|50.9838%|
| P51 retention | Graph | BF16 |27.311999798|37.803199768|38.4124%|
| P51 retention | Graph | FP16 |27.474519968|37.953359604|38.1402%|

No own fresh full-matrix dual-stream/guard stress or sanitizer acceptance was
performed for this rejected candidate. Other candidates' certificates do not
transfer; synthetic probe guards have the narrower scope above.

## Own profile and reassessment

Own first balanced BF16 full call,17 profiler passes per kernel:

| Kernel | profiler us | registers | dynamic shared B | resource limits CTA/SM | issueactive |
|---|---:|---:|---:|---|---:|
| cooperative prep |17.280|52|65536|shared3,register4|14.610248%|
| corrected leaf |22.912|106|131200|shared1,register2|17.546221%|
| scalar recombine |6.624|18|0|shared32,register10|18.293527%|

All three have zero measured local read/write sectors. Profiler-process timing
is excluded from ABBA; these three times must not be summed into a decomposition
of the independent complete call. Nominal source TMA payload92274688B/5632commands
is2/3 the bytes and1/2 the commands of P53; native MMA source instruction count is
also1/2. These counts do not measure physical traffic. Tile width, grid and actual
capacity change together; no unique causal attribution or physical floor follows.
The experiment rejects this N64 leaf design as a way to recover complete latency.

After P52/P53/P54 fail to improve retained P51 by3%, own profiles, M8/M9/M12/M14,
and P35–P39 history were reread. Fresh primary reading of EmuGEMM §II-C1/III-A/B
supports per-row/column power-of-two scales and fused integer accumulation as
design ingredients. Its large-matrix results do not predict our1024 case; §V-G
explicitly discusses preprocessing at512–1024. PTX8.4 integer WGMMA specifies
INT32 overflow behavior; any new fullK integer pairing needs its own bound.

- https://arxiv.org/html/2606.25453v1
- https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions

Next numerical work will examine finer scale groups with fullK512 INT8
corrections, preserving all7+8 dots, original inputs and gates. Existing M9 used
coarse64-row/32-column groups; actual P35–P37 hardware requantized already packed
INT8 values and paid extra barriers. Neither is evidence for a direct fine-group
producer. Numerical screening must precede a hardware proposal; prep, metadata,
reloads, conversion and synchronization costs must all be counted before retention.
