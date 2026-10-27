> WITHDRAWN AS ACCEPTED BF16 EVIDENCE: this version converts BF16 to FP16, violating a prior user constraint reaffirmed2026-10-17. Raw measurements are retained for audit; this is not the current production implementation.

# Precision-valid BF16 SW profile after iter379

Frozen b28da3f library on H100 g08. Two sequential Nsight Compute captures of one4096^3 compute kernel, after100 warmed complete calls with two rotating original BF16 input bindings. Both configurations form FP32 transforms, store FP16 intermediates and execute seven products. These profiler replay times exclude packing and host launch overhead; they are not complete-call benchmark times or cuBLAS speedups.

| ID | Tile/BK/stages | Registers | Resident CTAs | Tensor pipe busy | DRAM cycles active | L2 busy | Eligible warps/scheduler |
|---|---|---:|---:|---:|---:|---:|---:|
|142|128/256/2|254|1|57.183%|40.019%|64.054%|0.1634|
|160|64/128/4|207|1|45.607%|52.600%|82.486%|0.2961|

The selected wide consumer has no local load/store requests in this capture. Registers restrict residency, but previous inline producer experiments with higher residency did not produce a selected-path gain. Therefore occupancy alone is not an adequate tuning objective. Tensor Core and DRAM counters do not establish a hardware floor. The smaller consumer raises L2 pressure and lowers tensor activity. Padded work, shared-memory consumption, pipeline scheduling, precision and full-call packing/launch costs must be modeled together.

Formal full-call4096 BF16 is351.220us versus cuBLASLt215.018us (0.6122x), and passes original FP64 checks. The compute-only profiler sample is280.288us; do not subtract it from formal full-call time to claim measured packing overhead because replay conditions differ.

Remaining research directions include two productive compute warpgroups sharing a producer and a different high-accuracy FP16 transform representation. Neither is implemented or certified by this capture. Previously rejected medium tiles, inline TMA and independent/inline wait1 schedules are retained in iterations373-377. FP16-input precision cannot be repaired by choosing between the present identical-arithmetic parameter variants. This is not a global impossibility result for other SW implementations.

Binary identity checks passed after capture. Raw counter CSVs, harness, script and source/library hashes are retained. The .ncu-rep files and executable are local diagnostic artifacts, not checked in.
