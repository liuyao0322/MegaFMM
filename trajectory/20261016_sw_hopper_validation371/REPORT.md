# First dynamic-K Hopper SW implementation checks

Related CTest 4/4 pass (18.84s). The SW implementation test covers exact integer
original AB, fractional independent rounded SW oracle, short K, ring wrap,
rectangular S/T planes, more than one CTA wave, Graph fresh inputs, unaligned
SW fallback, and shared Plan / independently guarded two-stream workspaces.

Targeted `test_sw_pipeline --hopper-only` runs 48 fresh ordinary/Graph calls per
tool across both dtypes and 32/128 consumer widths. memcheck and synccheck have
zero errors; racecheck zero errors/warnings/hazards. This is implementation and
synchronization validation, not original FP64 precision acceptance.

N32 uses 96 registers, N128 uses 250, both zero stack/local memory. Both retain
C7515 serialization warnings. NCU attempted once but failed with exit9 because
/tmp/nsight-compute-lock could not be opened (InterprocessLockFailed); no dynamic
profile result is claimed. Continue with static instruction/resource evidence.

Design uses the official PDL dependency before the first transformed operand,
not a cross-CTA software spin barrier. NVIDIA reference:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html

Source identities and complete logs are retained. Original precision gates remain
unchanged; the uncompensated algorithm's known numerical failure is not fixed.
