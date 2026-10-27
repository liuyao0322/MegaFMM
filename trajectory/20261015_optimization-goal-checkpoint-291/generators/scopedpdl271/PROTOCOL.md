Experimental synchronization protocol; not independently accepted yet.

CUDA12.4 runtime cudaTriggerProgrammaticLaunchCompletion documentation: every CTA must exit/trigger before dependent scheduling is enabled; trigger provides no memory visibility itself, and user can enforce visibility with a fence of the correct scope.
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__EXECUTION.html

Each of128primary CTAs owns one padded flag and one32x64input tile. Its leader overwrites its own flag0 and executes GPU-scope threadfence before triggering PDL. All128CTAs must trigger, so initial zeros have been fenced before any secondaryCTA can start; zero ownership is unique and does not race another initializer. Initialization therefore does not require an all-grid software barrier. This relies on documented fence visibility plus PDL scheduling condition; trigger alone is insufficient.

Every primary worker stores all8S/T planes for its tile, executes async-proxy global fence, and joins CTA syncthreads. Leader then release-stores1. Consumer acquire-loads corresponding flag1 before issuing TMA, providing publication of generic writes and async-proxy visibility. Primary never waits for secondary progress, and completion is correct even if PDL launches serialize.

The first transformed leaf requires8LHS tiles and4RHS tiles for each K half. Those same input regions cover every subsequent transformed leaf because each primary tile produces all8planes. Wait these24flags once. Raw P1/P2 never access packed data.

A following ordinary primary launch remains ordered after prior work in caller stream. Distinct concurrent invocations use distinct caller workspace, so no shared flags or cached tensors acrossstreams. Nohost allocation/synchronization inrun. Extra flags occupy16384bytes after4MiBpacked data; tmap remains based atpacked+2MiB.

Require independent poisoned fresh-input ordinary/Graph stress, multistream, and sanitizers before retaining. Numerical agreement alone does not establish memory-order proof. Preserve original FP64 gates and report bothFAIL.
