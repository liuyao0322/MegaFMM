# MegaFMM 多尺寸 SW 实现设计与 Agent 交接

日期：2026-10-15。状态：设计完成，尚未实现或测量新尺寸。

## 1. 任务与交付边界

用户希望了解如何把针对 1024 的实现推广到不同矩阵尺寸，并要求将设计文档交给另一个 agent 继续。本次仅提交设计与接手信息，不启动新一轮 GPU 实验。

建议后续实现目标：保留已测的 1024 专用路径，新增支持对齐 M/N/K 的单层 Strassen–Winograd 通用路径。先完成正确的形状推广，再按证据调优。奇数尺寸、非对齐输入、转置、一般 alpha/beta、递归分解及 Auto 选路分阶段处理。

精度问题仍未解决。功能推广完成不等于数值验收通过，也不等于获得性能收益。禁止通过修改参考、放宽门槛或忽略失败来宣称完成整体目标。用户已经停止成本过高的 FP8 残差修正路线，不得因本设计重新开启。

## 2. 接手时的仓库状态

- 工作目录：`/share/home/liuyao/workspace/MegaFMM`。
- 设计基点：分支 `opt/falcon-layout-fusion-331`，源码检查时 HEAD 为 `d43a096`。
- 当前整个 `solution/` 已在 iter336 从 `6d4167b` 逐字恢复，为布局/融合实验前的实现。
- `solution/sw_pipeline.cu` 的 SHA256：`748e7ace9de0b6469d47922cb7cb531bbb56a9d55c9915b0194c0e0a72e916e7`。
- 检查时 tracked 工作区改动为 `results/build.log`、`results/current.csv`，属于既有实验输出；不要清理或当作本任务源码修改提交。其他 untracked 历史产物也需保留。
- 已完成迭代至 336；实现 agent 开始前重新核对 Git 与 ITERATIONS，避免并行工作导致编号冲突。若无新增迭代，下一次正式实验为 337。
- 历史环境为 H100 / CUDA 12.4；旧 allocation107488 不保证继续有效，使用前必须验证。不要直接在登录节点启动 CUDA 作业。

先阅读：`HINTS.md`、`include/megafmm/gemm.h`、`solution/sw_pipeline.cu`、`solution/plan.cpp`、`scripts/bench.sh`、`scripts/log_iteration.py`，以及 `research/falcon_layout_fusion/REPORT.md`。历史 README、算法名字和较早 PLAN 中的参数可能过时，以当前源码为准。

## 3. 当前架构与需修改的硬编码

公开 `Problem` 已包含 M/N/K、lda/ldb/ldc/ldd、转置、dtype、alpha/beta；Dense 已支持一般形状。`Auto` 当前选择独立调优的 Dense，SWPipeline 是显式实验入口。

| 位置/概念 | 当前实现 | 通用路径要求 |
|---|---|---|
| `make_sw_pipeline` | 仅非转置、连续 1024³、alpha=1、beta=0 | 校验明确的对齐形状集合，保存形状元数据 |
| 输入变换 | Plane=512²、原矩阵 stride1024、象限偏移512 | A/B 分别按 M/K 与 K/N 计算，独立 plane pitch |
| `tensor_chain` | 大段静态 PTX，七乘积、每乘积四个 K tile | 可变 K tile 数；保留乘积次序、初始化及重组语义 |
| producer | q<512/BK，sequence==8 时等待变换 | 由 Ktiles 和数据依赖确定同步点 |
| TMA encode | 固定维度、stride、plane offset | 根据实际布局生成；检查全部 TMA 对齐条件 |
| 输出 | 固定 ldd1024 与象限偏移 | 使用 ldd、m/n 与合法输出边界 |
| launch grid | compute=(16,8)，transform=128×512 | 按工作量生成，并验证多波 CTA |
| workspace | 实际4MiB，但 plan 检查至少8MiB | 同一估算函数生成、校验并报告字节数 |
| SWState | 只有 dtype | 增加形状、布局、workspace、配置与 launch 元数据 |

当前计算使用转置后的叶子乘法 `B_leaf^T * A_leaf^T`。沿用 BM=64、BN=32 命名时，compute 的 x 对应原输出行方向，y 对应原输出列方向，不能按通常 GEMM 习惯直接交换。

## 4. 第一阶段支持集合

定义 A 为 M×K、B 为 K×N、FP32 输出为 M×N，沿用当前 row-major 语义。第一阶段要求：

- FP16 与 BF16；非转置；alpha=1、beta=0。
- 连续输入输出：lda=K、ldb=N、ldd=N；beta=0 时沿用现有 C0 校验语义。
- 正尺寸；M/N/K 均为256的倍数，作为保守初始集合。仍须校验 tensor map 的实际约束，不能仅凭此倍数判定支持。
- m=M/2、n=N/2、k=K/2；叶子尺寸满足 m%BN=0、n%BM=0、k%BK=0，初始 BM=64、BN=32、BK=128、Stages=8。
- Hopper 支持条件与资源限制在 plan 创建时检查；尺寸/字节数使用宽整数和溢出检查。

第一阶段不需要尾块 predicate。此限制也让 512³ 成为重要用例：每个乘积仅两个 K tile，能检查 K 循环缩短和 barrier 相位是否正确。

显式请求 SWPipeline 且不支持的形状应返回清楚的错误，不应悄悄换算法。Auto/Dense 保留现有行为；在后续获得数值和性能验收之前，Auto 不启用 SW。

## 5. 通用布局与 workspace

保留单层分解及七个乘积的现有代数关系。每个叶子 GEMM 计算 `(m×k)·(k×n)`，继续由一个 CTA 完成其输出 tile 对应的七个乘积与 FP32 重组，初期不新增全局 P 缓冲。

四个 S 平面各含 m×k 元素，四个 T 平面各含 k×n 元素。A/B 侧分别定义 row pitch 和 plane pitch，不再共享单一 Plane 常量。布局字段需明确以元素还是字节计量。

无额外 padding 时，FP16/BF16 workspace 为：

`W = sizeof(T) * (4*m*k + 4*k*n) = 8*k*(m+n) bytes`。

方阵边长 L 时为4L²字节：1024为4MiB、2048为16MiB、4096为64MiB。后续加入 padding/对齐时以实际 pitch 计算，不能继续使用理想公式作为分配大小。

建议 SWState 保存 Problem 或所需字段、m/n/k、Ktiles、S/T pitch/offset、workspace_bytes、config_id、compute_grid、transform_grid。静态 tile 配置作为少量模板实例存在，形状与 stride 作为运行时元数据传入。

TMA 中当前 A 侧将 K 按64拆成外维，需要重新推导实际维度与 stride。描述符必须绑定本次调用的 A/B/workspace 地址；不缓存变换数值，不复用绑定旧指针的 map。保持 run 无分配、无 host synchronization，并保持 Plan 的并发调用语义；不得在共享 Plan 上无保护地写入每次调用的描述符。

## 6. K 流水线与重组

通用 producer 顺序：

```text
Ktiles = k / BK                  # 第一阶段整除
for step = 0..6:
    product = existing_product_order[step]
    for q = 0..Ktiles-1:
        sequence = step*Ktiles + q
        slot = sequence % Stages
        wait for previous consumer of slot when necessary
        ensure transformed inputs are visible before their first load
        load this product's operands for q
        publish the ready barrier
```

consumer 需同样从 sequence 推导 slot 与 phase。每个乘积第一次 MMA 覆盖对应 accumulator，后续 K tile 累加。七个乘积结束后的 FP32 加减顺序优先保持现状，避免把形状推广与数值算法改变混成一次实验。

当前 tensor_chain 不是简单 C++ 循环，需要将其拆为可复用的 MMA/descriptor/epilogue 单元或采用可审计的代码生成。第一版不应为每个全局 K 完全展开所有 PTX，以免代码体积和寄存器压力随 K 增长。可在通用循环稳定后，为短且常见的 Ktiles 增加特化。

barrier 初始化计数、事务字节数、WGMMA wait/fence、shared proxy fence 与 slot 生命周期必须与实际生产/消费关系一致。尤其要测试总 tile 数小于 Stages、刚好绕回和多次绕回。

## 7. PDL 与调度

保留 transform→compute 两个 kernel 的 PDL 结构。前两个 raw 产品可尝试与变换重叠，在第一次访问 S/T 前完成依赖同步。等待位置不得固定为 sequence==8；初始顺序不变时可由2*Ktiles推导，也可显式表达 first-transformed-use。

producer 不得在同步完成前预取依赖变换结果的数据；consumer 与 producer 的 barrier 协议也要覆盖该依赖。PDL 并发是机会性的，程序必须在两 kernel 串行调度时也能完成，不得依赖实际重叠才能前进。

当前 kernel 名字含 cooperative，plan 还检查 cooperativeLaunch 与128个驻留CTA容量，但实际 launch 使用 PDL 普通 kernel，未采用 grid-wide cooperative barrier。应审查真实同步需求后调整遗留限制；不能直接把“全部 CTA 同时驻留”作为所有尺寸的要求，也不能未经分析删除保护条件。

沿用当前计算方向：`grid.x=m/BN`、`grid.y=n/BM`。更大方阵的 CTA 数为1024→128、2048→512、4096→2048。先让普通多波调度正确运行，后续才考虑 persistent 调度、CTA 顺序或 tile 调整。transform 的 grid 按两侧工作量选择，保持所有 CTA 正确触发 PDL launch completion。

官方依赖语义：[NVIDIA Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

## 8. 后续边界与接口扩展

第二阶段再考虑任意 size。不能仅使用 ceil(M/2)、ceil(N/2)、ceil(K/2)，然后依赖原始矩阵 TMA 的整矩阵 OOB 补零：越过逻辑象限边界可能读到相邻象限，而地址仍在原矩阵内。

可选方案为显式 padding/pack 输入、给象限建立有正确边界的加载表示，或对主体与尾部分别计算。第一种最易验证，但额外读写和内存必须进入 workspace 与完整调用计时。只对变换 kernel 加 mask 不能修复 raw 产品的跨象限读取。

随后扩展合法 leading dimension、转置和 alpha/beta。一般 epilogue 在 FP32 中计算 alpha*AB+beta*C0；beta=0 不读 C0；若接口允许 C0 与 D 别名，需保持正确语义。任意边界写回必须保证每个合法 D 元素恰好得到正确结果且无越界。

初期保持一层分解，避免同时引入递归带来的误差、workspace 与调度问题。

## 9. 实施顺序与验收

| 阶段 | 交付物 | 验收条件 |
|---|---|---|
| A：元数据 | SWState、workspace estimator、形状检查，保留1024专用路径 | workspace检查与报告一致；不支持形状清楚拒绝；接口约定保持 |
| B：通用kernel | 动态变换、TMA、K循环、grid与epilogue | 对齐方阵/长方阵实现验证通过；报告原始数值门槛的实际结果 |
| C：并发与测量 | Graph、多stream、sanitizer、完整延迟 | 无越界/竞态；无隐式分配同步；所有失败保留证据 |
| D：有依据的调优 | 少量tile变体、形状选择数据 | 每个形状与独立调优Dense公平比较；不以单kernel时间代替完整调用 |
| E：广泛形状 | 边界/stride/转置等选定扩展 | 独立边界验证；额外成本计入；未支持项保留明确行为 |

阶段 B 的功能验收与数学精度验收应分别报告。若新路径与单层低精度 SW 参考一致，但对原始输入 FP64 的门槛失败，只能称为推广实现正确，不能称为数值可用。当前基线已经失败，不要求通过隐式放宽门槛来让阶段 B 显示全绿。

## 10. 验证矩阵与指标

优先方阵512、1024、1536、2048、3072、4096；长方阵建议(1024,2048,512)、(2048,512,1024)、(512,1024,2048)，元组顺序为(M,N,K)。Ktiles包括2、4、6、8、12、16，以覆盖较短流水线和反复绕回。这是未来验证计划，并无新尺寸性能结论。

每个候选覆盖两种dtype与fresh input，至少包括普通随机、全零、稀疏、抵消和动态范围输入。保留两个不同参考：

1. 实现参考：独立的单层 SW 计算，模拟实际变换量化与重组规则；1024时另与冻结源码比较。浮点累加顺序不同造成的差异要解释，不能把所有新形状都要求逐位相同当成唯一标准。
2. 数学参考：同一份原始已舍入 A/B 的 FP64 普通 GEMM，保留当前相对 Frobenius 误差定义和 Dense 比较。

原始门槛固定为 `eF_SW <= 3*eF_dense + floor`，BF16 floor=1e-3、FP16 floor=1e-4；要求全部新输入轮次通过且无nonfinite。不得替换为已变换输入的参考。

当前1024基线 BF16 eF约0.0060152、FP16约0.000754626，分别高于约0.00100256、0.00010375的当次门槛；这些是历史样本，不是其他尺寸的预测。

对同步有改动的版本，执行代表性尺寸的 memcheck/racecheck/synccheck，并覆盖普通 launch、Graph、双stream独立workspace、调用间poison、改变输入/输出/workspace地址。用canary或等效方法检测workspace边界。不要将旧1024候选的sanitizer结果套用于新kernel。

性能报告逐行给出形状、dtype、配置、workspace、普通/Graph模式、完整SW延迟、独立调优Dense延迟、加速比、误差及数值是否通过。未知或失败项明确标记。先做小范围验证再扩大，避免大尺寸FP64参考导致无效的高成本循环。

## 11. 实验与提交约定

接手实现时阅读 AKO4ALL 与 cluster-llm-ops 技能及最新 HINTS，确认分配与构建环境。不要覆盖现有脚手架或历史数据。

现有 `scripts/bench.sh` 通过 MEGAFMM_SHAPES / MEGAFMM_DTYPE / MEGAFMM_ALGORITHM / MEGAFMM_GRAPH 等参数运行；先检查 benchmark 的形状列表格式和路由名称，再组织多形状调用，不猜测解析规则。

修改 solution 后的每次正式 benchmark 按现有协议执行：`bench.sh iter-N`，随后下一次工具调用记入 `scripts/log_iteration.py`，再下一次工具调用执行相应 Git commit。失败/部分结果也需记录，避免在日志和提交之前插入新探针或下一轮改动。不要把本次纯文档交接计作性能迭代。

最终应交付：源码、可重复的验证命令、原始结果与简洁报告；说明支持集合、回退/报错行为、workspace、数值失败项、相对1024专用路径的变化及各形状的真实性能。仅在数值和完整调用性能都接受后讨论让 Auto 使用 SW。

## 12. 给接手 Agent 的启动提示

> 在 `/share/home/liuyao/workspace/MegaFMM` 接手多尺寸 SW 推广。先阅读本文件、HINTS 和当前源码，核对 Git 状态与迭代号。保留1024专用实现，按 A→B→C 顺序实现对齐形状的通用单层路径，首先验证512/1024和一个长方阵，再扩大。保持FP16/BF16、原始输入FP64门槛、完整调用计时、run无分配无host同步、Auto=Dense。先解释并记录功能正确性与当前数值门槛失败的区别。不要恢复FP8修正路线或以递归、任意边界支持扩大首轮范围。复用实验协议和集群资源管理；每次正式bench后立即log、commit。报告真实通过项和仍未解决的问题。
