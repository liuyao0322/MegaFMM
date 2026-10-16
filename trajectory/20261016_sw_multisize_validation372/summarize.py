#!/usr/bin/env python3
"""Summarize frozen iter372 and six independent final-binary repetitions."""
from pathlib import Path
import csv, json, re, statistics, hashlib
root=Path(__file__).resolve().parent
bench=root.parent/'20261016_212727_iter-372'
rows=list(csv.DictReader((bench/'results.csv').open()))
assert len(rows)==28
summary=[]
pattern=re.compile(r'TIME shape=(\d+),(\d+),(\d+) dtype=(bf16|fp16) group=(\d+) case=(dense|solution) event_ms=([0-9.eE+-]+)')
for mode in ['launch','graph']:
    samples={}
    for process in [1,2,3]:
        assert (root/f'{mode}_{process}.exit').read_text().strip()=='2'
        procrows=list(csv.DictReader((root/f'{mode}_{process}.csv').open()))
        assert len(procrows)==6
        for r in procrows:
            assert r['actual']=='sw_hopper' and r['nonfinite']=='0'
        for match in pattern.finditer((root/f'{mode}_{process}.log').read_text()):
            m,n,k,dtype,group,case,event=match.groups()
            samples.setdefault((m,n,k,dtype),{}).setdefault((process,int(group)),{})[case]=float(event)
    for key, groups in samples.items():
        assert len(groups)==15 and all(set(v)=={'dense','solution'} for v in groups.values())
        dense=[v['dense'] for v in groups.values()]
        solution=[v['solution'] for v in groups.values()]
        paired=[v['dense']/v['solution'] for v in groups.values()]
        summary.append(dict(mode=mode,shape=list(key[:3]),dtype=key[3],groups=15,
          dense_median_us=1000*statistics.median(dense),sw_median_us=1000*statistics.median(solution),
          speedup_ratio_of_medians=statistics.median(dense)/statistics.median(solution),
          paired_median=statistics.median(paired),paired_min=min(paired),paired_max=max(paired),
          process_median_speedups=[statistics.median([v['dense']/v['solution'] for (p,g),v in groups.items() if p==proc]) for proc in [1,2,3]]))
assert len(summary)==12
(root/'replication_summary.json').write_text(json.dumps(summary,indent=2))
report='''# 多尺寸七乘法 SW 正式实现与复测（iter372）

生产源码提交：a9b5abf。H100 80GB / 132 SM，CUDA 12.4。完整 clean build 后测试，原 1024³ 源码 SHA256 保持 ebc37e9dd171dd5855cdaed91fbb030d250c052a4c459746dd33b82cc634264b。

正式路径仍执行七乘法 SW。1024³ 使用原生专用实现；其他对齐尺寸使用动态 K 的 TMA/WGMMA 七乘法；边界尺寸按逻辑半尺寸切象限、打包零填充、执行七个 WGMMA 叶乘积并掩码写回；1³ 使用七个 SIMT 叶乘积，单独标识，不冒充 Tensor Core 收益。未将完整 AB 替换为 dense，也未推广七 cuBLASLt 叶乘积的研究对照。

覆盖连续、非转置、alpha=1/beta=0、各维正数且 <=32768、A/B 16 字节对齐且 workspace 足够的显式 SWPipeline。其他布局或预算不足保留通用 SW。生产计算调用不分配或 host 同步，无跨调用输入值缓存。

28 项正式测量使用同进程独立调优 cuBLASLt，20 warmup、5 组×100 次、3 轮新输入原始 FP64 检查、2 轮换输入集。计时包含所有打包、变换、七乘法、输出重组和调用开销；耗时是 CUDA event 组均值的中位数，速度比为两侧中位数之比。setup/tune 不计入稳态；CSV 同时保留 wall 时间。GPU 未锁频，无置信区间。各进程固定 seed=1234567；独立复测指独立进程和计时，不代表使用新的独立随机种子。

全量支持不代表全量加速：1025×1537×769 明显更快，514×770×258 接近持平，1³ 减少启动开销；大尺寸仍慢于 cuBLAS，1024³ FP16 仍慢。原始精度门槛仍为 26/28 配置失败，仅 1³ 两种 dtype 通过，全部 nonfinite=0。

## 正式完整 28 配置

| M×N×K | dtype | cuBLASLt μs | SW μs | cuBLASLt / SW | 原始 eF | 精度门槛 |
|---|---|---:|---:|---:|---:|---|
'''
for r in rows:
 report+=f"| {r['m']}×{r['n']}×{r['k']} | {r['dtype']} | {float(r['dense_event_ms'])*1000:.3f} | {float(r['solution_event_ms'])*1000:.3f} | {float(r['speedup']):.4f}× | {float(r['solution_ef']):.8g} | {'PASS' if r['numerical_pass']=='1' else 'FAIL'} |\n"
report+='''
## 最终二进制独立重复

每模式 3 个新进程，每进程 5 组×100 次，共 15 个组，其他条件保持不变。下面保留同期 cuBLASLt、普通与 Graph 全部计时臂；不以 Graph 替代普通性能目标。最差配对组只表示已测样本下界，不是置信下界。

| 模式 | M×N×K | dtype | Lt μs | SW μs | 中位耗时比 | 配对组最小 / 中位 / 最大 |
|---|---|---|---:|---:|---:|---|
'''
for r in summary:
 report+=f"| {r['mode']} | {'×'.join(r['shape'])} | {r['dtype']} | {r['dense_median_us']:.3f} | {r['sw_median_us']:.3f} | {r['speedup_ratio_of_medians']:.4f}× | {r['paired_min']:.4f} / {r['paired_median']:.4f} / {r['paired_max']:.4f} |\n"
report+='''
## 自身验证和局限

相关 CTest 4/4（gpu_correctness、exact_winograd_dag、sw_pipeline_implementation、tuning_confirmation_policy）通过。完整 SW 实现测试包含舍入 SW 独立 oracle、奇数边界、未对齐回退、长 K 的分组尾部和缓存提示，以及共享不可变 Plan 的双流普通/Graph 新输入调用和输出/workspace guards。

memcheck、synccheck、racecheck 分别对当前二进制执行 --hopper-only 的 10 配置×12 次=120 次调用，覆盖两种 dtype、标量、两类边界和两种对齐消费者，共 360 次工具调用；自身日志为准。三种工具均零错误，racecheck 零警告、零 hazard。工具范围不包含所有 28 个基准配置或完整 16384³。各轮 poison 输出和 workspace，输入不断更新。实现测试不改变或替代原始 FP64 精度验收。

对齐大瓦片消费者达到 254 registers，当前 K 步之间的 wait0 限制重叠；这些是后续优化线索，不是已测硬件下限或唯一瓶颈。资源输出见 resources.txt。NCU 先前尝试受跨进程锁阻塞，未宣称新的 profiler 数据。所有失败候选和精确源码保存在 ../20261016_sw_multisize_candidates/，包括无效 C388 与编译失败 C402，均未作为性能证据。三类结构方向已试过但仍未实现普遍加速：片上变换融合、多消费 warpgroup 及中间结果暂存、独立七 cuBLASLt 叶乘积。宽松精度或直接 full-AB dense 均未用于制造达标结果。

源码与二进制哈希见 source.sha256 / binaries.sha256；完整原始全量计时见 ../20261016_212727_iter-372/results.csv 和 output.txt；本目录保留六次独立进程的全部日志、CSV 和返回码。
'''
(root/'REPORT.md').write_text(report)
print(json.dumps(summary,indent=2))
