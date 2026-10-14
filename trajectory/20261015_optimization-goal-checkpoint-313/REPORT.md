# 持续优化检查点：292–313，目标仍在进行

保留282(1bdfb29)，313(79dad0b)逐字恢复并完整复测。BF16普通调用已快于Dense，FP16仍未超过。没有物理下限或方向耗尽结论；继续优化，无需再次授权。

## 当前性能与精度

313新测：BF16 SW10.29344us / Dense12.46912us；FP16 SW10.24576us / Dense9.79072us。FP16还需减.45504us，即当前SW时间4.44%。BF16一组wall36.984us/event10.25536us异常保留；所有原始组均存档。独立调优Dense BF16这次cluster14，与此前cluster10不同，未强制劣化Dense。

完整282/303同期ABBA：普通BF28210.27952、30310.29696us；普通FP28210.22128、30310.25584us。303去掉WGMMA wait0之后warp.sync未改善普通调用，不保留。Graph BF10.00624/10.00256、FP10.05104/10.01648us，仅作独立模式记录。此前279/282普通与Graph对照见291检查点，勿把跨轮绝对数值相加当新对照。

原始FP64门槛仍FAIL两种dtype：最大eF约BF.0060151965/FP.00075462577，门槛BF.0010025568/FP.0001037528，非有限值0。所有性能实验不构成精度验收。Auto=Dense；未恢复补偿旧架构，保持用户保留的七乘法无补偿路线。

282已有72逐位baseline162检查、3quick sanitizer、256双stream/Graph调用通过；源码hash重新核对匹配。303也独立通过同样检查，但因普通调用无收益而拒绝；检查只说明实现与基线一致，不等于原始数值验收通过。

## 新profiling证据

- reassess292：选定282完整NCU，两dtype60regs，local读写sector0，约1.268M指令。wait/issue约1.558，barrier/issue约1.48，longscoreboard/issue约1.13/1.09。L2命中98.09/97.84%，L2throughput51.93/54.09%elapsed。不是HBM带宽饱和证据。
- reassess295：294两K128合组，60regs、local0、1.254M指令，barrier/issue降至.893/.890，但longscoreboard升至1.595/1.593，普通调用无改善。推迟释放slot可能把压力转到生产端；汇总计数器不证明唯一原因。
- reassess298：297减为256线程primary，32regs、local0、.535M指令；secondary60regs、local0。减少primary指令没有改善完整调用。
- profile299：cluster4+3 N64双CTA，124regs、local0、约.863M secondary指令，replay约15.2/15.1us，普通调用约16us。比282指令少仍显著更慢；不能将其归因于spill，也不能从最小profile断言全部是DSM。
- reassess303：302warp-leader readiness，62regs、local0、约1.423M指令；longscoreboard/issue下降，但wait/issue约1.76，完整调用更慢。优化单项stall不能替代完整计时。
- profile308：本地shared变换首个可运行版138regs、local0、5.223M指令；C7515警告明确WGMMA串行化。309展开K循环消除警告并降至18us，仍慢于282。
- profile306仅捕获旧304一个dtype的两kernel；profile311为错误phase候选的资源采集，不可作为有效性能结果。

上述replay时长仅诊断，未代替标准完整调用对照。stall数为相对issue_active的ratio，非总时间百分比。

## 试验结论

292仅primary leader触发PDL、293整叶合组、294两K合组、295单WG leader释放、296plane补齐256B、297primary256线程均未优于282。298生成分支编译失败；299修正后cluster双CTA4+3约16us；300只预取第一个K128原始块仍10.36FP。301primary256x512、302leader-only readiness、303取消postwait warp.sync、304非对称K128+K384均无普通调用收益。

305生成器变量覆盖失败，shell误继续，实际重复304；源码hash和配置验证这一点，明确排除为新架构证据。后续命令改为fail-fast。306CuTe不同类型共享auto声明编译失败；307 PTX寄存器命名编译失败，全部留档。

308–312为新的CTA-local shared transform架构：7x16FP32累加器跨K保存，原始四象限TMA到shared，CUDA在本地生成S/T，省掉独立primary和全局S/T。308约24us；309静态展开解除C7515约18us；310用相同swizzle下的物理线性offset做元素变换约16.4FP；311 K64双buffer raw wait错误使用q%2，输出严重错误，所有时间排除；312纠正为floor(q/2)%2恢复原近似误差，但约15.70FP仍显著更慢，不保留。

本地变换复制了每CTA的SIMT工作，减少全局流量和启动并没有抵消计算与同步成本。不能把它称为已验证赢家。

## 下一方向（未实施）

新PDL global split4+3：rank0算P1P2P6P7，rank1先算完P5P3P4再等U2/U3，消除299中等待U后再算两叶的尾链。去掉cluster初始化/终止barrier与DSM，用每warp独立global release/acquire flags保护vec4 U交换。primary依赖完成保证当次flag零初始化先于任何访问，不加全grid初始化barrier。6MiB+8192B workspace、N64/BK128七slot，预计rank1约百余寄存器，需实测、逐位验证、sanitizer和Graph/stream压力测试。

具体协议见proposals/global_split_proposal_next.md；只是设计假设，没有速度承诺。旧176使用608-worker合体pack/global发布，不能当作新PDL协议测量，但应作为反面历史审视。下一正式iteration314。

## 归档

iterations292_313.json含22次原始CSV和output哈希；source trajectory各自已提交。sha256.json记录诊断、ABBA、检查和生成器来源与哈希，大CSV用gzip。二进制ncu-rep和可执行文件留在results目录，未重复提交；复现脚本一并归档。
