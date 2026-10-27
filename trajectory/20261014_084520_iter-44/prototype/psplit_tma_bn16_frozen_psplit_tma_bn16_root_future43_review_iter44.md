本包是从实际 unchanged45（SHA3c84b310…）生成的私有 BN16 TMA future43 假设。只修改新prefix；没有编译、链接、GPU、Git、Plan注册、public48或formal44启动。formal43唯一owner试验47 BN64不受本包影响。

候选 BM64/BN16/BK64、两bank、384线程、min1。原rawA4 TMA/32768TX/SW128/1024对齐/current-A每call encode、SIMT B最终7槽onecast、all128 B writers、rawbar phase/full publication、wholeA3+4/whole7P、ascending K16/wait0/postfence/release、两bank tail drain、原三CTA epochs、fixed U/RN alpha/beta FMA均保持。20个有序替换逐步SHA验证，正向生成和逆向精确恢复完整45父源码。

N16的CuTE RS m64n16k16 atom每product每线程8个FP32 P；WG0 P24+A48、WG1 P32+A64。两个consumer均采用128源码quota，保留共同INC，位于分支和P clear之前；producer DEC40。请求池128×(40+128+128)=37888 words，owned initial pool的算术ceil要求99 words/thread；真实compiler allocation、粒度、ownership、初始状态到DEC40/INC128是否合法均需fresh machine gate，不能继承旧45/42的I或GPR。

两份当前quota请求需75776>65536 words。本包严格min1，没有第二candidate。虽然预测两份shared95232×2=190464B较小，也不能据此宣称2CTA。未来min2须独立修改/证明quota、startup owned pool和lifetime；当前最高GPR、spill、实际CAN和residency全部未知。

B原Layout_MN_SW32_Atom模板的16×64×7 halfword布局为 `q*1024+(k//8)*128+n%16+16*(k%8)`；绝对byte地址 `sw32(O+32768+2*raw)`，`O=origin+bank*47104`，`sw32(x)=x^((x&128)>>3)`。每槽2048B，每K16 atom512B，descriptor start为 `O+32768+q*2048+atom*512`。按pinned CuTE complement/coalesce，N recast shape2被W2完全分割后，residual shape1 coalesce成stride0，canonical N `(2,1):(1,0)`；leading源码字段0、stride256B/encoded16、layout type3/base offset0。leading对于extent1不被坐标索引，该字段负例检验源码一致性，不宣称任意非零unused leading会导致N16计算错误。实际compiled descriptor仍待fresh验证。

CLayout_64x16的128×8坐标完整覆盖64×16，A仍128×4atoms×8 coordinates覆盖64×64。预期A32768B、B14336B、每bank47104B、两bankpayload94208B；Pipeline barrier32B在94208，raw_done16B在94240，align1024后logical SharedStorage95232B。7168个tail words/28672B在原两bank typed union内，双方retire后才激活。全部sizeof/offsetof/Driver shared actual保持null。

Tail stride从product2048/row32改成product1024/row16，原xor_mask截取低4位`&15`以维持父源码“within each row”的明确合同。未裁剪旧mask也可能通过交换相邻行保持全tile双射，不能称其必然alias；此处裁剪是已声明的tail地址适配，U数学不变。模型分别检查行内范围、全tile唯一性和全部7168 word覆盖。

1024 grid从父45的8×16/128 CTA变为8×32/256 CTA。同一rawA复用范围减半、A transport/transform重复增加，P/B/tail per CTA减少。更多CTA是否改善端到端延迟必须实际测量；不能把source copy计数当HBM traffic、bank conflict、occupancy或residency。

实际CPU模型执行exit0：五绝对origin×两banks、opaque vector8和完整slot/bank双射、独立descriptor重构、A/C fragments、tail words、padded/极限A stride、3种D stride共3145728唯一输出word、6561整数DAG和160wholeK cases。40 interleaved schedules检验32768TX/current phase/128 writer publication/wholeA48/64和P24/32 pending/wait0/bothbank release/tail/U；42个负例均被拒绝。

一次CPU authoring shell exit1已保留真实combined stdout/stderr及exit：宽泛的substring240残留quota断言误匹配预期raw_done94240，model.py未写成，后续wrapper读文件失败。该失败发生在模型进程启动之前，actual model process exit为null；不伪装成CUDA/GPU或model进程失败。修成精确quota token后assembly exit0，model首次真实进程exit0；源码独立验证三次exit0（第三次补齐旧46 CUDA保护域）。candidate/API没有随模型authoring改动。

此前BN64冻结模型仅被读取作为新prefix CPU模型模板，没有执行或改写旧模型。旧142future、42个旧library/source（40个旧CUDA加internal.h/on_gpu.sh）和10个pinned headers逐项保护；formal43 owner六个接口和新47源码是明确可变观察域。线程槽满，未能新启动独立subagent；ROOT将在冻结后独立只读验收几何/逆向证明。本包完整actualCPU日志供复核。

protocol.json要求future fresh compiled/PTX/SASS/cubin/GPR/pool/init/scalarization/wholeA+P lifetime/ABI/zeroSTACKLOCALspill/Driver/SAN，再保留原始FP64门限运行完整动态与ordinary/Graph独立性能轴。CPU整数模型不代表GPU FP16/BF16精度。本包所有actual CUDA、CAN、Driver、GPR、shared ABI、SAN、precision、latency和相对cuBLAS加速比均为null。
