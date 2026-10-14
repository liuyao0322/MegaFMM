from pathlib import Path
import subprocess,re,json,hashlib,sys
cuo='/share/app/cuda/cuda-12.4/bin/cuobjdump'
header='Actual CUDA12.4 BK32 BN32/BN16 eight rawAB TMA inplace SS four instances; complete address/predicate/instruction streams; staticcounts notdynamicexecution.\n'
objblocks=[];resources=[];elf=[];identity={}
for stem in ['megakernel_wgmma_psplit_tma_inplace_ss']:
 obj=f'build/CMakeFiles/megafmm.dir/solution/{stem}.cu.o'
 raw=subprocess.check_output([cuo,'--dump-sass',obj],text=True)
 blocks=[b for b in re.split(r'(?m)^\s*Function : ',raw)[1:] if 'winograd_wgmma_psplit_tma_inplace_ss' in b.splitlines()[0]];assert len(blocks)==4
 objblocks+=blocks;resources.append(subprocess.check_output([cuo,'--dump-resource-usage',obj],text=True));elf.append(subprocess.check_output([cuo,'--dump-elf',obj],text=True))
 identity[obj]=hashlib.sha256(Path(obj).read_bytes()).hexdigest()
 for suffix in ['ptx','cubin']:
  p=Path('.scratch/psplit_tma_inplace_ss_nvcc_keep_iter47',stem+'.'+suffix);assert p.exists(),p
  out=Path('results',f'psplit_tma_inplace_ss_revision2_{stem}_iter47.{suffix}');out.write_bytes(p.read_bytes());identity[str(out)]=hashlib.sha256(out.read_bytes()).hexdigest()
Path('results/psplit_tma_inplace_ss_revision2_object_sass_iter47.txt').write_text(header+''.join('\nFunction : '+b for b in objblocks))
Path('results/psplit_tma_inplace_ss_revision2_object_resources_iter47.txt').write_text(''.join(resources))
Path('results/psplit_tma_inplace_ss_revision2_object_elf_iter47.txt').write_text(''.join(elf))
Path('results/small_psplit_tma_inplace_ss_revision2_resources_iter47.txt').write_bytes(subprocess.check_output([cuo,'--dump-resource-usage','build/megafmm_bench']))
Path('results/small_psplit_tma_inplace_ss_revision2_cuobjdump_iter47.txt').write_bytes(subprocess.check_output([cuo,'--version']))
symbols=[b.splitlines()[0].strip() for b in objblocks]
def streams(bs):return [(b.splitlines()[0].strip(),re.findall(r'/\*([0-9a-f]+)\*/\s+(.+?);',b)) for b in bs]
targets = [('bench','build/megafmm_bench','results/small_psplit_tma_inplace_ss_revision2_sass_iter47.txt'),('test','build/test_correctness','results/psplit_tma_inplace_ss_revision2_test_sass_iter47.txt')]
if '--bench-only' in sys.argv: targets = targets[:1]
for label,binary,out in targets:
 raw=subprocess.check_output([cuo,'--dump-sass','--function',','.join(symbols),binary],text=True,stderr=subprocess.DEVNULL)
 linked=[b for b in re.split(r'(?m)^\s*Function : ',raw)[1:] if 'winograd_wgmma_psplit_tma_inplace_ss' in b.splitlines()[0]];assert len(linked)==4
 assert dict(streams(objblocks))==dict(streams(linked)),label
 Path(out).write_text(header+''.join('\nFunction : '+b for b in linked))
 identity[binary]=hashlib.sha256(Path(binary).read_bytes()).hexdigest()
 identity[label+'_instructions']={k:len(v) for k,v in streams(linked)}
 identity[label+'_raw_sass_sha256']=hashlib.sha256(Path(out).read_bytes()).hexdigest()
identity['status']='PASS_COMPLETE_NEW51_52_FOUR_INSTANCE_INSTRUCTION_STREAMS_STATIC_LAUNCH_GATE_PENDING'
identity['scope']='Full address/predicate/instructions; rawheaderflags differences preserved, notfulltextclaim. PTX/cubins were preserved by --keep in actual CMake object compilation, nosecond PTX compile.'
Path('results/psplit_tma_inplace_ss_revision2_bench_link_identity_iter47.json' if '--bench-only' in sys.argv else 'results/psplit_tma_inplace_ss_revision2_link_identity_iter47.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps({'status':identity['status'],'object':identity[obj],'bench':identity.get('build/megafmm_bench'),'test':identity.get('build/test_correctness'),'instruction_counts':identity.get('bench_instructions')},indent=2))
