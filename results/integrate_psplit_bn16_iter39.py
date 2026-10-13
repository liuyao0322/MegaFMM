from pathlib import Path
import hashlib,json,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base='d0ea861605612d0e4d95e2e6596ff352a2d264dd';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==base
manifests=[Path('.scratch/psplit_bn16_future38_manifest.json'),Path('.scratch/psplit_bn16_ss_future38_manifest.json'),Path('.scratch/psplit_bn16_tailbanks_future38_manifest.json')]
for m in manifests:
 j=json.loads(m.read_text())
 for f,h in j['files'].items():assert sha(f)==h;src=Path(f);out=Path('results',f'psplit_bn16_frozen_{src.stem}_iter39{src.suffix}');assert not out.exists();out.write_bytes(src.read_bytes())
 out=Path('results',f'psplit_bn16_frozen_{m.stem}_iter39{m.suffix}');assert not out.exists();out.write_bytes(m.read_bytes())
proof=[]
for kind,prefix,oldtrait,newtrait,oldname,newname,expected in [('raw','rsa','PSplitBN16Future38Traits','PSplitRSABN16Traits','psplit_bn16_future38','psplit_rsa_bn16','015f6ca799af720a859ffdf73db2655de3f927bb31e3e0bf5751832ea4d8353d'),('ss','ss','PSplitBN16SSFuture38Traits','PSplitSSBN16Traits','psplit_bn16_ss_future38','psplit_ss_bn16','ed8123e87b931ab202241f170ff32fc666678d3a7a69481193de4b90f3ad8b86')]:
 src=Path(f'.scratch/psplit_bn16_tailbanks_future38_{kind}_candidate.cu');assert sha(src)==expected
 s=src.read_text().replace(oldtrait,newtrait).replace(oldname,newname)
 s=s.replace('// CPU-only future38 BN16 source; uncompiled and not registered in any Plan.','// Native Mega42 RS-A BN16 P3+4; actual compiled launch gate required.').replace('// CPU-only future38 SS BN16 companion; uncompiled and not registered.','// Native Mega43 SS BN16 P3+4; actual compiled launch gate required.')
 out=Path(f'solution/megakernel_wgmma_psplit_{prefix}_bn16.cu');assert not out.exists();out.write_text(s)
 inverse=s.replace(newtrait,oldtrait).replace(newname,oldname).splitlines(keepends=True);inverse[0]=src.read_text().splitlines(keepends=True)[0];assert ''.join(inverse)==src.read_text()
 proof.append({'source':str(out),'sha256':sha(out),'prototype':str(src),'prototype_sha256':expected,'inverse_rename_and_first_comment_exact':True})
p=Path('solution/plan.cpp');s=p.read_text()
for mode,key in [('plan','run'),('policy','prepare')]:
 marker=f'    if ({mode}.policy.mega_variant == 41) {{' if mode=='plan' else '    if (policy.mega_variant == 41) {'
 at=s.index(marker);end=s.index('    } else if (',at)+5;block=s[at:end]
 pre=block.replace('== 41','== 43').replace('psplit_hybrid_s','psplit_ss_bn16')+' else '+block.lstrip().replace('== 41','== 42').replace('psplit_hybrid_s','psplit_rsa_bn16')+' else '
 s=s[:at]+pre+block.lstrip()+s[end:]
marker='      if (impl_->policy.mega_variant == 41) {';at=s.index(marker);end=s.index('\n      }',at)+8;block=s[at:end]
s=s[:at]+block.replace('== 41','== 43').replace('psplit_hybrid_s','psplit_ss_bn16').replace('_64x32','_64x16')+'\n\n'+block.replace('== 41','== 42').replace('psplit_hybrid_s','psplit_rsa_bn16').replace('_64x32','_64x16')+'\n\n'+s[at:]
s=s.replace('policy.mega_variant > 41','policy.mega_variant > 43').replace('mega_variant must be between 0 and 41','mega_variant must be between 0 and 43')
marker='    const bool supported = policy.mega_variant == 41\n                             ? (wgmma_psplit_hybrid_s_can_implement(p, 0) || packed_mma_can_implement(p, 7))'
assert marker in s
s=s.replace(marker,marker.replace('== 41','== 43').replace('psplit_hybrid_s','psplit_ss_bn16')+'\n                           : policy.mega_variant == 42\n                             ? (wgmma_psplit_rsa_bn16_can_implement(p, 0) || packed_mma_can_implement(p, 7))\n                           : policy.mega_variant == 41\n                             ? (wgmma_psplit_hybrid_s_can_implement(p, 0) || packed_mma_can_implement(p, 7))')
p.write_text(s)
p=Path('include/megafmm/kernels.h');s=p.read_text();at=s.index('// Native hybrid S RS-A');end=s.index('\n\n// Native RS-A P3+4 BK32:',at);block=s[at:end]
s=s[:at]+block.replace('// Native hybrid S RS-A P3+4 BK32: config0 maps to41, BN32, parentK%64 and currentAB16.','// Native SS BN16 P3+4 BK32: config0 maps to43, parentK%64 and currentAB16.').replace('psplit_hybrid_s','psplit_ss_bn16')+'\n\n'+block.replace('// Native hybrid S RS-A P3+4 BK32: config0 maps to41, BN32, parentK%64 and currentAB16.','// Native RS-A BN16 P3+4 BK32: config0 maps to42, parentK%64 and currentAB16.').replace('psplit_hybrid_s','psplit_rsa_bn16')+'\n\n'+s[at:];p.write_text(s)
p=Path('include/megafmm/gemm.h');s=p.read_text().replace('  int mega_variant=0;','  // 42=RS-A BN16 BK32 P3+4, rawA4/finalB7; initial quotas48/96/96 and minCTAs2.\n  // 43=SS BN16 BK32 P3+4, finalA7/finalB7; initial quotas64/80/80 and minCTAs2.\n  // WholeK/fixedU/onecast and currentAB16/parentK%64; actual fresh gate required.\n  int mega_variant=0;');p.write_text(s)
p=Path('bench/benchmark.cu');s=p.read_text().replace('0..41','0..43').replace('o.mega_variant > 41','o.mega_variant > 43').replace('between 0 and 41','between 0 and 43');p.write_text(s)
p=Path('CMakeLists.txt');s=p.read_text().replace('solution/megakernel_wgmma_psplit_hybrid_s.cu)','solution/megakernel_wgmma_psplit_hybrid_s.cu solution/megakernel_wgmma_psplit_rsa_bn16.cu solution/megakernel_wgmma_psplit_ss_bn16.cu)');s+='\n'
for name in ['rsa','ss']:
 d=Path(f'.scratch/psplit_bn16_{name}_nvcc_keep_iter39');d.mkdir(exist_ok=True)
 s+=f'set_source_files_properties(solution/megakernel_wgmma_psplit_{name}_bn16.cu\n  PROPERTIES COMPILE_OPTIONS "--keep;--keep-dir=${{CMAKE_CURRENT_SOURCE_DIR}}/{d}")\n'
p.write_text(s)
Path('results/psplit_bn16_source_normalization_iter39.json').write_text(json.dumps({'schema':'megafmm.iter39.bn16-source-normalization.v1','base_checkpoint':base,'new_sources':proof,'protected_manifests':{str(m):sha(m) for m in manifests}},indent=2)+'\n')
oldcu={f:hashlib.sha256(subprocess.check_output(['git','show',base+':'+f])).hexdigest() for f in subprocess.check_output(['git','ls-tree','-r','--name-only',base,'solution'],text=True).splitlines() if f.endswith('.cu')};assert len(oldcu)==35
for f,h in oldcu.items():assert sha(f)==h
Path('.scratch/psplit_bn16_old_cuda_iter39.json').write_text(json.dumps(oldcu,indent=2)+'\n')
print('INTEGRATION_READY',proof)
