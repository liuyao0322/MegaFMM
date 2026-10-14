from pathlib import Path
import hashlib,json
p=Path('solution/megakernel_wgmma_psplit_tma_inplace_ss.cu');old=p.read_text();s=old;steps=[]
h=lambda x:hashlib.sha256(x.encode()).hexdigest()
def rep(a,b,n=1):
 global s
 assert s.count(a)==n,(a[:40],s.count(a),n)
 before=h(s);s=s.replace(a,b);steps.append({'old':a,'new':b,'count':n,'before_sha256':before,'after_sha256':h(s)})
a=s.index('  float s1[2],s2[2],s3[2],s4[2];');b=s.index('\n}\ntemplate<class Element,int Cols,class SmemB>',a)
rep(s[a:b],'''  float s1[2],s2[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j) {
    const float a11=psplit_tma_inplace_ss_unpack_word<Element>(raw.x,j);
    const float a21=psplit_tma_inplace_ss_unpack_word<Element>(raw.z,j);
    const float a22=psplit_tma_inplace_ss_unpack_word<Element>(raw.w,j);
    s1[j]=__fadd_rn(a21,a22);
    s2[j]=__fsub_rn(s1[j],a11);
  }
  // Final stores consume one original RN result each; unrounded S2 feeds S4.
  psplit_tma_inplace_ss_store_pair(&a(row,k,4),psplit_tma_inplace_ss_round_pair<Element>(s1[0],s1[1]));
  float s4[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j)
    s4[j]=__fsub_rn(psplit_tma_inplace_ss_unpack_word<Element>(raw.y,j),s2[j]);
  psplit_tma_inplace_ss_store_pair(&a(row,k,5),psplit_tma_inplace_ss_round_pair<Element>(s2[0],s2[1]));
  psplit_tma_inplace_ss_store_pair(&a(row,k,2),psplit_tma_inplace_ss_round_pair<Element>(s4[0],s4[1]));
  float s3[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j)
    s3[j]=__fsub_rn(psplit_tma_inplace_ss_unpack_word<Element>(raw.x,j),psplit_tma_inplace_ss_unpack_word<Element>(raw.z,j));
  psplit_tma_inplace_ss_store_pair(&a(row,k,6),psplit_tma_inplace_ss_round_pair<Element>(s3[0],s3[1]));''')
a=s.index('  float t1[2],t2[2],t3[2],t4[2];');b=s.index('\n}\ntemplate<class Element,int Cols,class SmemA',a)
rep(s[a:b],'''  float t1[2],t2[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j) {
    const float b11=psplit_tma_inplace_ss_unpack_word<Element>(raw.x,j);
    const float b12=psplit_tma_inplace_ss_unpack_word<Element>(raw.y,j);
    const float b22=psplit_tma_inplace_ss_unpack_word<Element>(raw.w,j);
    t1[j]=__fsub_rn(b12,b11);
    t2[j]=__fsub_rn(b22,t1[j]);
  }
  psplit_tma_inplace_ss_store_pair(&b(col,k,4),psplit_tma_inplace_ss_round_pair<Element>(t1[0],t1[1]));
  float t4[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j)
    t4[j]=__fsub_rn(t2[j],psplit_tma_inplace_ss_unpack_word<Element>(raw.z,j));
  psplit_tma_inplace_ss_store_pair(&b(col,k,5),psplit_tma_inplace_ss_round_pair<Element>(t2[0],t2[1]));
  psplit_tma_inplace_ss_store_pair(&b(col,k,3),psplit_tma_inplace_ss_round_pair<Element>(t4[0],t4[1]));
  float t3[2];
  CUTE_UNROLL
  for(int j=0;j<2;++j)
    t3[j]=__fsub_rn(psplit_tma_inplace_ss_unpack_word<Element>(raw.w,j),psplit_tma_inplace_ss_unpack_word<Element>(raw.y,j));
  psplit_tma_inplace_ss_store_pair(&b(col,k,6),psplit_tma_inplace_ss_round_pair<Element>(t3[0],t3[1]));''')
rep('warpgroup_reg_dealloc<48>()','warpgroup_reg_dealloc<(Cols==32 ? 40 : 48)>()')
rep('constexpr int WG0Quota=Cols==32 ? 88 : 64;','constexpr int WG0Quota=Cols==32 ? 96 : 80;')
rep('constexpr int WG1Quota=Cols==32 ? 104 : 72;','constexpr int WG1Quota=Cols==32 ? 104 : 80;')
rev=s
for step in reversed(steps):
 assert rev.count(step['new'])==step['count'];rev=rev.replace(step['new'],step['old']);assert h(rev)==step['before_sha256']
assert rev==old
out=Path('results/psplit_tma_inplace_ss_revision2_source_iter47.cu');assert not out.exists();out.write_text(s)
Path('results/psplit_tma_inplace_ss_revision2_source_delta_iter47.json').write_text(json.dumps({'old_source_path':'results/psplit_tma_inplace_ss_revision1_source_iter47.cu','old_sha256':h(old),'new_source_path':str(out),'new_sha256':h(s),'steps':steps,'inverse_byte_exact':True,'scope':'Original RN DAG and onecast operands preserved; independent S3/T3 scheduled after S4/T4; fresh sourcefit quotas based initial actual role37 and effective startupI80; not resource certification'},indent=2)+'\n')
p.write_text(s)
keep=Path('.scratch/psplit_tma_inplace_ss_nvcc_keep_iter47');keep.rename('.scratch/psplit_tma_inplace_ss_nvcc_keep_failed_revision1_iter47');keep.mkdir()
r=Path('results/psplit_tma_inplace_ss_revision1_build_runner_iter47.py').read_text().replace('revision1','revision2')
Path('results/psplit_tma_inplace_ss_revision2_build_runner_iter47.py').write_text(r)
Path('results/psplit_tma_inplace_ss_revision2_owner_iter47.py').write_bytes(Path(__file__).read_bytes())
print(h(s))
