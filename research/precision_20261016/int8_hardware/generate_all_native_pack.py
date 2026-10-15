"""Prepared P49 generator; apply only as a new separately measured candidate."""
from pathlib import Path
r=Path(__file__).resolve().parent
for name in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(r/'p47'/name).read_text()
 if name=='pack.cuh':
  f=s.index('void pack_pair(')
  a=s.index('   if constexpr(Group==2){',f)
  b=s.index('  }\n };\n if constexpr(Balance)',a)
  stores='''   direct_native_store<T,0,8>(a,target,high,q);
   direct_native_store<T,1,8>(SideA?b:cc,target,high,q);
   direct_native_store<T,(SideA?3:2),8>(d,target,high,q);
   if constexpr(SideA){
    float s1=__fadd_rn(cc,d),s2=__fsub_rn(s1,a);
    direct_native_store<T,4,0>(s1,target,high,q);direct_native_store<T,5,1>(s2,target,high,q);
    direct_native_store<T,6,2>(__fsub_rn(a,cc),target,high,q);direct_native_store<T,2,3>(__fsub_rn(b,s2),target,high,q);
   }else{
    float t1=__fsub_rn(b,a),t2=__fsub_rn(d,t1);
    direct_native_store<T,4,0>(t1,target,high,q);direct_native_store<T,5,1>(t2,target,high,q);
    direct_native_store<T,6,2>(__fsub_rn(d,b),target,high,q);direct_native_store<T,3,3>(__fsub_rn(t2,cc),target,high,q);
   }
'''
  s=s[:a]+stores+s[b:]
  a=s.index('\n constexpr int Rows=',f)
  s=s[:a]+'\n // P49 Group0 writes all planes; all collective barriers precede this helper.\n if constexpr(Group!=0)return;'+s[a:]
 (r/name).write_text(s)
