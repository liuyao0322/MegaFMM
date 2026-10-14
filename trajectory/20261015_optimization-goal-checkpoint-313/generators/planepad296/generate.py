from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
s=s.replace('constexpr size_t WorkspaceBytes=size_t(8)*Plane*2;','constexpr int PlaneStorage=Plane+128; //256-byte padding per transformed plane\nconstexpr size_t WorkspaceBytes=size_t(8)*PlaneStorage*2;')
s=s.replace('[(j*Plane+i)/2]=word','[(j*PlaneStorage+i)/2]=word')
s=s.replace('128,512ULL*512*2','128,uint64_t(PlaneStorage)*2')
s=s.replace('uint64_t strides[2]={dims[0]*2,dims[0]*dims[1]*2};','uint64_t strides[2]={dims[0]*2,transformed ? uint64_t(PlaneStorage)*2:dims[0]*dims[1]*2};')
assert 'j*Plane+i' not in s
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('impl->bytes=size_t(4)<<20;','impl->bytes=(size_t(4)<<20)+2048;').replace('workspace=4194304','workspace=4196352;transformed_plane_padding_bytes=256')
Path('solution/plan.cpp').write_text(p)
