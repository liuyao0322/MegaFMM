# Temporary new-prefix assembly fragment; frozen as provenance, never imported.
def geometry():
 origins=[0,1024,2048,8192,65536]
 for o in origins:
  for bank in range(2):
   lo=o+bank*47104;seen=set();image=bytearray(47104)
   for q in range(4):
    for r in range(64):
     for k in range(0,64,8):
      addr=a_at(o,bank,r,k,q);assert addr%16==0
      words=[(raw_a(r,k+j,q)*131+bank*16001+(o//1024)*1009)&65535 for j in range(8)]
      image[addr-lo:addr-lo+16]=struct.pack('<8H',*words)
      for j in range(8):
       a=a_at(o,bank,r,k+j,q);assert a==addr+j*2==ref_a(o,bank,r,k+j,q)
       assert lo<=a<lo+32768 and a not in seen;seen.add(a)
      assert struct.unpack_from('<8H',image,addr-lo)==tuple(words);add('A_opaque_halfwords_roundtrip',8);add('A_vector8')
   for q in range(7):
    for k in range(64):
     for n in range(0,16,8):
      addr=b_at(o,bank,n,k,q);assert addr%16==0
      words=[(raw_b(n+j,k,q)*137+q*71+bank*19001+(o//1024)*1013)&65535 for j in range(8)]
      image[addr-lo:addr-lo+16]=struct.pack('<8H',*words)
      for j in range(8):
       a=b_at(o,bank,n+j,k,q);assert a==addr+j*2==ref_b(o,bank,n+j,k,q)
       assert lo+32768<=a<lo+47104 and a not in seen;seen.add(a)
      assert struct.unpack_from('<8H',image,addr-lo)==tuple(words);add('B_opaque_halfwords_roundtrip',8);add('B_vector8')
   assert seen==set(range(lo,lo+47104,2));add('full_bank_bijections')
   for q in range(7):
    for atom in range(4):
     start=lo+32768+q*2048+atom*512;assert start%512==0
     # Pinned CuTE logical_divide N mode: complement(2:1,target2)->1:0.
     # Consequently canonical (2,1):(1,0) leading=0; extent1 never indexes it.
     canonical_n_shape=(2,1);canonical_n_stride=(1,0)
     canonical_k_shape=(8,2);canonical_k_stride=(2,16)
     leading=canonical_n_stride[1];stride=canonical_k_stride[1]
     assert leading==0 and stride==16
     desc=(start>>4)|(leading<<16)|(stride<<32)|(3<<62)
     assert desc&0x3fff==start>>4
     assert (desc>>16)&0x3fff==0 and (desc>>32)&0x3fff==16 and desc>>62==3
     for n in range(16):
      for ka in range(16):
       reconstructed=sw(start+(n%16)*2+(ka//8)*256+(ka%8)*32,1)
       assert reconstructed==b_at(o,bank,n,16*atom+ka,q);add('B_descriptor_coordinates')
     add('B_descriptor_atoms')
 assert 94208+32==94240 and 94240+16==94256
 assert (94256+1023)//1024*1024==95232
 assert 7*64*16*4==28672<47104<94208
 assert 28672<=94208<94240<95232
 cseen={};aseen=set()
 for t in range(128):
  for v in range(8):
   r,c=ccoord(t,v);assert (r,c) not in cseen and 0<=r<64 and 0<=c<16;cseen[r,c]=(t,v);add('C_fragment_coordinates')
  for atom in range(4):
   for v in range(8):
    x=128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4%2)
    r,k=x%64,x//64+16*atom;assert (r,k) not in aseen;aseen.add((r,k));add('A_RS_fragment_coordinates')
 assert len(cseen)==1024 and len(aseen)==4096
 words=set()
 for q in range(7):
  for r,c in cseen:
   a=tail(q,r,c);assert q*1024+r*16<=a<q*1024+(r+1)*16
   assert a not in words;words.add(a);add('tail_words')
 assert words==set(range(7168))
 for ldd in [1024,1025,1032]:
  out=set()
  for bx in range(8):
   for by in range(32):
    for linear in range(1024):
     owner=linear%384;r,c=divmod(linear,16);assert 0<=owner<384
     rr=bx*64+r;cc=by*16+c
     for dr,dc in [(0,0),(0,512),(512,0),(512,512)]:
      address=(rr+dr)*ldd+cc+dc;assert address not in out;out.add(address);add('D_unique_owned_words')
  assert len(out)==1024*1024
  for r in range(1024):
   assert r*ldd+1023 in out and r*ldd in out
   for col in range(1024,ldd):assert r*ldd+col not in out;add('D_padding_words_untouched')
 for lda,ldb in [(1024,1024),(1032,1040),(1040,1032),((1<<39)-8,1024)]:
  for bx in range(8):
   for by in range(32):
    for stage in range(8):
     rb,cb,kb=bx*64,by*16,stage*64
     for q in range(4):
      mh,kh=(q//2)*512,(q%2)*512
      for r,k in [(0,0),(0,56),(7,8),(8,16),(63,56)]:
       row=rb+mh+r;col=kb+kh+k;address=(row*lda+col)*2
       assert 0<=row<1024 and 0<=col<=1016 and address%16==0
       assert address+16<=(1023*lda+1024)*2;add('A_global_vector_offset_checks')
     for task in range(128):
      k,col=divmod(task,2);col*=8;gk,gc=kb+k,cb+col
      for dr,dc in [(0,0),(0,512),(512,0),(512,512)]:
       row,cc=gk+dr,gc+dc;address=(row*ldb+cc)*2
       assert 0<=row<1024 and 0<=cc<=1016 and address%16==0
       assert address+16<=(1023*ldb+1024)*2;add('B_global_vector_offset_checks')
 return {'origins':origins,'bank_bytes':47104,'two_bank_payload_bytes':94208,'expected_pipeline_offset':94208,'expected_raw_done_offset':94240,'expected_logical_shared_bytes':95232,'tail_bytes':28672,'B_GMMA_descriptor_prediction':{'leading_bytes':0,'stride_bytes':256,'layout_type':3,'base_offset':0,'N_residual_extent1_leading_field_unused':True},'actual_sizeof_and_offsets':None}
