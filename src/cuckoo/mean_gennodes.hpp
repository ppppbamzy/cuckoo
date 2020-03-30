#ifndef MEAN_GENNODES_HPP
#define MEAN_GENNODES_HPP

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <x86intrin.h>
#include <assert.h>
#include <vector>
#include <bitset>
#include "mean_params.hpp"
#include "mean_trimedges.hpp"
#include "../crypto/siphashxN.h"
#include "../threads/barrier.hpp"

void edgetrimmer::genUnodes(const u32 id, const u32 uorv) {
    u64 rdtsc0, rdtsc1;
#ifdef NEEDSYNC
    u32 last[NX];;
#endif
  
    rdtsc0 = __rdtsc();
    u8 const *base = (u8 *)buckets;
    indexer<ZBUCKETSIZE> dst;
    const u32 starty = NY *  id    / nthreads;
    const u32   endy = NY * (id+1) / nthreads;
    u32 edge = starty << YZBITS, endedge = edge + NYZ;
#if NSIPHASH == 4
    const __m128i vxmask = _mm_set1_epi64x(XMASK);
    const __m128i vyzmask = _mm_set1_epi64x(YZMASK);
    __m128i v0, v1, v2, v3, v4, v5, v6, v7;
    const u32 e2 = 2 * edge + uorv;
    __m128i vpacket0 = _mm_set_epi64x(e2+2, e2+0);
    __m128i vpacket1 = _mm_set_epi64x(e2+6, e2+4);
    const __m128i vpacketinc = _mm_set1_epi64x(8);
    u64 e1 = edge;
    __m128i vhi0 = _mm_set_epi64x((e1+1)<<YZBITS, (e1+0)<<YZBITS);
    __m128i vhi1 = _mm_set_epi64x((e1+3)<<YZBITS, (e1+2)<<YZBITS);
    const __m128i vhiinc = _mm_set1_epi64x(4<<YZBITS);
#elif NSIPHASH == 8
    const __m256i vxmask = _mm256_set1_epi64x(XMASK);
    const __m256i vyzmask = _mm256_set1_epi64x(YZMASK);
    const __m256i vinit = _mm256_load_si256((__m256i *)&sip_keys);
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    const u32 e2 = 2 * edge + uorv;
    __m256i vpacket0 = _mm256_set_epi64x(e2+6, e2+4, e2+2, e2+0);
    __m256i vpacket1 = _mm256_set_epi64x(e2+14, e2+12, e2+10, e2+8);
    const __m256i vpacketinc = _mm256_set1_epi64x(16);
    u64 e1 = edge;
    __m256i vhi0 = _mm256_set_epi64x((e1+3)<<YZBITS, (e1+2)<<YZBITS, (e1+1)<<YZBITS, (e1+0)<<YZBITS);
    __m256i vhi1 = _mm256_set_epi64x((e1+7)<<YZBITS, (e1+6)<<YZBITS, (e1+5)<<YZBITS, (e1+4)<<YZBITS);
    const __m256i vhiinc = _mm256_set1_epi64x(8<<YZBITS);
#endif
    offset_t sumsize = 0;
    for (u32 my = starty; my < endy; my++, endedge += NYZ) {
      dst.matrixv(my);
#ifdef NEEDSYNC
      for (u32 x=0; x < NX; x++)
        last[x] = edge;
#endif
      for (; edge < endedge; edge += NSIPHASH) {
// bit        28..21     20..13    12..0
// node       XXXXXX     YYYYYY    ZZZZZ
#if NSIPHASH == 1
        const u32 node = sipnode(&sip_keys, edge, uorv);
        const u32 ux = node >> YZBITS;
        const BIGTYPE0 zz = (BIGTYPE0)edge << YZBITS | (node & YZMASK);
#ifndef NEEDSYNC
// bit        39..21     20..13    12..0
// write        edge     YYYYYY    ZZZZZ
        *(BIGTYPE0 *)(base+dst.index[ux]) = zz;
        dst.index[ux] += BIGSIZE0;
#else
        if (zz) {
          for (; unlikely(last[ux] + NNONYZ <= edge); last[ux] += NNONYZ, dst.index[ux] += BIGSIZE0)
            *(u32 *)(base+dst.index[ux]) = 0;
          *(u32 *)(base+dst.index[ux]) = zz;
          dst.index[ux] += BIGSIZE0;
          last[ux] = edge;
        }
#endif
#elif NSIPHASH == 4
        v7 = v3 = _mm_set1_epi64x(sip_keys.k3);
        v4 = v0 = _mm_set1_epi64x(sip_keys.k0);
        v5 = v1 = _mm_set1_epi64x(sip_keys.k1);
        v6 = v2 = _mm_set1_epi64x(sip_keys.k2);

        v3 = XOR(v3,vpacket0); v7 = XOR(v7,vpacket1);
        SIPROUNDX2N; SIPROUNDX2N;
        v0 = XOR(v0,vpacket0); v4 = XOR(v4,vpacket1);
        v2 = XOR(v2, _mm_set1_epi64x(0xffLL));
        v6 = XOR(v6, _mm_set1_epi64x(0xffLL));
        SIPROUNDX2N; SIPROUNDX2N; SIPROUNDX2N; SIPROUNDX2N;
        v0 = XOR(XOR(v0,v1),XOR(v2,v3));
        v4 = XOR(XOR(v4,v5),XOR(v6,v7));

        vpacket0 = _mm_add_epi64(vpacket0, vpacketinc);
        vpacket1 = _mm_add_epi64(vpacket1, vpacketinc);
        v1 = _mm_srli_epi64(v0, YZBITS) & vxmask;
        v5 = _mm_srli_epi64(v4, YZBITS) & vxmask;
        v0 = (v0 & vyzmask) | vhi0;
        v4 = (v4 & vyzmask) | vhi1;
        vhi0 = _mm_add_epi64(vhi0, vhiinc);
        vhi1 = _mm_add_epi64(vhi1, vhiinc);

        u32 ux;
#ifndef __SSE41__
#define extract32(x, imm) _mm_cvtsi128_si32(_mm_srli_si128((x), 4 * (imm)))
#else
#define extract32(x, imm) _mm_extract_epi32(x, imm)
#endif
#ifndef NEEDSYNC
#define STORE0(i,v,x,w) \
  ux = extract32(v,x);\
  *(u64 *)(base+dst.index[ux]) = _mm_extract_epi64(w,i%2);\
  dst.index[ux] += BIGSIZE0;
#else
  u32 zz;
#define STORE0(i,v,x,w) \
  zz = extract32(w,x);\
  if (i || likely(zz)) {\
    ux = extract32(v,x);\
    for (; unlikely(last[ux] + NNONYZ <= edge+i); last[ux] += NNONYZ, dst.index[ux] += BIGSIZE0)\
      *(u32 *)(base+dst.index[ux]) = 0;\
    *(u32 *)(base+dst.index[ux]) = zz;\
    dst.index[ux] += BIGSIZE0;\
    last[ux] = edge+i;\
  }
#endif
        STORE0(0,v1,0,v0); STORE0(1,v1,2,v0);
        STORE0(2,v5,0,v4); STORE0(3,v5,2,v4);
#elif NSIPHASH == 8
        v7 = v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
        v4 = v0 = _mm256_permute4x64_epi64(vinit, 0x00);
        v5 = v1 = _mm256_permute4x64_epi64(vinit, 0x55);
        v6 = v2 = _mm256_permute4x64_epi64(vinit, 0xAA);

        v3 = XOR(v3,vpacket0); v7 = XOR(v7,vpacket1);
        SIPROUNDX2N; SIPROUNDX2N;
        v0 = XOR(v0,vpacket0); v4 = XOR(v4,vpacket1);
        v2 = XOR(v2,_mm256_set1_epi64x(0xffLL));
        v6 = XOR(v6,_mm256_set1_epi64x(0xffLL));
        SIPROUNDX2N; SIPROUNDX2N; SIPROUNDX2N; SIPROUNDX2N;
        v0 = XOR(XOR(v0,v1),XOR(v2,v3));
        v4 = XOR(XOR(v4,v5),XOR(v6,v7));

        vpacket0 = _mm256_add_epi64(vpacket0, vpacketinc);
        vpacket1 = _mm256_add_epi64(vpacket1, vpacketinc);
        v1 = _mm256_srli_epi64(v0, YZBITS) & vxmask;
        v5 = _mm256_srli_epi64(v4, YZBITS) & vxmask;
        v0 = (v0 & vyzmask) | vhi0;
        v4 = (v4 & vyzmask) | vhi1;
        vhi0 = _mm256_add_epi64(vhi0, vhiinc);
        vhi1 = _mm256_add_epi64(vhi1, vhiinc);

        u32 ux;
#ifndef NEEDSYNC
#define STORE0(i,v,x,w) \
  ux = _mm256_extract_epi32(v,x);\
  *(u64 *)(base+dst.index[ux]) = _mm256_extract_epi64(w,i%4);\
  dst.index[ux] += BIGSIZE0;
#else
  u32 zz;
#define STORE0(i,v,x,w) \
  zz = _mm256_extract_epi32(w,x);\
  if (i || likely(zz)) {\
    ux = _mm256_extract_epi32(v,x);\
    for (; unlikely(last[ux] + NNONYZ <= edge+i); last[ux] += NNONYZ, dst.index[ux] += BIGSIZE0)\
      *(u32 *)(base+dst.index[ux]) = 0;\
    *(u32 *)(base+dst.index[ux]) = zz;\
    dst.index[ux] += BIGSIZE0;\
    last[ux] = edge+i;\
  }
#endif
        STORE0(0,v1,0,v0); STORE0(1,v1,2,v0); STORE0(2,v1,4,v0); STORE0(3,v1,6,v0);
        STORE0(4,v5,0,v4); STORE0(5,v5,2,v4); STORE0(6,v5,4,v4); STORE0(7,v5,6,v4);
#else
#error not implemented
#endif
      }
#ifdef NEEDSYNC
      for (u32 ux=0; ux < NX; ux++) {
        for (; last[ux]<endedge-NNONYZ; last[ux]+=NNONYZ) {
          *(u32 *)(base+dst.index[ux]) = 0;
          dst.index[ux] += BIGSIZE0;
        }
      }
#endif
      sumsize += dst.storev(buckets, my);
    }
    rdtsc1 = __rdtsc();
    if (!id) printf("genUnodes round %2d size %u rdtsc: %lu\n", uorv, sumsize/BIGSIZE0, rdtsc1-rdtsc0);
    tcounts[id] = sumsize/BIGSIZE0;
}

void edgetrimmer::genVnodes(const u32 id, const u32 uorv) {
    u64 rdtsc0, rdtsc1;
#if NSIPHASH == 4
    const __m128i vxmask = _mm_set1_epi64x(XMASK);
    const __m128i vyzmask = _mm_set1_epi64x(YZMASK);
    const __m128i ff = _mm_set1_epi64x(0xffLL);
    __m128i v0, v1, v2, v3, v4, v5, v6, v7;
    __m128i vpacket0, vpacket1, vhi0, vhi1;
#elif NSIPHASH == 8
    const __m256i vxmask = _mm256_set1_epi64x(XMASK);
    const __m256i vyzmask = _mm256_set1_epi64x(YZMASK);
    const __m256i vinit = _mm256_load_si256((__m256i *)&sip_keys);
    __m256i vpacket0, vpacket1, vhi0, vhi1;
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
#endif
    const u32 NONDEGBITS = std::min(BIGSLOTBITS, 2 * YZBITS) - ZBITS;
    const u32 NONDEGMASK = (1 << NONDEGBITS) - 1;
    indexer<ZBUCKETSIZE> dst;
    indexer<TBUCKETSIZE> small;
  
    rdtsc0 = __rdtsc();
    offset_t sumsize = 0;
    u8 const *base = (u8 *)buckets;
    u8 const *small0 = (u8 *)tbuckets[id];
    const u32 startux = NX *  id    / nthreads;
    const u32   endux = NX * (id+1) / nthreads;
    for (u32 ux = startux; ux < endux; ux++) { // matrix x == ux
      small.matrixu(0);
      for (u32 my = 0 ; my < NY; my++) {
        u32 edge = my << YZBITS;
        u8    *readbig = buckets[ux][my].bytes;
        u8 const *endreadbig = readbig + buckets[ux][my].size;
// printf("id %d x %d y %d size %u read %d\n", id, ux, my, buckets[ux][my].size, readbig-base);
        for (; readbig < endreadbig; readbig += BIGSIZE0) {
// bit     39/31..21     20..13    12..0
// read         edge     UYYYYY    UZZZZ   within UX partition
          BIGTYPE0 e = *(BIGTYPE0 *)readbig;
#if BIGSIZE0 > 4
          e &= BIGSLOTMASK0;
#elif defined NEEDSYNC
          if (unlikely(!e)) { edge += NNONYZ; continue; }
#endif
          edge += ((u32)(e >> YZBITS) - edge) & (NNONYZ-1);
// if (ux==78 && my==243) printf("id %d ux %d my %d e %08x prefedge %x edge %x\n", id, ux, my, e, e >> YZBITS, edge);
          const u32 uy = (e >> ZBITS) & YMASK;
// bit         39..13     12..0
// write         edge     UZZZZ   within UX UY partition
          *(u64 *)(small0+small.index[uy]) = ((u64)edge << ZBITS) | (e & ZMASK);
// printf("id %d ux %d y %d e %010lx e' %010x\n", id, ux, my, e, ((u64)edge << ZBITS) | (e >> YBITS));
          small.index[uy] += SMALLSIZE;
        }
        if (unlikely(edge >> NONYZBITS != (((my+1) << YZBITS) - 1) >> NONYZBITS))
        { printf("OOPS1: id %d ux %d y %d edge %x vs %x\n", id, ux, my, edge, ((my+1)<<YZBITS)-1); exit(0); }
      }
      u8 *degs = tdegs[id];
      small.storeu(tbuckets+id, 0);
      dst.matrixu(ux);
      for (u32 uy = 0 ; uy < NY; uy++) {
        assert(NZ <= sizeof(zbucket8));
        memset(degs, 0xff, NZ);
        u8 *readsmall = tbuckets[id][uy].bytes, *endreadsmall = readsmall + tbuckets[id][uy].size;
// if (id==1) printf("id %d ux %d y %d size %u sumsize %u\n", id, ux, uy, tbuckets[id][uy].size/BIGSIZE, sumsize);
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE)
          degs[*(u32 *)rdsmall & ZMASK]++;
        u16 *zs = tzs[id];
#ifdef SAVEEDGES
        u32 *edges0 = buckets[ux][uy].edges;
#else
        u32 *edges0 = tedges[id];
#endif
        u32 *edges = edges0, edge = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall+=SMALLSIZE) {
// bit         39..13     12..0
// read          edge     UZZZZ    sorted by UY within UX partition
          const u64 e = *(u64 *)rdsmall;
          edge += ((e >> ZBITS) - edge) & NONDEGMASK;
// if (id==0) printf("id %d ux %d uy %d e %010lx pref %4x edge %x mask %x\n", id, ux, uy, e, e>>ZBITS, edge, NONDEGMASK);
          *edges = edge;
          const u32 z = e & ZMASK;
          *zs = z;
          const u32 delta = degs[z] ? 1 : 0;
          edges += delta;
          zs    += delta;
        }
        if (unlikely(edge >> NONDEGBITS != EDGEMASK >> NONDEGBITS))
        { printf("OOPS2: id %d ux %d uy %d edge %x vs %x\n", id, ux, uy, edge, EDGEMASK); exit(0); }
        assert(edges - edges0 < NTRIMMEDZ);
        const u16 *readz = tzs[id];
        const u32 *readedge = edges0;
        int64_t uy34 = (int64_t)uy << YZZBITS;
#if NSIPHASH == 4
        const __m128i vuy34 = _mm_set1_epi64x(uy34);
        const __m128i vuorv = _mm_set1_epi64x(uorv);
        for (; readedge <= edges-NSIPHASH; readedge += NSIPHASH, readz += NSIPHASH) {
          v4 = v0 = _mm_set1_epi64x(sip_keys.k0);
          v5 = v1 = _mm_set1_epi64x(sip_keys.k1);
          v6 = v2 = _mm_set1_epi64x(sip_keys.k2);
          v7 = v3 = _mm_set1_epi64x(sip_keys.k3);

          vpacket0 = _mm_slli_epi64(_mm_cvtepu32_epi64(*(__m128i*) readedge     ), 1) | vuorv;
          vhi0     = vuy34 | _mm_slli_epi64(_mm_cvtepu16_epi64(_mm_set_epi64x(0,*(u64*)readz)), YZBITS);
          vpacket1 = _mm_slli_epi64(_mm_cvtepu32_epi64(*(__m128i*)(readedge + 2)), 1) | vuorv;
          vhi1     = vuy34 | _mm_slli_epi64(_mm_cvtepu16_epi64(_mm_set_epi64x(0,*(u64*)(readz + 2))), YZBITS);

          v3 = XOR(v3,vpacket0); v7 = XOR(v7,vpacket1);
          SIPROUNDX2N; SIPROUNDX2N;
          v0 = XOR(v0,vpacket0); v4 = XOR(v4,vpacket1);
          v2 = XOR(v2,ff);
          v6 = XOR(v6,ff);
          SIPROUNDX2N; SIPROUNDX2N; SIPROUNDX2N; SIPROUNDX2N;
          v0 = XOR(XOR(v0,v1),XOR(v2,v3));
          v4 = XOR(XOR(v4,v5),XOR(v6,v7));

          v1 = _mm_srli_epi64(v0, YZBITS) & vxmask;
          v5 = _mm_srli_epi64(v4, YZBITS) & vxmask;
          v0 = vhi0 | (v0 & vyzmask);
          v4 = vhi1 | (v4 & vyzmask);

          u32 vx;
#define STORE(i,v,x,w) \
  vx = extract32(v,x);\
  *(u64 *)(base+dst.index[vx]) = _mm_extract_epi64(w,i%2);\
  dst.index[vx] += BIGSIZE;
          STORE(0,v1,0,v0); STORE(1,v1,2,v0);
          STORE(2,v5,0,v4); STORE(3,v5,2,v4);
        }
#elif NSIPHASH == 8
        const __m256i vuy34  = _mm256_set1_epi64x(uy34);
        const __m256i vuorv  = _mm256_set1_epi64x(uorv);
        for (; readedge <= edges-NSIPHASH; readedge += NSIPHASH, readz += NSIPHASH) {
          v7 = v3 = _mm256_permute4x64_epi64(vinit, 0xFF);
          v4 = v0 = _mm256_permute4x64_epi64(vinit, 0x00);
          v5 = v1 = _mm256_permute4x64_epi64(vinit, 0x55);
          v6 = v2 = _mm256_permute4x64_epi64(vinit, 0xAA);

          vpacket0 = _mm256_slli_epi64(_mm256_cvtepu32_epi64(*(__m128i*) readedge     ), 1) | vuorv;
          vhi0     = vuy34 | _mm256_slli_epi64(_mm256_cvtepu16_epi64(_mm_set_epi64x(0,*(u64*)readz)), YZBITS);
          vpacket1 = _mm256_slli_epi64(_mm256_cvtepu32_epi64(*(__m128i*)(readedge + 4)), 1) | vuorv;
          vhi1     = vuy34 | _mm256_slli_epi64(_mm256_cvtepu16_epi64(_mm_set_epi64x(0,*(u64*)(readz + 4))), YZBITS);

          v3 = XOR(v3,vpacket0); v7 = XOR(v7,vpacket1);
          SIPROUNDX2N; SIPROUNDX2N;
          v0 = XOR(v0,vpacket0); v4 = XOR(v4,vpacket1);
          v2 = XOR(v2,_mm256_set1_epi64x(0xffLL));
          v6 = XOR(v6,_mm256_set1_epi64x(0xffLL));
          SIPROUNDX2N; SIPROUNDX2N; SIPROUNDX2N; SIPROUNDX2N;
          v0 = XOR(XOR(v0,v1),XOR(v2,v3));
          v4 = XOR(XOR(v4,v5),XOR(v6,v7));
    
          v1 = _mm256_srli_epi64(v0, YZBITS) & vxmask;
          v5 = _mm256_srli_epi64(v4, YZBITS) & vxmask;
          v0 = vhi0 | (v0 & vyzmask);
          v4 = vhi1 | (v4 & vyzmask);

          u32 vx;
#define STORE(i,v,x,w) \
  vx = _mm256_extract_epi32(v,x);\
  *(u64 *)(base+dst.index[vx]) = _mm256_extract_epi64(w,i%4);\
  dst.index[vx] += BIGSIZE;
// printf("Id %d ux %d y %d edge %08x e' %010lx vx %d\n", id, ux, uy, readedge[i], _mm256_extract_epi64(w,i%4), vx);

          STORE(0,v1,0,v0); STORE(1,v1,2,v0); STORE(2,v1,4,v0); STORE(3,v1,6,v0);
          STORE(4,v5,0,v4); STORE(5,v5,2,v4); STORE(6,v5,4,v4); STORE(7,v5,6,v4);
        }
#endif
        for (; readedge < edges; readedge++, readz++) { // process up to 7 leftover edges if NSIPHASH==8
          const u32 node = sipnode(&sip_keys, *readedge, uorv);
          const u32 vx = node >> YZBITS; // & XMASK;
// bit        39..34    33..21     20..13     12..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition
          *(u64 *)(base+dst.index[vx]) = uy34 | ((u64)*readz << YZBITS) | (node & YZMASK);
// printf("id %d ux %d y %d edge %08x e' %010lx vx %d\n", id, ux, uy, *readedge, uy34 | ((u64)(node & YZMASK) << ZBITS) | *readz, vx);
          dst.index[vx] += BIGSIZE;
        }
      }
      sumsize += dst.storeu(buckets, ux);
    }
    rdtsc1 = __rdtsc();
    if (!id) printf("genVnodes round %2d size %u rdtsc: %lu\n", uorv, sumsize/BIGSIZE, rdtsc1-rdtsc0);
    tcounts[id] = sumsize/BIGSIZE;
}

#endif
