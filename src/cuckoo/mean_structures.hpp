#ifndef MEAN_STRUCTURES_HPP
#define MEAN_STRUCTURES_HPP

#include "../crypto/siphashxN.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <x86intrin.h>
#include <assert.h>
#include <vector>
#include <bitset>
#include "../threads/barrier.hpp"
#include "mean_params.hpp"

template<u32 BUCKETSIZE>
struct zbucket {
  u32 size;
  // should avoid different values of RENAMESIZE in different threads of one process
  static const u32 RENAMESIZE = 2*NZ2 + 2*(COMPRESSROUND ? NZ1 : 0);
  union alignas(16) {
    u8 bytes[BUCKETSIZE];
    struct {
#ifdef SAVEEDGES
      u32 words[BUCKETSIZE/sizeof(u32) - RENAMESIZE - NTRIMMEDZ];
#else
      u32 words[BUCKETSIZE/sizeof(u32) - RENAMESIZE];
#endif
      u32 renameu1[NZ2];
      u32 renamev1[NZ2];
      u32 renameu[COMPRESSROUND ? NZ1 : 0];
      u32 renamev[COMPRESSROUND ? NZ1 : 0];
#ifdef SAVEEDGES
      u32 edges[NTRIMMEDZ];
#endif
    };
  };
  u32 setsize(u8 const *end) {
    size = end - bytes;
    assert(size <= BUCKETSIZE);
    return size;
  }
};

template<u32 BUCKETSIZE>
using yzbucket = zbucket<BUCKETSIZE>[NY];
template <u32 BUCKETSIZE>
using matrix = yzbucket<BUCKETSIZE>[NX];

template<u32 BUCKETSIZE>
struct indexer {
  offset_t index[NX];

  void matrixv(const u32 y) {
    const yzbucket<BUCKETSIZE> *foo = 0;
    for (u32 x = 0; x < NX; x++)
      index[x] = foo[x][y].bytes - (u8 *)foo;
  }
  offset_t storev(yzbucket<BUCKETSIZE> *buckets, const u32 y) {
    u8 const *base = (u8 *)buckets;
    offset_t sumsize = 0;
    for (u32 x = 0; x < NX; x++)
      sumsize += buckets[x][y].setsize(base+index[x]);
    return sumsize;
  }
  void matrixu(const u32 x) {
    const yzbucket<BUCKETSIZE> *foo = 0;
    for (u32 y = 0; y < NY; y++)
      index[y] = foo[x][y].bytes - (u8 *)foo;
  }
  offset_t storeu(yzbucket<BUCKETSIZE> *buckets, const u32 x) {
    u8 const *base = (u8 *)buckets;
    offset_t sumsize = 0;
    for (u32 y = 0; y < NY; y++)
      sumsize += buckets[x][y].setsize(base+index[y]);
    return sumsize;
  }
};

class edgetrimmer; // avoid circular references

typedef struct {
  u32 id;
  pthread_t thread;
  edgetrimmer *et;
} thread_ctx;

typedef u8 zbucket8[2*MAXNZNYZ1];
typedef u16 zbucket16[NTRIMMEDZ];
typedef u32 zbucket32[NTRIMMEDZ];

// maintains set of trimmable edges
class edgetrimmer {
public:
  siphash_keys sip_keys;
  yzbucket<ZBUCKETSIZE> *buckets;
  yzbucket<TBUCKETSIZE> *tbuckets;
  zbucket32 *tedges;
  zbucket16 *tzs;
  zbucket8 *tdegs;
  offset_t *tcounts;
  u32 ntrims;
  u32 nthreads;
  bool showall;
  trim_barrier barry;

#if NSIPHASH > 4
  void* operator new(size_t size) noexcept {
#if !defined(_WIN32)
    void* newobj;
    int tmp = posix_memalign(&newobj, NSIPHASH * sizeof(u32), sizeof(edgetrimmer));
    if (tmp != 0) return nullptr;
#else
    void* newobj = _aligned_malloc(sizeof(edgetrimmer), NSIPHASH * sizeof(u32));
    if (newobj == NULL) return nullptr;
#endif
    return newobj;
  }
#endif

  void touch(u8 *p, const offset_t n) {
    for (offset_t i=0; i<n; i+=4096)
      *(u32 *)(p+i) = 0;
  }
  edgetrimmer(const u32 n_threads, const u32 n_trims, const bool show_all) : barry(n_threads) {
    assert(sizeof(matrix<ZBUCKETSIZE>) == NX * sizeof(yzbucket<ZBUCKETSIZE>));
    assert(sizeof(matrix<TBUCKETSIZE>) == NX * sizeof(yzbucket<TBUCKETSIZE>));
    nthreads = n_threads;
    ntrims   = n_trims;
    showall = show_all;
    buckets  = new yzbucket<ZBUCKETSIZE>[NX];
    touch((u8 *)buckets, sizeof(matrix<ZBUCKETSIZE>));
    tbuckets = new yzbucket<TBUCKETSIZE>[nthreads];
    touch((u8 *)tbuckets, nthreads * sizeof(yzbucket<TBUCKETSIZE>));
#ifdef SAVEEDGES
    tedges  = 0;
#else
    tedges  = new zbucket32[nthreads];
#endif
    tdegs   = new zbucket8[nthreads];
    tzs     = new zbucket16[nthreads];
    tcounts = new offset_t[nthreads];
  }
  ~edgetrimmer() {
    delete[] buckets;
    delete[] tbuckets;
    delete[] tedges;
    delete[] tdegs;
    delete[] tzs;
    delete[] tcounts;
  }
  offset_t count() const {
    offset_t cnt = 0;
    for (u32 t = 0; t < nthreads; t++)
      cnt += tcounts[t];
    return cnt;
  }

  void genUnodes(const u32 id, const u32 uorv);
  void genVnodes(const u32 id, const u32 uorv);

  template <u32 SRCSIZE, u32 DSTSIZE, bool TRIMONV>
  void trimedges(const u32 id, const u32 round) {
    const u32 SRCSLOTBITS = std::min(SRCSIZE * 8, 2 * YZBITS);
    const u64 SRCSLOTMASK = (1ULL << SRCSLOTBITS) - 1ULL;
    const u32 SRCPREFBITS = SRCSLOTBITS - YZBITS;
    const u32 SRCPREFMASK = (1 << SRCPREFBITS) - 1;
    const u32 DSTSLOTBITS = std::min(DSTSIZE * 8, 2 * YZBITS);
    const u64 DSTSLOTMASK = (1ULL << DSTSLOTBITS) - 1ULL;
    const u32 DSTPREFBITS = DSTSLOTBITS - YZZBITS;
    const u32 DSTPREFMASK = (1 << DSTPREFBITS) - 1;
    u64 rdtsc0, rdtsc1;
    indexer<ZBUCKETSIZE> dst;
    indexer<TBUCKETSIZE> small;
  
    rdtsc0 = __rdtsc();
    offset_t sumsize = 0;
    u8 const *base = (u8 *)buckets;
    u8 const *small0 = (u8 *)tbuckets[id];
    const u32 startvx = NY *  id    / nthreads;
    const u32   endvx = NY * (id+1) / nthreads;
    for (u32 vx = startvx; vx < endvx; vx++) {
      small.matrixu(0);
      for (u32 ux = 0 ; ux < NX; ux++) {
        u32 uxyz = ux << YZBITS;
        zbucket<ZBUCKETSIZE> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        const u8 *readbig = zb.bytes, *endreadbig = readbig + zb.size;
// printf("id %d vx %d ux %d size %u\n", id, vx, ux, zb.size/SRCSIZE);
        int cnt = 0;
        for (; readbig < endreadbig; readbig += SRCSIZE) {
// bit        39..34    33..21     20..13     12..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition
          const u64 e = *(u64 *)readbig & SRCSLOTMASK;
          uxyz += ((u32)(e >> YZBITS) - uxyz) & SRCPREFMASK;
// if (round==6) printf("id %d vx %d ux %d e %010lx suffUXYZ %05x suffUXY %03x UXYZ %08x UXY %04x mask %x\n", id, vx, ux, e, (u32)(e >> YZBITS), (u32)(e >> YZZBITS), uxyz, uxyz>>ZBITS, SRCPREFMASK);
          const u32 vy = (e >> ZBITS) & YMASK;
// bit     41/39..34    33..26     25..13     12..0
// write      UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition
          (u64)tbuckets[id][vy].bytes[cnt*DSTSIZE] = ((u64)uxyz << ZBITS) | (e & ZMASK);
          //*(u64 *)(tbuckets[id+vy][0].bytes + cnt*DSTSIZE) = ((u64)uxyz << ZBITS) | (e & ZMASK);
          uxyz &= ~ZMASK;
          small.index[vy] += DSTSIZE;
          cnt++;
        }
        if (unlikely(uxyz >> YZBITS != ux))
        { printf("OOPS3: id %d vx %d ux %d UXY %x\n", id, vx, ux, uxyz); exit(0); }
      }
      u8 *degs = tdegs[id];
      small.storeu(tbuckets+id, 0);
      TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      for (u32 vy = 0 ; vy < NY; vy++) {
        const u64 vy34 = (u64)vy << YZZBITS;
        assert(NZ <= sizeof(zbucket8));
        memset(degs, 0xff, NZ);
        u8    *readsmall = tbuckets[id][vy].bytes, *endreadsmall = readsmall + tbuckets[id][vy].size;
// printf("id %d vx %d vy %d size %u sumsize %u\n", id, vx, vy, tbuckets[id][vx].size/BIGSIZE, sumsize);
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += DSTSIZE)
          degs[*(u32 *)rdsmall & ZMASK]++;
        u32 ux = 0;
        int cnt = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += DSTSIZE) {
// bit     41/39..34    33..26     25..13     12..0
// read       UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition
// bit        39..37    36..30     29..15     14..0      with XBITS==YBITS==7
// read       UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition
          const u64 e = *(u64 *)rdsmall & DSTSLOTMASK;
          ux += ((u32)(e >> YZZBITS) - ux) & DSTPREFMASK;
// printf("id %d vx %d vy %d e %010lx suffUX %02x UX %x mask %x\n", id, vx, vy, e, (u32)(e >> YZZBITS), ux, SRCPREFMASK);
// bit    41/39..34    33..21     20..13     12..0
// write     VYYYYY    VZZZZZ     UYYYYY     UZZZZ   within UX partition
          if (TRIMONV) {
            *(u64 *)(buckets[ux][vx].bytes + cnt*DSTSIZE) = vy34 | ((e & ZMASK) << YZBITS) | ((e >> ZBITS) & YZMASK);
            dst.index[ux] += degs[e & ZMASK] ? DSTSIZE : 0;
            cnt += degs[e & ZMASK] ? 1 : 0;
          } else {
            *(u64 *)(buckets[vx][ux].bytes + cnt*DSTSIZE) = vy34 | ((e & ZMASK) << YZBITS) | ((e >> ZBITS) & YZMASK);
            dst.index[ux] += degs[e & ZMASK] ? DSTSIZE : 0;
            cnt += degs[e & ZMASK] ? 1 : 0;
          }
        }
        if (unlikely(ux >> DSTPREFBITS != XMASK >> DSTPREFBITS))
        { printf("OOPS4: id %d vx %x ux %x vs %x\n", id, vx, ux, XMASK); }
      }
      sumsize += TRIMONV ? dst.storev(buckets, vx) : dst.storeu(buckets, vx);
    }
    rdtsc1 = __rdtsc();
    if (showall || (!id && !(round & (round+1))))
      printf("trimedges id %d round %2d size %u rdtsc: %lu\n", id, round, sumsize/DSTSIZE, rdtsc1-rdtsc0);
    tcounts[id] = sumsize/DSTSIZE;
  }

  template <u32 SRCSIZE, u32 DSTSIZE, bool TRIMONV>
  void trimrename(const u32 id, const u32 round) {
    const u32 SRCSLOTBITS = std::min(SRCSIZE * 8, (TRIMONV ? YZBITS : YZ1BITS) + YZBITS);
    const u64 SRCSLOTMASK = (1ULL << SRCSLOTBITS) - 1ULL;
    const u32 SRCPREFBITS = SRCSLOTBITS - YZBITS;
    const u32 SRCPREFMASK = (1 << SRCPREFBITS) - 1;
    const u32 SRCPREFBITS2 = SRCSLOTBITS - YZZBITS;
    const u32 SRCPREFMASK2 = (1 << SRCPREFBITS2) - 1;
    u64 rdtsc0, rdtsc1;
    indexer<ZBUCKETSIZE> dst;
    indexer<TBUCKETSIZE> small;
    u32 maxnnid = 0;
  
    rdtsc0 = __rdtsc();
    offset_t sumsize = 0;
    u8 const *base = (u8 *)buckets;
    u8 const *small0 = (u8 *)tbuckets[id];
    const u32 startvx = NY *  id    / nthreads;
    const u32   endvx = NY * (id+1) / nthreads;
    for (u32 vx = startvx; vx < endvx; vx++) {
      small.matrixu(0);
      for (u32 ux = 0 ; ux < NX; ux++) {
        u32 uyz = 0;
        zbucket<ZBUCKETSIZE> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        const u8 *readbig = zb.bytes, *endreadbig = readbig + zb.size;
// printf("id %d vx %d ux %d size %u\n", id, vx, ux, zb.size/SRCSIZE);
        for (; readbig < endreadbig; readbig += SRCSIZE) {
// bit        39..37    36..22     21..15     14..0
// write      UYYYYY    UZZZZZ     VYYYYY     VZZZZ   within VX partition  if TRIMONV
// bit            36...22     21..15     14..0
// write          VYYYZZ'     UYYYYY     UZZZZ   within UX partition  if !TRIMONV
          const u64 e = *(u64 *)readbig & SRCSLOTMASK;
          if (TRIMONV)
            uyz += ((u32)(e >> YZBITS) - uyz) & SRCPREFMASK;
          else uyz = e >> YZBITS;
// if (round==32 && ux==25) printf("id %d vx %d ux %d e %010lx suffUXYZ %05x suffUXY %03x UXYZ %08x UXY %04x mask %x\n", id, vx, ux, e, (u32)(e >> YZBITS), (u32)(e >> YZZBITS), uxyz, uxyz>>ZBITS, SRCPREFMASK);
          const u32 vy = (e >> ZBITS) & YMASK;
// bit        39..37    36..30     29..15     14..0
// write      UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition  if TRIMONV
// bit            36...30     29...15     14..0
// write          VXXXXXX     VYYYZZ'     UZZZZ   within UX UY partition  if !TRIMONV
          *(u64 *)(small0+small.index[vy]) = ((u64)(ux << (TRIMONV ? YZBITS : YZ1BITS) | uyz) << ZBITS) | (e & ZMASK);
// if (TRIMONV&&vx==75&&vy==83) printf("id %d vx %d vy %d e %010lx e15 %x ux %x\n", id, vx, vy, ((u64)uxyz << ZBITS) | (e & ZMASK), uxyz, uxyz>>YZBITS);
          if (TRIMONV)
            uyz &= ~ZMASK;
          small.index[vy] += SRCSIZE;
        }
      }
      u16 *degs = (u16 *)tdegs[id];
      small.storeu(tbuckets+id, 0);
      TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      u32 newnodeid = 0;
      u32 *renames = TRIMONV ? buckets[0][vx].renamev : buckets[vx][0].renameu;
      u32 *endrenames = renames + NZ1;
      for (u32 vy = 0 ; vy < NY; vy++) {
        assert(2*NZ <= sizeof(zbucket8));
        memset(degs, 0xff, 2*NZ);
        u8    *readsmall = tbuckets[id][vy].bytes, *endreadsmall = readsmall + tbuckets[id][vy].size;
// printf("id %d vx %d vy %d size %u sumsize %u\n", id, vx, vy, tbuckets[id][vx].size/BIGSIZE, sumsize);
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += SRCSIZE)
          degs[*(u32 *)rdsmall & ZMASK]++;
        u32 ux = 0;
        u32 nrenames = 0;
        for (u8 *rdsmall = readsmall; rdsmall < endreadsmall; rdsmall += SRCSIZE) {
// bit        39..37    36..30     29..15     14..0
// read       UXXXXX    UYYYYY     UZZZZZ     VZZZZ   within VX VY partition  if TRIMONV
// bit            36...30     29...15     14..0
// read           VXXXXXX     VYYYZZ'     UZZZZ   within UX UY partition  if !TRIMONV
          const u64 e = *(u64 *)rdsmall & SRCSLOTMASK;
          if (TRIMONV)
            ux += ((u32)(e >> YZZBITS) - ux) & SRCPREFMASK2;
          else ux = e >> YZZ1BITS;
          const u32 vz = e & ZMASK;
          u16 vdeg = degs[vz];
// if (TRIMONV&&vx==75&&vy==83) printf("id %d vx %d vy %d e %010lx e37 %x ux %x vdeg %d nrenames %d\n", id, vx, vy, e, e>>YZZBITS, ux, vdeg, nrenames);
          if (vdeg) {
            if (vdeg < 32) {
              degs[vz] = vdeg = 32 + nrenames++;
              *renames++ = vy << ZBITS | vz;
              if (renames == endrenames) {
                endrenames += (TRIMONV ? sizeof(yzbucket<ZBUCKETSIZE>) : sizeof(zbucket<ZBUCKETSIZE>)) / sizeof(u32);
                renames = endrenames - NZ1;
              }
            }
// bit       36..22     21..15     14..0
// write     VYYZZ'     UYYYYY     UZZZZ   within UX partition  if TRIMONV
            if (TRIMONV)
                 *(u64 *)(base+dst.index[ux]) = ((u64)(newnodeid + vdeg-32) << YZBITS ) | ((e >> ZBITS) & YZMASK);
            else *(u32 *)(base+dst.index[ux]) = ((newnodeid + vdeg-32) << YZ1BITS) | ((e >> ZBITS) & YZ1MASK);
// if (vx==44&&vy==58) printf("  id %d vx %d vy %d newe %010lx\n", id, vx, vy, vy28 | ((vdeg) << YZBITS) | ((e >> ZBITS) & YZMASK));
            dst.index[ux] += DSTSIZE;
          }
        }
        newnodeid += nrenames;
        if (TRIMONV && unlikely(ux >> SRCPREFBITS2 != XMASK >> SRCPREFBITS2))
        { printf("OOPS6: id %d vx %d vy %d ux %x vs %x\n", id, vx, vy, ux, XMASK); exit(0); }
      }
      if (newnodeid > maxnnid)
        maxnnid = newnodeid;
      sumsize += TRIMONV ? dst.storev(buckets, vx) : dst.storeu(buckets, vx);
    }
    rdtsc1 = __rdtsc();
    if (showall || !id) printf("trimrename id %d round %2d size %u rdtsc: %lu maxnnid %d\n", id, round, sumsize/DSTSIZE, rdtsc1-rdtsc0, maxnnid);
    assert(maxnnid < NYZ1);
    tcounts[id] = sumsize/DSTSIZE;
  }

  template <bool TRIMONV>
  void trimedges1(const u32 id, const u32 round) {
    u64 rdtsc0, rdtsc1;
    indexer<ZBUCKETSIZE> dst;
  
    rdtsc0 = __rdtsc();
    offset_t sumsize = 0;
    u8 *degs = tdegs[id];
    u8 const *base = (u8 *)buckets;
    const u32 startvx = NY *  id    / nthreads;
    const u32   endvx = NY * (id+1) / nthreads;
    for (u32 vx = startvx; vx < endvx; vx++) {
      TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      assert(NYZ1 <= sizeof(zbucket8));
      memset(degs, 0xff, NYZ1);
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        // printf("id %d vx %d ux %d size %d\n", id, vx, ux, zb.size/SRCSIZE);
        for (; readbig < endreadbig; readbig++)
          degs[*readbig & YZ1MASK]++;
      }
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        for (; readbig < endreadbig; readbig++) {
// bit       29..22    21..15     14..7     6..0
// read      UYYYYY    UZZZZ'     VYYYY     VZZ'   within VX partition
          const u32 e = *readbig;
          const u32 vyz = e & YZ1MASK;
          // printf("id %d vx %d ux %d e %08lx vyz %04x uyz %04x\n", id, vx, ux, e, vyz, e >> YZ1BITS);
// bit       29..22    21..15     14..7     6..0
// write     VYYYYY    VZZZZ'     UYYYY     UZZ'   within UX partition
          *(u32 *)(base+dst.index[ux]) = (vyz << YZ1BITS) | (e >> YZ1BITS);
          dst.index[ux] += degs[vyz] ? sizeof(u32) : 0;
        }
      }
      sumsize += TRIMONV ? dst.storev(buckets, vx) : dst.storeu(buckets, vx);
    }
    rdtsc1 = __rdtsc();
    if (showall || (!id && !(round & (round+1))))
      printf("trimedges1 id %d round %2d size %u rdtsc: %lu\n", id, round, sumsize/sizeof(u32), rdtsc1-rdtsc0);
    tcounts[id] = sumsize/sizeof(u32);
  }

  template <bool TRIMONV>
  void trimrename1(const u32 id, const u32 round) {
    u64 rdtsc0, rdtsc1;
    indexer<ZBUCKETSIZE> dst;
    u32 maxnnid = 0;
  
    rdtsc0 = __rdtsc();
    offset_t sumsize = 0;
    u16 *degs = (u16 *)tdegs[id];
    u8 const *base = (u8 *)buckets;
    const u32 startvx = NY *  id    / nthreads;
    const u32   endvx = NY * (id+1) / nthreads;
    for (u32 vx = startvx; vx < endvx; vx++) {
      TRIMONV ? dst.matrixv(vx) : dst.matrixu(vx);
      memset(degs, 0xff, 2 * NYZ1); // sets each u16 entry to 0xffff
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        // printf("id %d vx %d ux %d size %d\n", id, vx, ux, zb.size/SRCSIZE);
        for (; readbig < endreadbig; readbig++)
          degs[*readbig & YZ1MASK]++;
      }
      u32 newnodeid = 0;
      u32 *renames = TRIMONV ? buckets[0][vx].renamev1 : buckets[vx][0].renameu1;
      u32 *endrenames = renames + NZ2;
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE> &zb = TRIMONV ? buckets[ux][vx] : buckets[vx][ux];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
        for (; readbig < endreadbig; readbig++) {
// bit       29...15     14...0
// read      UYYYZZ'     VYYZZ'   within VX partition
          const u32 e = *readbig;
          const u32 vyz = e & YZ1MASK;
          u16 vdeg = degs[vyz];
          if (vdeg) {
            if (vdeg < 32) {
              degs[vyz] = vdeg = 32 + newnodeid++;
              *renames++ = vyz;
              if (renames == endrenames) {
                endrenames += (TRIMONV ? sizeof(yzbucket<ZBUCKETSIZE>) : sizeof(zbucket<ZBUCKETSIZE>)) / sizeof(u32);
                renames = endrenames - NZ2;
                assert(renames < buckets[NX][0].renameu1);
              }
            }
// bit       25...15     14...0
// write     VYYZZZ"     UYYZZ'   within UX partition
            *(u32 *)(base+dst.index[ux]) = ((vdeg - 32) << (TRIMONV ? YZ1BITS : YZ2BITS)) | (e >> YZ1BITS);
            dst.index[ux] += sizeof(u32);
          }
        }
      }
      if (newnodeid > maxnnid)
        maxnnid = newnodeid;
      sumsize += TRIMONV ? dst.storev(buckets, vx) : dst.storeu(buckets, vx);
    }
    rdtsc1 = __rdtsc();
    if (showall || !id) printf("trimrename1 id %d round %2d size %u rdtsc: %lu maxnnid %d\n", id, round, sumsize/sizeof(u32), rdtsc1-rdtsc0, maxnnid);
    assert(maxnnid < NYZ2);
    tcounts[id] = sumsize/sizeof(u32);
  }

  void trim() {
    void *etworker(void *vp);
    barry.clear();
    thread_ctx *threads = new thread_ctx[nthreads];
    for (u32 t = 0; t < nthreads; t++) {
      genUnodes(t, 0);
      genVnodes(t, 1);
    }
    for (u32 t = 0; t < nthreads; t++) {
      threads[t].id = t;
      threads[t].et = this;
      int err = pthread_create(&threads[t].thread, NULL, etworker, (void *)&threads[t]);
      assert(err == 0);
    }
    for (u32 t = 0; t < nthreads; t++) {
      int err = pthread_join(threads[t].thread, NULL);
      assert(err == 0);
    }
    delete[] threads;
  }
  void barrier() {
    barry.wait();
  }
#ifdef EXPANDROUND
#define BIGGERSIZE BIGSIZE+1
#else
#define BIGGERSIZE BIGSIZE
#define EXPANDROUND COMPRESSROUND
#endif
  void trimmer(u32 id) {
    // genUnodes(id, 0);
    barrier();
    // genVnodes(id, 1);
    for (u32 round = 2; round < ntrims-2; round += 2) {
      barrier();
      if (round < COMPRESSROUND) {
        if (round < EXPANDROUND)
          trimedges<BIGSIZE, BIGSIZE, true>(id, round);
        else if (round == EXPANDROUND)
          trimedges<BIGSIZE, BIGGERSIZE, true>(id, round);
        else trimedges<BIGGERSIZE, BIGGERSIZE, true>(id, round);
      } else if (round==COMPRESSROUND) {
        trimrename<BIGGERSIZE, BIGGERSIZE, true>(id, round);
      } else trimedges1<true>(id, round);
      barrier();
      if (round < COMPRESSROUND) {
        if (round+1 < EXPANDROUND)
          trimedges<BIGSIZE, BIGSIZE, false>(id, round+1);
        else if (round+1 == EXPANDROUND)
          trimedges<BIGSIZE, BIGGERSIZE, false>(id, round+1);
        else trimedges<BIGGERSIZE, BIGGERSIZE, false>(id, round+1);
      } else if (round==COMPRESSROUND) {
        trimrename<BIGGERSIZE, sizeof(u32), false>(id, round+1);
      } else trimedges1<false>(id, round+1);
    }
    barrier();
    trimrename1<true >(id, ntrims-2);
    barrier();
    trimrename1<false>(id, ntrims-1);
  }
};


#endif