// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2020 John Tromp
// The edge-trimming memory optimization is due to Dave Andersen
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html
// xenoncat demonstrated at https://github.com/xenoncat/cuckoo_pow
// how bucket sorting avoids random memory access latency
// my own cycle finding is run single threaded to avoid losing cycles
// to race conditions (typically takes under 1% of runtime)

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <x86intrin.h>
#include <assert.h>
#include <vector>
#include <bitset>
#include "mean_params.hpp"
#include "mean_trimedges.hpp"
#include "mean_gennodes.hpp"
#include "../crypto/siphashxN.h"
#include "../threads/barrier.hpp"

// The matrix solver stores all edges in a matrix of NX * NX buckets,
// where NX = 2^XBITS is the number of possible values of the 'X' bits.
// Edge i between nodes ui = siphash24(2*i) and vi = siphash24(2*i+1)
// resides in the bucket at (uiX,viX)
// In each trimming round, either a matrix row or a matrix column (NX buckets)
// is bucket sorted on uY or vY respectively, and then within each bucket
// uZ or vZ values are counted and edges with a count of only one are eliminated,
// while remaining edges are bucket sorted back on vX or uX respectively.
// When sufficiently many edges have been eliminated, a pair of compression
// rounds remap surviving Y,Z values in each row or column into 15 bit
// combined YZ values, allowing the remaining rounds to avoid the sorting on Y,
// and directly count YZ values in a cache friendly 32KB.
// A final pair of compression rounds remap YZ values from 15 into 11 bits.

void *etworker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  tp->et->trimmer(tp->id);
  pthread_exit(NULL);
  return 0;
}

#define NODEBITS (EDGEBITS + 1)

// grow with cube root of size, hardly affected by trimming
const u32 MAXPATHLEN = 16 << (EDGEBITS/3);

const u32 CUCKOO_SIZE = 2 * NX * NYZ2;

int nonce_cmp(const void *a, const void *b) {
  return *(u32 *)a - *(u32 *)b;
}

typedef word_t proof[PROOFSIZE];

// break circular reference with forward declaration
class solver_ctx;

typedef struct {
  u32 id;
  pthread_t thread;
  solver_ctx *solver;
} match_ctx;

class solver_ctx {
public:
  edgetrimmer *trimmer;
  u32 *cuckoo = 0;
  bool showcycle;
  proof cycleus;
  proof cyclevs;
  std::bitset<NXY> uxymap;
  std::vector<word_t> sols; // concatanation of all proof's indices

  solver_ctx(const u32 n_threads, const u32 n_trims, bool allrounds, bool show_cycle) {
    trimmer = new edgetrimmer(n_threads, n_trims, allrounds);
    showcycle = show_cycle;
    cuckoo = 0;
  }
  void setheadernonce(char* const headernonce, const u32 len, const u32 nonce) {
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer->sip_keys);
    sols.clear();
  }
  ~solver_ctx() {
    delete trimmer;
  }
  u64 sharedbytes() const {
    return sizeof(matrix<ZBUCKETSIZE>);
  }
  u32 threadbytes() const {
    return sizeof(thread_ctx) + sizeof(yzbucket<TBUCKETSIZE>) + sizeof(zbucket8) + sizeof(zbucket16) + sizeof(zbucket32);
  }
  void recordedge(const u32 i, const u32 u2, const u32 v2) {
    const u32 u1 = u2/2;
    const u32 ux = u1 >> YZ2BITS;
    u32 uyz = trimmer->buckets[ux][(u1 >> Z2BITS) & YMASK].renameu1[u1 & Z2MASK];
    assert(uyz < NYZ1);
    const u32 v1 = v2/2;
    const u32 vx = v1 >> YZ2BITS;
    u32 vyz = trimmer->buckets[(v1 >> Z2BITS) & YMASK][vx].renamev1[v1 & Z2MASK];
    assert(vyz < NYZ1);
#if COMPRESSROUND > 0
    uyz = trimmer->buckets[ux][uyz >> Z1BITS].renameu[uyz & Z1MASK];
    vyz = trimmer->buckets[vyz >> Z1BITS][vx].renamev[vyz & Z1MASK];
#endif
    const u32 u = cycleus[i] = (ux << YZBITS) | uyz;
    const u32 v = cyclevs[i] = (vx << YZBITS) | vyz;
    printf(" (%x,%x)", 2*u, 2*v+1);
#ifdef SAVEEDGES
    u32 *readedges = trimmer->buckets[ux][uyz >> ZBITS].edges, *endreadedges = readedges + NTRIMMEDZ;
    for (; readedges < endreadedges; readedges++) {
      u32 edge = *readedges;
      if (sipnode(&trimmer->sip_keys, edge, 1) == v && sipnode(&trimmer->sip_keys, edge, 0) == u) {
        sols.push_back(edge);
        return;
      }
    }
    assert(0);
#else
    uxymap[u >> ZBITS] = 1;
#endif
  }

  void solution(const u32 *us, u32 nu, const u32 *vs, u32 nv) {
    printf("Nodes");
    u32 ni = 0;
    recordedge(ni++, *us, *vs);
    while (nu--)
      recordedge(ni++, us[(nu+1)&~1], us[nu|1]); // u's in even position; v's in odd
    while (nv--)
      recordedge(ni++, vs[nv|1], vs[(nv+1)&~1]); // u's in odd position; v's in even
    printf("\n");
    if (showcycle) {
#ifndef SAVEEDGES
      void *matchworker(void *vp);

      sols.resize(sols.size() + PROOFSIZE);
      match_ctx *threads = new match_ctx[trimmer->nthreads];
      for (u32 t = 0; t < trimmer->nthreads; t++) {
        threads[t].id = t;
        threads[t].solver = this;
        int err = pthread_create(&threads[t].thread, NULL, matchworker, (void *)&threads[t]);
        assert(err == 0);
      }
      for (u32 t = 0; t < trimmer->nthreads; t++) {
        int err = pthread_join(threads[t].thread, NULL);
        assert(err == 0);
      }
      delete[] threads;
#endif
      qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), nonce_cmp);
    }
  }

  const u32 CUCKOO_NIL = ~0;

  u32 path(u32 u, u32 *us) const {
    u32 nu, u0 = u;
    for (nu = 0; u != CUCKOO_NIL; u = cuckoo[u]) {
      if (nu >= MAXPATHLEN) {
        while (nu-- && us[nu] != u) ;
        if (!~nu)
          printf("maximum path length exceeded\n");
        else printf("illegal %4d-cycle from node %d\n", MAXPATHLEN-nu, u0);
        pthread_exit(NULL);
      }
      us[nu++] = u;
    }
    return nu-1;
  }
  
  void findcycles() {
    u32 us[MAXPATHLEN], vs[MAXPATHLEN];
    u64 rdtsc0, rdtsc1;
  
    rdtsc0 = __rdtsc();
    for (u32 vx = 0; vx < NX; vx++) {
      for (u32 ux = 0 ; ux < NX; ux++) {
        zbucket<ZBUCKETSIZE> &zb = trimmer->buckets[ux][vx];
        u32 *readbig = zb.words, *endreadbig = readbig + zb.size/sizeof(u32);
// printf("vx %d ux %d size %u\n", vx, ux, zb.size/4);
        for (; readbig < endreadbig; readbig++) {
// bit        21..11     10...0
// write      UYYZZZ'    VYYZZ'   within VX partition
          const u32 e = *readbig;
          const u32 uxyz = (ux << YZ2BITS) | (e >> YZ2BITS);
          const u32 vxyz = (vx << YZ2BITS) | (e & YZ2MASK);
          const u32 u0 = uxyz << 1, v0 = (vxyz << 1) | 1;
          if (u0 != CUCKOO_NIL) {
            u32 nu = path(u0, us), nv = path(v0, vs);
// printf("vx %02x ux %02x e %08x uxyz %06x vxyz %06x u0 %x v0 %x nu %d nv %d\n", vx, ux, e, uxyz, vxyz, u0, v0, nu, nv);
            if (us[nu] == vs[nv]) {
              const u32 min = nu < nv ? nu : nv;
              for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
              const u32 len = nu + nv + 1;
              printf("%4d-cycle found\n", len);
              if (len == PROOFSIZE)
                solution(us, nu, vs, nv);
            } else if (nu < nv) {
              while (nu--)
                cuckoo[us[nu+1]] = us[nu];
              cuckoo[u0] = v0;
            } else {
              while (nv--)
                cuckoo[vs[nv+1]] = vs[nv];
              cuckoo[v0] = u0;
            }
          }
        }
      }
    }
    rdtsc1 = __rdtsc();
    printf("findcycles rdtsc: %lu\n", rdtsc1-rdtsc0);
  }

  int solve() {
    assert((u64)CUCKOO_SIZE * sizeof(u32) <= trimmer->nthreads * sizeof(yzbucket<TBUCKETSIZE>));
    trimmer->trim();
    cuckoo = (u32 *)trimmer->tbuckets;
    memset(cuckoo, CUCKOO_NIL, CUCKOO_SIZE * sizeof(u32));
    findcycles();
    return sols.size() / PROOFSIZE;
  }

  void *matchUnodes(match_ctx *mc) {
    u64 rdtsc0, rdtsc1;
  
    rdtsc0 = __rdtsc();
    const u32 starty = NY *  mc->id    / trimmer->nthreads;
    const u32   endy = NY * (mc->id+1) / trimmer->nthreads;
    u32 edge = starty << YZBITS, endedge = edge + NYZ;
  #if NSIPHASH == 4
    const __m128i vnodemask = _mm_set1_epi64x(EDGEMASK);
    siphash_keys &sip_keys = trimmer->sip_keys;
    __m128i v0, v1, v2, v3, v4, v5, v6, v7;
    const u32 e2 = 2 * edge;
    __m128i vpacket0 = _mm_set_epi64x(e2+2, e2+0);
    __m128i vpacket1 = _mm_set_epi64x(e2+6, e2+4);
    const __m128i vpacketinc = _mm_set1_epi64x(8);
  #elif NSIPHASH == 8
    const __m256i vnodemask = _mm256_set1_epi64x(EDGEMASK);
    const __m256i vinit = _mm256_load_si256((__m256i *)&trimmer->sip_keys);
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    const u32 e2 = 2 * edge;
    __m256i vpacket0 = _mm256_set_epi64x(e2+6, e2+4, e2+2, e2+0);
    __m256i vpacket1 = _mm256_set_epi64x(e2+14, e2+12, e2+10, e2+8);
    const __m256i vpacketinc = _mm256_set1_epi64x(16);
  #endif
    for (u32 my = starty; my < endy; my++, endedge += NYZ) {
      for (; edge < endedge; edge += NSIPHASH) {
  // bit        28..21     20..13    12..0
  // node       XXXXXX     YYYYYY    ZZZZZ
  #if NSIPHASH == 1
        const u32 nodeu = sipnode(&trimmer->sip_keys, edge, 0);
        if (uxymap[nodeu >> ZBITS]) {
          for (u32 j = 0; j < PROOFSIZE; j++) {
            if (cycleus[j] == nodeu && cyclevs[j] == sipnode(&trimmer->sip_keys, edge, 1)) {
              sols[sols.size()-PROOFSIZE + j] = edge;
            }
          }
        }
  // bit        39..21     20..13    12..0
  // write        edge     YYYYYY    ZZZZZ
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
        v0 = v0 & vnodemask;
        v4 = v4 & vnodemask;
        v1 = _mm_srli_epi64(v0, ZBITS);
        v5 = _mm_srli_epi64(v4, ZBITS);

        u32 uxy;
  #define MATCH(i,v,x,w) \
  uxy = extract32(v,x);\
  if (uxymap[uxy]) {\
    u32 u = extract32(w,x);\
    for (u32 j = 0; j < PROOFSIZE; j++) {\
      if (cycleus[j] == u && cyclevs[j] == sipnode(&trimmer->sip_keys, edge+i, 1)) {\
        sols[sols.size()-PROOFSIZE + j] = edge + i;\
      }\
    }\
  }
        MATCH(0,v1,0,v0); MATCH(1,v1,2,v0);
        MATCH(2,v5,0,v4); MATCH(3,v5,2,v4);
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
        v0 = v0 & vnodemask;
        v4 = v4 & vnodemask;
        v1 = _mm256_srli_epi64(v0, ZBITS);
        v5 = _mm256_srli_epi64(v4, ZBITS);
  
        u32 uxy;
  #define MATCH(i,v,x,w) \
  uxy = _mm256_extract_epi32(v,x);\
  if (uxymap[uxy]) {\
    u32 u = _mm256_extract_epi32(w,x);\
    for (u32 j = 0; j < PROOFSIZE; j++) {\
      if (cycleus[j] == u && cyclevs[j] == sipnode(&trimmer->sip_keys, edge+i, 1)) {\
        sols[sols.size()-PROOFSIZE + j] = edge + i;\
      }\
    }\
  }
        MATCH(0,v1,0,v0); MATCH(1,v1,2,v0); MATCH(2,v1,4,v0); MATCH(3,v1,6,v0);
        MATCH(4,v5,0,v4); MATCH(5,v5,2,v4); MATCH(6,v5,4,v4); MATCH(7,v5,6,v4);
  #else
  #error not implemented
  #endif
      }
    }
    rdtsc1 = __rdtsc();
    if (trimmer->showall || !mc->id) printf("matchUnodes id %d rdtsc: %lu\n", mc->id, rdtsc1-rdtsc0);
    pthread_exit(NULL);
    return 0;
  }
};

void *matchworker(void *vp) {
  match_ctx *tp = (match_ctx *)vp;
  tp->solver->matchUnodes(tp);
  pthread_exit(NULL);
  return 0;
}
