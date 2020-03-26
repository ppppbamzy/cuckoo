#ifndef MEAN_PARAMS_HPP
#define MEAN_PARAMS_HPP

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include <stdarg.h>
#include <stdio.h>
#include <chrono>
#include <ctime>
#include "../crypto/blake2.h"
#include "../crypto/siphash.hpp"

#ifdef SIPHASH_COMPAT
#include <stdio.h>
#endif

// proof-of-work parameters
#ifndef EDGEBITS
// the main parameter is the 2-log of the graph size,
// which is the size in bits of the node identifiers
#define EDGEBITS 29
#endif
#ifndef PROOFSIZE
// the next most important parameter is the (even) length
// of the cycle to be found. a minimum of 12 is recommended
#define PROOFSIZE 42
#endif

// save some keystrokes since i'm a lazy typer
typedef uint32_t u32;
typedef uint64_t u64;

#ifndef MAX_SOLS
#define MAX_SOLS 4
#endif

#if EDGEBITS > 30
typedef uint64_t word_t;
#elif EDGEBITS > 14
typedef u32 word_t;
#else // if EDGEBITS <= 14
typedef uint16_t word_t;
#endif

// number of edges
#define NEDGES ((word_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK ((word_t)NEDGES - 1)
#define NODE1MASK EDGEMASK

// Common Solver parameters, to return to caller
struct SolverParams {
        u32 nthreads = 0;
        u32 ntrims = 0;
        bool showcycle;
        bool allrounds;
        bool mutate_nonce = 1;
        bool cpuload = 1;

        // Common cuda params
        u32 device = 0;

        // Cuda-lean specific params
        u32 blocks = 0;
        u32 tpb = 0;

        // Cuda-mean specific params
        u32 expand = 0;
        u32 genablocks = 0;
        u32 genatpb = 0;
        u32 genbtpb = 0;
        u32 trimtpb = 0;
        u32 tailtpb = 0;
        u32 recoverblocks = 0;
        u32 recovertpb = 0;
};

// Solutions result structs to be instantiated by caller,
// and filled by solver if desired
struct Solution {
 u64 nonce = 0;
 u64 proof[PROOFSIZE];
};

struct SolverSolutions {
 u32 edge_bits = 0;
 u32 num_sols = 0;
 Solution sols[MAX_SOLS];
};

#define MAX_NAME_LEN 256

// last error reason, to be picked up by stats
// to be returned to caller
char LAST_ERROR_REASON[MAX_NAME_LEN];

// Solver statistics, to be instantiated by caller
// and filled by solver if desired
struct SolverStats {
        u32 device_id = 0;
        u32 edge_bits = 0;
        char plugin_name[MAX_NAME_LEN]; // will be filled in caller-side
        char device_name[MAX_NAME_LEN];
        bool has_errored = false;
        char error_reason[MAX_NAME_LEN];
        u32 iterations = 0;
        u64 last_start_time = 0;
        u64 last_end_time = 0;
        u64 last_solution_time = 0;
};

// generate edge endpoint in cuckoo graph without partition bit
word_t sipnode(siphash_keys *keys, word_t edge, u32 uorv) {
  return keys->siphash24(2*edge + uorv) & EDGEMASK;
}

enum verify_code { POW_OK, POW_HEADER_LENGTH, POW_TOO_BIG, POW_TOO_SMALL, POW_NON_MATCHING, POW_BRANCH, POW_DEAD_END, POW_SHORT_CYCLE};
const char *errstr[] = { "OK", "wrong header length", "edge too big", "edges not ascending", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

// verify that edges are ascending and form a cycle in header-generated graph
int verify(word_t edges[PROOFSIZE], siphash_keys *keys) {
  word_t uvs[2*PROOFSIZE];
  word_t xor0 = 0, xor1 = 0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (edges[n] > EDGEMASK)
      return POW_TOO_BIG;
    if (n && edges[n] <= edges[n-1])
      return POW_TOO_SMALL;
    xor0 ^= uvs[2*n  ] = sipnode(keys, edges[n], 0);
    xor1 ^= uvs[2*n+1] = sipnode(keys, edges[n], 1);
  }
  if (xor0|xor1)              // optional check for obviously bad proofs
    return POW_NON_MATCHING;
  u32 n = 0, i = 0, j;
  do {                        // follow cycle
    for (u32 k = j = i; (k = (k+2) % (2*PROOFSIZE)) != i; ) {
      if (uvs[k] == uvs[i]) { // find other edge endpoint identical to one at i
        if (j != i)           // already found one before
          return POW_BRANCH;
        j = k;
      }
    }
    if (j == i) return POW_DEAD_END;  // no matching endpoint
    i = j^1;
    n++;
  } while (i != 0);           // must cycle back to start or we would have found branch
  return n == PROOFSIZE ? POW_OK : POW_SHORT_CYCLE;
}

// convenience function for extracting siphash keys from header
void setheader(const char *header, const u32 headerlen, siphash_keys *keys) {
  char hdrkey[32];
  // SHA256((unsigned char *)header, headerlen, (unsigned char *)hdrkey);
  blake2b((void *)hdrkey, sizeof(hdrkey), (const void *)header, headerlen, 0, 0);
#ifdef SIPHASH_COMPAT
  u64 *k = (u64 *)hdrkey;
  u64 k0 = k[0];
  u64 k1 = k[1];
  printf("k0 k1 %lx %lx\n", k0, k1);
  k[0] = k0 ^ 0x736f6d6570736575ULL;
  k[1] = k1 ^ 0x646f72616e646f6dULL;
  k[2] = k0 ^ 0x6c7967656e657261ULL;
  k[3] = k1 ^ 0x7465646279746573ULL;
#endif
  keys->setkeys(hdrkey);
}

// edge endpoint in cuckoo graph with partition bit
word_t sipnode_(siphash_keys *keys, word_t edge, u32 uorv) {
  return sipnode(keys, edge, uorv) << 1 | uorv;
}

u64 timestamp() {
        using namespace std::chrono;
        high_resolution_clock::time_point now = high_resolution_clock::now();
        auto dn = now.time_since_epoch();
        return dn.count();
}

/////////////////////////////////////////////////////////////////
// Declarations to make it easier for callers to link as required
/////////////////////////////////////////////////////////////////

#ifndef C_CALL_CONVENTION
#define C_CALL_CONVENTION 0
#endif

// convention to prepend to called functions
#if C_CALL_CONVENTION
#define CALL_CONVENTION extern "C"
#else
#define CALL_CONVENTION
#endif

// Ability to squash printf output at compile time, if desired
#ifndef SQUASH_OUTPUT
#define SQUASH_OUTPUT 0
#endif

void print_log(const char *fmt, ...) {
        if (SQUASH_OUTPUT) return;
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
}
//////////////////////////////////////////////////////////////////
// END caller QOL
//////////////////////////////////////////////////////////////////


// The node bits are logically split into 3 groups:
// XBITS 'X' bits (most significant), YBITS 'Y' bits, and ZBITS 'Z' bits (least significant)
// Here we have the default XBITS=YBITS=7, ZBITS=15 summing to EDGEBITS=29
// nodebits   XXXXXXX YYYYYYY ZZZZZZZZZZZZZZZ
// bit%10     8765432 1098765 432109876543210
// bit/10     2222222 2211111 111110000000000

#ifndef XBITS
// 7 seems to give best performance
#define XBITS 7
#endif

#define YBITS XBITS

// size in bytes of a big bucket entry
#ifndef BIGSIZE
#if EDGEBITS <= 15
#define BIGSIZE 4
// no compression needed
#define COMPRESSROUND 0
#else
#define BIGSIZE 5
// YZ compression round; must be even
#ifndef COMPRESSROUND
#define COMPRESSROUND 14
#endif
#endif
#endif
// size in bytes of a small bucket entry
#define SMALLSIZE BIGSIZE

// initial entries could be smaller at percent or two slowdown
#ifndef BIGSIZE0
#if EDGEBITS < 30 && !defined SAVEEDGES
#define BIGSIZE0 4
#else
#define BIGSIZE0 BIGSIZE
#endif
#endif
// but they may need syncing entries
#if BIGSIZE0 == 4 && EDGEBITS > 27
#define NEEDSYNC
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#if EDGEBITS >= 30
typedef u64 offset_t;
#else
typedef u32 offset_t;
#endif

#if BIGSIZE0 > 4
typedef u64 BIGTYPE0;
#else
typedef u32 BIGTYPE0;
#endif

// node bits have two groups of bucketbits (X for big and Y for small) and a remaining group Z of degree bits
const u32 NX        = 1 << XBITS;
const u32 XMASK     = NX - 1;
const u32 NY        = 1 << YBITS;
const u32 YMASK     = NY - 1;
const u32 XYBITS    = XBITS + YBITS;
const u32 NXY       = 1 << XYBITS;
const u32 ZBITS     = EDGEBITS - XYBITS;
const u32 NZ        = 1 << ZBITS;
const u32 ZMASK     = NZ - 1;
const u32 YZBITS    = EDGEBITS - XBITS;
const u32 NYZ       = 1 << YZBITS;
const u32 YZMASK    = NYZ - 1;
const u32 YZ1BITS   = YZBITS < 15 ? YZBITS : 15;  // compressed YZ bits
const u32 NYZ1      = 1 << YZ1BITS;
const u32 MAXNZNYZ1 = NYZ1 < NZ ? NZ : NYZ1;
const u32 YZ1MASK   = NYZ1 - 1;
const u32 Z1BITS    = YZ1BITS - YBITS;
const u32 NZ1       = 1 << Z1BITS;
const u32 Z1MASK    = NZ1 - 1;
const u32 YZ2BITS   = YZBITS < 11 ? YZBITS : 11;  // more compressed YZ bits
const u32 NYZ2      = 1 << YZ2BITS;
const u32 YZ2MASK   = NYZ2 - 1;
const u32 Z2BITS    = YZ2BITS - YBITS;
const u32 NZ2       = 1 << Z2BITS;
const u32 Z2MASK    = NZ2 - 1;
const u32 YZZBITS   = YZBITS + ZBITS;
const u32 YZZ1BITS  = YZ1BITS + ZBITS;

const u32 BIGSLOTBITS   = BIGSIZE * 8;
const u32 SMALLSLOTBITS = SMALLSIZE * 8;
const u64 BIGSLOTMASK   = (1ULL << BIGSLOTBITS) - 1ULL;
const u64 SMALLSLOTMASK = (1ULL << SMALLSLOTBITS) - 1ULL;
const u32 BIGSLOTBITS0  = BIGSIZE0 * 8;
const u64 BIGSLOTMASK0  = (1ULL << BIGSLOTBITS0) - 1ULL;
const u32 NONYZBITS     = BIGSLOTBITS0 - YZBITS;
const u32 NNONYZ        = 1 << NONYZBITS;

// for p close to 0, Pr(X>=k) < e^{-n*p*eps^2} where k=n*p*(1+eps)
// see https://en.wikipedia.org/wiki/Binomial_distribution#Tail_bounds
// eps should be at least 1/sqrt(n*p/64)
// to give negligible bad odds of e^-64.

// 1/32 reduces odds of overflowing z bucket on 2^30 nodes to 2^14*e^-32
// (less than 1 in a billion) in theory. not so in practice (fails first at mean30 -n 1549)
// 3/64 works well for 29, would need to be enlarged to 1/16 for EDGEBITS=27
#ifndef BIGEPS
#define BIGEPS 3/64
#endif

// 176/256 is safely over 1-e(-1) ~ 0.63 trimming fraction
#ifndef TRIMFRAC256
#define TRIMFRAC256 176
#endif

const u32 NTRIMMEDZ  = NZ * TRIMFRAC256 / 256;

const u32 ZBUCKETSLOTS = NZ + NZ * BIGEPS;
#ifdef SAVEEDGES
const u32 ZBUCKETSIZE = NTRIMMEDZ * (BIGSIZE + sizeof(u32));  // assumes EDGEBITS <= 32
#else
const u32 ZBUCKETSIZE = ZBUCKETSLOTS * BIGSIZE0; 
#endif
const u32 TBUCKETSIZE = ZBUCKETSLOTS * BIGSIZE; 

#define likely(x)   __builtin_expect((x)!=0, 1)
#define unlikely(x) __builtin_expect((x), 0)

#endif