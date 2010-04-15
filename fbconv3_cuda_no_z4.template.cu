#raw
#include <stdio.h>
#end raw

#set NF=4
#set Z=4
#set xyzw = ['x','y','z','w']


texture<float4, 1, cudaReadModeElementType> tex_float4;
__constant__ float constant[$Z][$N_FILTER_ROWS][$FILTER_W][$N_OUTPUT4S][$NF];

#define uint unsigned int

#if $IMUL_FAST
#define IMUL(a, b) __mul24(a, b)
#else
#define IMUL(a, b) a*b
#end if


extern "C" {

#for nk in xrange($N_KERNELS)

  __global__
  void cudafilter_kernel_${nk}
  (
   float4 *input
#for o in xrange($N_OUTPUT4S)
   , float4 *output$o
#end for
   )
  {

#if $PAD_SHARED_IN
    __shared__ float shared_in[$BLOCK_H][$N_FILTER_ROWS][$Z][$INPUT_BLOCK_W+1];
#else
    __shared__ float shared_in[$BLOCK_H][$N_FILTER_ROWS][$Z][$INPUT_BLOCK_W];
#end if

    // -- input/output "pointers"
    const uint in_idx =				   \
      IMUL(IMUL(blockIdx.y, $BLOCK_H), $INPUT_W) + \
      IMUL(IMUL($nk, $INPUT_W), $N_FILTER_ROWS) +  \
      IMUL(threadIdx.y, $INPUT_W) +		   \
      IMUL(blockIdx.x, $BLOCK_W) + threadIdx.x ;
    
    const uint out_idx =				\
      IMUL(IMUL(blockIdx.y, $BLOCK_H), $OUTPUT_W) +	\
      IMUL(threadIdx.y, $OUTPUT_W) +			\
      IMUL(blockIdx.x, $BLOCK_W) + threadIdx.x ;
    
    // -- XXX
    float4 input_v4;

    // -------------------------------------------------------------------------
    // -- load input to shared memory
    // -------------------------------------------------------------------------
#for nfr in xrange($N_FILTER_ROWS)
#for i in xrange($N_LOAD_ITERATIONS)
#if $i==($N_LOAD_ITERATIONS-1)
    if( (threadIdx.x + IMUL($BLOCK_W, $i)) < $INPUT_BLOCK_W )
#end if
      {
	input_v4 = tex1Dfetch(tex_float4, in_idx + IMUL($INPUT_W, $nfr) + IMUL($BLOCK_W, $i));
#for d in xrange($NF)
	shared_in[threadIdx.y][$nfr][$d][threadIdx.x + IMUL($BLOCK_W, $i)] = input_v4.$xyzw[$d];
#end for
      }
#end for
#end for
    __syncthreads();

    // -------------------------------------------------------------------------
    // -- compute dot products
    // -------------------------------------------------------------------------
    float value, weight;

#for o in xrange($N_OUTPUT4S)
#for n in xrange($NF)
    float sum${o}${n} = 0;
#end for
#end for

#for d in xrange($Z)
#for nfr in xrange($N_FILTER_ROWS)
#for i in xrange($FILTER_W)
    value = shared_in[threadIdx.y][$nfr][$d][threadIdx.x+$i];
#for o in xrange($N_OUTPUT4S)
#for n in xrange($NF)
    weight = constant[$d][$nfr][$i][$o][$n];
    sum${o}${n} += value*weight;
#end for
#end for
#end for
#end for
#end for


    // -------------------------------------------------------------------------
    // -- output results
    // -------------------------------------------------------------------------

/*     output[s_out_idx[threadIdx.y][threadIdx.x]].x += sum0; */
#for o in xrange($N_OUTPUT4S)
#for n in xrange($NF)
    output${o}[out_idx].$xyzw[$n] += sum${o}${n};
#end for
#end for

  }
#end for



}

