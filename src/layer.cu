
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include "externel_headers.cuh"
#include "layer.cuh"

template<class computeType>
void layer<computeType>::allocIOMemory()
{
  int inSize  = batch_size * channels_in * width_in * height_in * sizeof( computeType );
  int outSize = batch_size * channels_out * width_out * height_out * sizeof( computeType );

  if( first_pr ) {
	HandleCudaError( cudaMalloc( &srcPtr,     inSize ) );
	HandleCudaError( cudaMalloc( &srcDiffPtr, inSize ) );
  }
  HandleCudaError( cudaMalloc( &dstPtr, outSize ) );
  HandleCudaError( cudaMalloc( &dstDiffPtr, outSize ) );

  if( last_pr ) {
    switch( learning_problem ) {
      case classification:
    	HandleCudaError( cudaMalloc( &tGroundTruth, batch_size * sizeof( computeType ) ) );
        break;
      case regression:
    	HandleCudaError( cudaMalloc( &tGroundTruth, outSize ) );
        break;
    }
  }
}

template<class computeType>
void layer<computeType>::freeIOMemory()
{
  if( first_pr ) {
	if( srcPtr ) {
	  HandleCudaError( cudaFree( srcPtr ) );
	}
	if( srcDiffPtr ) {
      HandleCudaError( cudaFree( srcDiffPtr ) );
	}
  }
  if( dstPtr ) {
	HandleCudaError( cudaFree( dstPtr ) );
  }
  if( dstDiffPtr ) {
	HandleCudaError( cudaFree( dstDiffPtr ) );
  }
  if( last_pr ) {
	if( tGroundTruth ) {
      HandleCudaError( cudaFree( tGroundTruth ) );
	}
  }
}

template<class computeType>
bool layer<computeType>::adaptable_weight()
{
  return adaptable_weight_pr;
}

template<class computeType>
__global__ void computeDifferenceClassificationKernel( computeType* prediction, computeType* groundTruth,
		                                               int batch_size, int nr_classes, double scale )
{
  int gidx = blockDim.x * blockIdx.x + threadIdx.x;
  if( gidx < batch_size ) {
	int position = (int) groundTruth[ gidx ];
	prediction[ gidx*nr_classes + position ] -= (computeType) 1.0;
	for( int i = 0; i<nr_classes; i++ ) {
	  prediction[ gidx*nr_classes + i ] *= scale;
	}
  }
}

template<class computeType>
__global__ void computeDifferenceRegressionKernel( computeType* prediction, computeType* groundTruth,
		                                           int batch_size, int nr_classes, double scale )
{
  int gidx = blockDim.x * blockIdx.x + threadIdx.x;
  if( gidx < batch_size * nr_classes ) {
    prediction[ gidx ] = scale * ( groundTruth[ gidx ] - prediction[ gidx ] );
  }
}

template<class computeType>
void layer<computeType>::compute_difference()
{
  int nBlocks    = 0;
  int nr_classes = channels_out * width_out * height_out;

  HANDLE_CUDA_ERROR( cudaMemcpy( dstDiffPtr, dstPtr, nr_classes * batch_size * sizeof( computeType ) ) );

  if( learning_problem == classification ) {
    nBlocks = ( batch_size + tperblock - 1 ) / tperblock;
    computeDifferenceClassificationKernel<computeType><<<nBlocks,tperblock>>>( dstDiffPtr, tGroundTruth,
    		                                                                   batch_size, nr_classes, scale_diff );
  } else {
	nBlocks = ( batch_size * nr_classes + tperblock - 1 ) / tperblock;
	computeDifferenceRegressionKernel<computeType><<<nBlocks,tperblock>>>( dstDiffPtr, tGroundTruth,
                                                                           batch_size, nr_classes, scale_diff );
  }
}

template<class computeType>
bool layer<computeType>::first()
{
  return first_pr;
}

template<class computeType>
bool layer<computeType>::last()
{
  return last_pr;
}

template<class computeType>
layer<computeType>::layer()
{
  srcPtr       = NULL;
  dstPtr       = NULL;
  srcDiffPtr   = NULL;
  dstDiffPtr   = NULL;
  tGroundTruth = NULL;

  channels_in  = -1;
  width_in     = -1;
  height_in    = -1;

  channels_out = -1;
  width_out    = -1;
  height_out   = -1;

  tperblock           = 512;
  batch_size          = -1;
  adaptable_weight_pr = false;

  learning_problem    = classification;
  last_pr             = false;
  first_pr            = false;

  scale_diff          = 5e-4;
}
