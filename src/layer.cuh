/*
 * layer.cuh
 *
 *  Created on: May 31, 2018
 *      Author: hondo
 */

#ifndef LAYER_CUH_
#define LAYER_CUH_
#include "externel_headers.cuh"

template<class computeType>
class layer {
private:
  /** problem settings **/
  LEARNING_PROBLEM learning_problem;

  /** in/out tensor dims **/
  int  batch_size;
  int  channels_in;
  int  width_in;
  int  height_in;
  int  channels_out;
  int  width_out;
  int  height_out;


  /** neural net position/type data **/
  bool first_pr;
  bool last_pr;
  bool adaptable_weight_pr;

  /** launch configuration **/
  int tperblock;

  /** delta scaling factor **/
  double scale_diff;

  void compute_difference();
  void setTensorDims( int b_size, int c_in,
		              int w_in,   int h_in,
		              int c_out,  int w_out,
		              int h_out );
  void allocIOMemory();
  void freeIOMemory();
public:
  /** IO memory fields **/
  computeType* srcPtr;
  computeType* dstPtr;
  computeType* srcDiffPtr;
  computeType* dstDiffPtr;
  computeType* tGroundTruth;

  /** obligatory functionalities **/
  virtual void forward()=0;
  virtual void backward()=0;

  bool first();
  bool last();
  bool adaptable_weight();
  void threadsPerBlock( int t_per_block );
  layer();
  virtual ~layer() { freeIOMemory(); };
};

#endif /* LAYER_CUH_ */
