//
// Created by 何振邦 on 2020/5/12.
//

#ifndef HJNN_LAYER_H
#define HJNN_LAYER_H
#include "Activation.h"
class LayerInfo{
public:
	int neuronNum;
	ActivationFunc* func;
	LayerInfo(int n,ActivationFunc* f=SIGMOID){
		neuronNum=n;
		func=f;
	}
};
class Layer {
public:
	int neuronNum;
	Eigen::VectorXd in;
	Eigen::VectorXd out;
	Eigen::VectorXd backwardAccumulation;
	ActivationFunc* activationFunc;

	Layer(int n,ActivationFunc* func=SIGMOID);
	Layer(const LayerInfo& info);
	void activate();
	Eigen::VectorXd derivative();
};


#endif //HJNN_LAYER_H
