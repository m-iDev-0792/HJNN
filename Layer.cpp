//
// Created by 何振邦 on 2020/5/12.
//

#include "Layer.h"
Layer::Layer(int n,ActivationFunc* func){
	neuronNum=n;
	activationFunc=func;
}
Layer::Layer(const LayerInfo& info):Layer(info.neuronNum,info.func){

}
void Layer::activate() {
	out= activationFunc->calculate(in);
}
Eigen::VectorXd Layer::derivative() {
	return activationFunc->derivative(in);
}
