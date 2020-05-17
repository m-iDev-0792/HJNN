//
// Created by 何振邦 on 2020/5/12.
//

#ifndef HJNN_ACTIVATION_H
#define HJNN_ACTIVATION_H

#include "Eigen/Core"
#include <cmath>
class ActivationFunc{
public:
	virtual int getType()=0;
	virtual double calculate(const double &x)=0;
	virtual Eigen::MatrixXd calculate(const Eigen::MatrixXd& x)=0;
	virtual double derivative(const double &x)=0;
	virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd& x)=0;
};
class Sigmoid:public ActivationFunc{
public:
	int getType()override {
		return 1;
	}
	double calculate(const double &x) override {
		return 1.0/(1.0+ std::exp(-x));
	}
	Eigen::MatrixXd calculate(const Eigen::MatrixXd& x) override{
		return (1.0/(1.0+(-1*x.array()).exp())).matrix();
	}
	double derivative(const double &x) override{
		auto fx= calculate(x);
		return fx*(1-fx);
	}
	Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override{
		auto fx=(1.0/(1.0+(-1*x.array()).exp()));
		return (fx*(1-fx)).matrix();
	}
};
class Tanh:public ActivationFunc{
public:
	int getType()override {
		return 2;
	}
	double calculate(const double &x) override {
		double ex,emx;
		ex=std::exp(x),emx=std::exp(-x);
		return (ex-emx)/(ex+emx);
	}
	Eigen::MatrixXd calculate(const Eigen::MatrixXd& x) override{
		auto ex=x.array().exp();
		auto emx=(-1*x.array()).exp();
		return ((ex-emx)/(ex+emx)).matrix();
	}
	double derivative(const double &x) override{
		double ex,emx;
		ex=std::exp(x),emx=std::exp(-x);
		auto deno=ex+emx;
		return 4.0/(deno*deno);
	}
	Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override{
		auto ex=x.array().exp();
		auto emx=(-1*x.array()).exp();
		auto deno=ex+emx;
		return (4.0/(deno*deno)).matrix();
	}
};
class ReLu:public ActivationFunc{
public:
	int getType()override {
		return 3;
	}
	double calculate(const double &x) override {
		return x>0?x:0;
	}
	Eigen::MatrixXd calculate(const Eigen::MatrixXd& x) override{
		return x.array().max(0.0).matrix();
	}
	double derivative(const double &x) override{
		return x>0?1:0;
	}
	Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override{
		return (x.array()>0.0).cast<double>().matrix();
	}
};
class LeakyReLu:public ActivationFunc{
public:
	double alpha;//usually between 0.1 and 0.3

	LeakyReLu(double _alpha=0.2):alpha(_alpha){}
	int getType()override {
		return 4;
	}
	double calculate(const double &x) override {
		return x>0?x:alpha*x;
	}
	Eigen::MatrixXd calculate(const Eigen::MatrixXd& x) override{
		return x.array().max(alpha*x.array()).matrix();
	}
	double derivative(const double &x) override{
		return x>0?1:alpha;
	}
	Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override{
		return ((x.array()>0.0).cast<double>()*(1-alpha)+Eigen::ArrayXXd::Constant(x.rows(),x.cols(),alpha)).matrix();
	}
};
class Softplus:public ActivationFunc{
public:
	int getType()override {
		return 5;
	}
	double calculate(const double &x) override {
		return std::log(1+std::exp(x));
	}
	Eigen::MatrixXd calculate(const Eigen::MatrixXd& x) override{
		return (x.array().exp()+1.0).log().matrix();
	}
	double derivative(const double &x) override{
		return 1.0/(1.0+ std::exp(-x));
	}
	Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override{
		return (1.0/(1.0+(-1*x.array()).exp())).matrix();
	}
};
class SoftMax:public ActivationFunc{
public:
	int getType()override {
		return 6;
	}
	double calculate(const double &x) override {
		return 1.0;
	}
	Eigen::MatrixXd calculate(const Eigen::MatrixXd& x) override{
		auto ex=x.array().exp();
		auto exsum=Eigen::VectorXd::Ones(ex.rows())*ex.colwise().sum().matrix();
		return (ex/exsum.array()).matrix();
	}
	double derivative(const double &x) override{
		return -1.0;
	}
	Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override{
		auto ex=x.array().exp();
		auto exsum=Eigen::VectorXd::Ones(ex.rows())*ex.colwise().sum().matrix();
		auto fx=ex/exsum.array();
		return (fx*(1.0-fx)).matrix();
	}
};
class Linear:public ActivationFunc{
public:
	int getType()override {
		return 0;
	}
	double calculate(const double &x) override {
		return x;
	}
	Eigen::MatrixXd calculate(const Eigen::MatrixXd& x) override{
		return x;
	}
	double derivative(const double &x) override{
		return 1.0;
	}
	Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override{
		return Eigen::MatrixXd::Constant(x.rows(),x.cols(),1.0);
	}
};
static ActivationFunc* SIGMOID=new Sigmoid();
static ActivationFunc* TANH=new Tanh();
static ActivationFunc* LINEAR=new Linear();
static ActivationFunc* RELU=new ReLu();
static ActivationFunc* LEAKY_RELU=new LeakyReLu();
static ActivationFunc* SOFTPLUS=new Softplus();
static ActivationFunc* SOFTMAX=new SoftMax();
static ActivationFunc* ACTIVATION_LIST[]={LINEAR,SIGMOID,TANH,RELU,LEAKY_RELU,SOFTPLUS,SOFTMAX};
#endif //HJNN_ACTIVATION_H
