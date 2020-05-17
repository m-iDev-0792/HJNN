//
// Created by 何振邦 on 2020/5/12.
//

#ifndef HJNN_HJNN_H
#define HJNN_HJNN_H
#include "Layer.h"
#include <vector>
#include <iostream>
#include <fstream>
class HJNN {
public:
	//config
	std::vector<Layer> layers;
	std::vector<Eigen::MatrixXd> weights;
	std::vector<Eigen::VectorXd> biases;

	//internal state
	double learningRate;
	Eigen::VectorXd truthMinusOutput;
	Eigen::VectorXd truth;

	//construct
	HJNN(const std::vector<LayerInfo>& info);
	HJNN(const std::string& modelName);

	void forward();
	void backward();

	void train(Eigen::MatrixXd data,Eigen::MatrixXd truths,double threshold,int maxRounds,double _learningRate);
	Eigen::MatrixXd predict(const Eigen::MatrixXd &data);

	double calculateLoss();
	double calculateAverageLoss(const Eigen::MatrixXd& _output, const Eigen::MatrixXd& _truth);

	//save and load model
	bool saveModel(const std::string &modelName);
	bool loadModel(const std::string &modelName);

	//helper
	double lastLoss;
	int lossIncreaseTime;
	void printNetInfo();
	int rounddebug;
	bool debug=false;
	std::ofstream log;
protected:
};


#endif //HJNN_HJNN_H
