//
// Created by 何振邦 on 2020/5/14.
//

#ifndef HJNN_UTILITY_H
#define HJNN_UTILITY_H

#include "Eigen/Core"
#include <random>
#include <utility>

class NormalDistrbution2D{
public:
	double x;
	double y;
	double delta;
	NormalDistrbution2D(double _x,double _y,double _delta):gen(std::random_device()()),dx(_x,_delta),dy(_y,_delta){
		x=_x;y=_y;delta=_delta;
	}
	void genrate(double& rx,double& ry){
		rx=dx(gen);
		ry=dy(gen);
	}

private:
	std::mt19937 gen;
	std::normal_distribution<double> dx;
	std::normal_distribution<double> dy;
};
// <data,tag>
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
genrate2ClassNormalData2d(double x1, double y1, double delta1, double x2, double y2, double delta2, int dataNum);

#endif //HJNN_UTILITY_H
