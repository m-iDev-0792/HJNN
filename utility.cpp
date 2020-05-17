//
// Created by 何振邦 on 2020/5/14.
//

#include "utility.h"

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
genrate2ClassNormalData2d(double x1, double y1, double delta1, double x2, double y2, double delta2, int dataNum){
	Eigen::MatrixXd data=Eigen::MatrixXd::Zero(2,dataNum);
	Eigen::MatrixXd tag=Eigen::MatrixXd::Zero(1,dataNum);
	NormalDistrbution2D positive(x1,y1,delta1),negative(x2,y2,delta2);
	for(int i=0;i<dataNum;++i){
		if(i&1){
			positive.genrate(data(0,i),data(1,i));
			tag(0,i)=1;
		}else{
			negative.genrate(data(0,i),data(1,i));
			tag(0,i)=-1;
		}
	}
	return std::make_pair(data,tag);
}