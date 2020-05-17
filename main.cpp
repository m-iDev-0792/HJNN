#include <iostream>
#include "Eigen/Core"
#include "HJNN.h"
#include "utility.h"
using namespace std;
using namespace Eigen;

int main() {
	vector<LayerInfo> info;
	info.emplace_back(2,LINEAR);
	info.emplace_back(2,SIGMOID);
	info.emplace_back(1,LINEAR);
	HJNN hjnn(info);
	auto data=genrate2ClassNormalData2d(-5,2,1,6,-2,1,20);
	hjnn.train(data.first,data.second,0.001,100,0.1);
	hjnn.printNetInfo();
	auto result=hjnn.predict(data.first);
	cout<<"original data:\n"<<data.first<<"\ntruth:"<<data.second<<"\npredict:"<<result;
//	if(hjnn.saveModel("normal.model"))cout<<"\nsave successfully!";
	return 0;
}
