//
// Created by 何振邦 on 2020/5/12.
//

#include "HJNN.h"
#include <ctime>
#include <chrono>
#include <sstream>
#include <iomanip>
HJNN::HJNN(const std::vector<LayerInfo> &info) {
	for(auto& l:info)layers.push_back(Layer(l));
	srand((unsigned)time(nullptr));//set seed to create random matrix for weights
	for(int i=0;i<layers.size()-1;++i){
		weights.push_back(Eigen::MatrixXd::Random(layers[i+1].neuronNum,layers[i].neuronNum));
		biases.push_back(Eigen::VectorXd::Zero(layers[i+1].neuronNum));
	}
}
HJNN::HJNN(const std::string &modelName) {
	if(!loadModel(modelName))std::cout<<"loading model "<<modelName<<" failed, net might not be usable!";
}
void HJNN::forward() {
	for (int i = 0; i < layers.size() - 1; ++i) {
		layers[i + 1].in = weights[i] * layers[i].out + biases[i];
		layers[i + 1].activate();
	}
}

double HJNN::calculateLoss() {
	truthMinusOutput = truth - layers.back().out;
	return truthMinusOutput.squaredNorm()/truthMinusOutput.size();
}

void HJNN::backward() {
	//Network layout brief
	//layer[i-1]--> w[i-1]--> layer[i]--> w[i]--> layer[i+1]

	//------------------------------------------------------
	// 1. Calculate backward accumulation
	//backwardAccumulation for last layer(output layer)
	const auto &output = layers.back().out;
	//here Loss=∑(y-y_predict)^2/N
	//thus dLoss/dy_predict=2/N*(y-y_predict)*(-1)
	auto lossDerivative = (-2.0 / output.size()) * truthMinusOutput.array();
	layers.back().backwardAccumulation = lossDerivative * layers.back().derivative().array();
	if(debug){
		log<<"------round"<<rounddebug<<" start---------\n";
		log<<"[last layer]\ntruth="<<truth<<", output="<<output<<", last layer accum="<<layers.back().backwardAccumulation<<std::endl;
	}

	//backwardAccumulation for other layer
	for (int i = layers.size() - 2;
	     i > 0; --i) {//Note. we don't need to calculate backwardAccumulation layer[0](input layer),that's why i>0
		auto temp = (weights[i].transpose() * layers[i + 1].backwardAccumulation).array();
		layers[i].backwardAccumulation = temp * layers[i].derivative().array();
		if(debug)log<<"[layer"<<i<<"] data\n"<<"accum from last layer=\n"<<temp<<"\nlayer "<<i<<"deri=\n"<<layers[i].derivative()<<"\n";
	}

	//------------------------------------------------------
	// 2. update weight
	if(debug)std::cout<<"[delta weight and bias]\n";
	for (int i = 0; i < weights.size(); ++i) {
		weights[i] -= learningRate * layers[i + 1].backwardAccumulation * layers[i].out.transpose();
		biases[i] -= learningRate * layers[i + 1].backwardAccumulation;
		if(debug)log<<"delta weight"<<i<<"=\n"<<(learningRate * layers[i + 1].backwardAccumulation * layers[i].out.transpose())<<
		"\ndelta bias"<<i<<"=\n"<<(learningRate * layers[i + 1].backwardAccumulation)<<std::endl;
	}
	if(debug)log<<"------round"<<rounddebug<<" end---------"<<std::endl;
}
void HJNN::train(Eigen::MatrixXd data,Eigen::MatrixXd truths,double threshold,int maxRounds,double _learningRate){
	if(data.rows()!=layers[0].neuronNum){
		std::cerr<<"data dimension does not match the input neuron number of network! "<<std::endl;
		return;
	}
	lastLoss=0;
	lossIncreaseTime=0;
	if(debug){
		auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		std::stringstream ss;
		ss << std::put_time(std::localtime(&t), "%F %T");
		std::string str = "log_"+ss.str()+".txt";
		log.open(str);
	}

	int dataNum=data.cols();
	int rounds=0;
	learningRate=_learningRate;
	double loss=0;
	Eigen::ArrayXd lossList=Eigen::ArrayXd::Zero(dataNum);
	while(rounds<maxRounds){
		int index=rounds++%dataNum;
		rounddebug=rounds;
		//set input and truth
		layers[0].out=data.col(index);
		truth=truths.col(index);
		//do a round
		forward();
		loss=lossList(index)=calculateLoss();
		std::cout<<"round "<<rounds<<",loss "<<loss<<std::endl;
		if(loss>lastLoss&&loss>threshold){
			if(++lossIncreaseTime>10){
				std::cerr<<"loss continue increase, net probably will not converge, stop training!"<<std::endl;
				break;
			}
		}else lossIncreaseTime=0;
		lastLoss=loss;
		if(loss>1e100){
			std::cerr<<"loss is too large, net probably will not converge, stop training!"<<std::endl;
			break;
		}
//		if(loss>threshold)backward();
		backward();
		//if average loss is less than threshold then terminate training
//		if(index>=dataNum-1){
//			double avgLoss=lossList.sum()/dataNum<threshold;
//			if(avgLoss<threshold)break;
//			if(index==dataNum-1)std::cout<<"one iteration completed, average loss="<<avgLoss<<std::endl;
//		}
	}
	std::cout<<"Training completed! "<<rounds<<" rounds trained, average loss="<<lossList.sum()/dataNum<<std::endl;
	if(debug)log.close();
}
Eigen::MatrixXd HJNN::predict(const Eigen::MatrixXd &data) {
	Eigen::MatrixXd result=data;
	for(int i=0;i<weights.size();++i){
		result=weights[i]*result +biases[i]*Eigen::MatrixXd::Ones(1,result.cols());
		result=layers[i+1].activationFunc->calculate(result);
	}
	return result;
}
double HJNN::calculateAverageLoss(const Eigen::MatrixXd& _output, const Eigen::MatrixXd& _truth){
	Eigen::ArrayXd diff=_truth-_output;
	return (diff*diff).mean();
}
void HJNN::printNetInfo() {
	std::cout<<"/------Net Info------/\n";
	std::cout<<layers.size()<<" layers in the net:\n";
	for(int i=0;i<layers.size();++i){
		std::cout<<"layer "<<i<<": "<<layers[i].in.size()<<" neuron in, "<<layers[i].out.size()<<" neuron out\n";
	}
	std::cout<<"\nweights and biases\n";
	for(int i=0;i<weights.size();++i){
		std::cout<<"weight "<<i<<"\n"<<weights[i]<<"\nbias "<<i<<"\n"<<biases[i]<<"\n\n";
	}
	std::cout<<"/-----Net Info End----/\n";
}
#define TCV(x) reinterpret_cast<char*>(&x) //[T]o [C]har* from [V]alue
#define TCP(x) reinterpret_cast<char*>(x)  //[T]o [C]har* from [P]ointer
bool HJNN::saveModel(const std::string &modelName) {
	std::ofstream model(modelName,std::ios::binary|std::ios::out);
	if(!model.is_open()){
		std::cerr<<"can't create file "<<modelName<<std::endl;
		return false;
	}
	//save layer info
	int layerNum=layers.size();
	model.write(TCV(layerNum), sizeof(layerNum));
	for(int i=0;i<layerNum;++i){
		int layerInfo[2]={layers[i].neuronNum,layers[i].activationFunc->getType()};
		model.write(TCP(layerInfo), sizeof(int) * 2);
	}
	//save weight
	int weightNum=weights.size();
	model.write(TCV(weightNum), sizeof(weightNum));
	for(int i=0;i<weightNum;++i){
		int weightSize[2]={static_cast<int>(weights[i].rows()),static_cast<int>(weights[i].cols())};
		model.write(TCP(weightSize), sizeof(int) * 2);
		//data are packed in column majority order
		model.write(TCP(weights[i].data()), sizeof(double) * weightSize[0] * weightSize[1]);
	}
	//save bias
	int biasNum=biases.size();
	model.write(TCV(biasNum),sizeof(biasNum));
	for(int i=0;i<biasNum;++i){
		int size=biases[i].size();
		model.write(TCV(size),sizeof(size));
		model.write(TCP(biases[i].data()),sizeof(double)*size);
	}
	return true;
}
bool HJNN::loadModel(const std::string &modelName) {
	std::ifstream model(modelName,std::ios::binary|std::ios::in);
	if(!model.is_open()){
		std::cerr<<"can't load model from file "<<modelName<<std::endl;
		return false;
	}
	//load layer info
	int layerNum;
	if(!model.read(TCV(layerNum), sizeof(int)))return false;
	layers.clear();
	std::cout<<"["<<layerNum<<" layers found]"<<std::endl;
	for(int i=0;i<layerNum;++i){
		int layerInfo[2]={0};
		if(!model.read(TCP(layerInfo), sizeof(int) * 2))return false;
		std::cout<<"layer"<<i<<" has "<<layerInfo[0]<<" neuron(s), func type="<<layerInfo[1]<<std::endl;
		layers.push_back(Layer(LayerInfo(layerInfo[0],ACTIVATION_LIST[layerInfo[1]])));
	}
	//load weight
	int weightNum=weights.size();
	if(!model.read(TCV(weightNum), sizeof(int)))return false;
	std::cout<<"["<<weightNum<<" weight(s) found]"<<std::endl;
	weights.clear();
	for(int i=0;i<weightNum;++i){
		int weightSize[2]={0};
		if(!model.read(TCP(weightSize), sizeof(int) * 2))return false;
		const int row=weightSize[0];const int col=weightSize[1];
		std::cout<<"weight"<<i<<" size="<<row<<"x"<<col<<std::endl;
		double *wdata=new double[row*col];
		//data are packed in column majority order
		if(!model.read(TCP(wdata), sizeof(double) * row * col))return false;
		Eigen::MatrixXd w=Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> >(wdata, row, col);
		weights.push_back(w);
		delete [] wdata;
	}
	//load bias
	int biasNum=biases.size();
	if(!model.read(TCV(biasNum),sizeof(biasNum)))return false;
	std::cout<<"["<<biasNum<<" bias(es) found]"<<std::endl;
	biases.clear();
	for(int i=0;i<biasNum;++i){
		int size;
		if(!model.read(TCV(size),sizeof(size)))return false;
		std::cout<<"bias"<<i<<" size="<<size<<std::endl;
		double *bdata=new double[size];
		if(!model.read(TCP(bdata),sizeof(double)*size))return false;
		Eigen::MatrixXd b=Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> >(bdata, size, 1);
		biases.push_back(b);
		delete [] bdata;
	}
	std::cout<<"load model from file "<<modelName<<" successfully"<<std::endl;
	return true;
}