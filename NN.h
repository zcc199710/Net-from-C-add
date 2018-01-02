#ifndef NN_H
#define NN_H

#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class NN
{
public:
	NN() {};
	~NN() {};
	//初始化神经网络
	void BeginNN(std::vector<int> layer_neuron_num_);
	//初始化权值矩阵，调用initWeight()函数.
	void BeginWeights(int type = 0, double a = 0., double b = 0.1);
	//初始化偏置项.
	void Begindross(cv::Scalar& bias);
	//forward()：执行前向运算，包括线性运算和非线性激活，同时计算误差
	void Forward_Trans();   //同时激活函数包含在function里
	//backward()：执行反向传播，调用updateWeights()函数更新权值。
	void Backward_Trans();
	//从xml文件中从指定的列开始提取一定数目的样本和标签。默认从第0列开始读取 
	void get_file_from_xml(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);
	//Train  训练函数（小于设定的阙值 作为迭代终止条件）
	void train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);
	//经过train 之后 就得到了一个超级矩阵（权值矩阵模型）
	//Test  测试函数  测试没有训练过的数据 看看正确率为多少  其中调用了下面的predite_one函数 远离还是一样的（代替了一次前向传播）
	void test(cv::Mat &input, cv::Mat &target_);
	//Predict  仅仅一个样例  与上面的test照应
	int forward(cv::Mat &input);
	//Predict  多余一个的样例 循环调用 predict_one()
	vector<int> predict(cv::Mat &input);
	//训练过一次之后要把模型给保存起来，要不然下次还要训练
	void save_model(std::string filename);
	//加载模型
	void load_model(std::string filename);

protected:
	//初始化矩阵
	void BeginWeight(cv::Mat &dst, int type, double a, double b);
	//激活函数
	cv::Mat activationFunction(cv::Mat &x, std::string func_type);
	//计算delta误差
	void deltaError();
	//更新权重
	void update_Weights();

public:
	vector<int> Layer_Neurons_nums;  //每一层神经元数目（layerneuronnum）
	string Activation_Functions = "sigmoid"; //前向传播中 的 激活函数
	int output_Neurons = 10;      //输出层神经元的个数
	float rates;        //学习率
	float accuracy = 0.;
	vector<double> loss_vec;
	float fine_tune_factor = 1.01;

protected:
	//层（layer）权值矩阵（weights）偏置项（bias）
	vector<Mat> layers;
	vector<Mat> weights;
	vector<Mat> dross;
	vector<Mat> delta_err;
	Mat output_error;
	Mat target;
	Mat board;
	float loss;

};
	
#endif  NN_H