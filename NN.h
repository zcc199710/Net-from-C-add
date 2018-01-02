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
	//��ʼ��������
	void BeginNN(std::vector<int> layer_neuron_num_);
	//��ʼ��Ȩֵ���󣬵���initWeight()����.
	void BeginWeights(int type = 0, double a = 0., double b = 0.1);
	//��ʼ��ƫ����.
	void Begindross(cv::Scalar& bias);
	//forward()��ִ��ǰ�����㣬������������ͷ����Լ��ͬʱ�������
	void Forward_Trans();   //ͬʱ�����������function��
	//backward()��ִ�з��򴫲�������updateWeights()��������Ȩֵ��
	void Backward_Trans();
	//��xml�ļ��д�ָ�����п�ʼ��ȡһ����Ŀ�������ͱ�ǩ��Ĭ�ϴӵ�0�п�ʼ��ȡ 
	void get_file_from_xml(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);
	//Train  ѵ��������С���趨����ֵ ��Ϊ������ֹ������
	void train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);
	//����train ֮�� �͵õ���һ����������Ȩֵ����ģ�ͣ�
	//Test  ���Ժ���  ����û��ѵ���������� ������ȷ��Ϊ����  ���е����������predite_one���� Զ�뻹��һ���ģ�������һ��ǰ�򴫲���
	void test(cv::Mat &input, cv::Mat &target_);
	//Predict  ����һ������  �������test��Ӧ
	int forward(cv::Mat &input);
	//Predict  ����һ�������� ѭ������ predict_one()
	vector<int> predict(cv::Mat &input);
	//ѵ����һ��֮��Ҫ��ģ�͸�����������Ҫ��Ȼ�´λ�Ҫѵ��
	void save_model(std::string filename);
	//����ģ��
	void load_model(std::string filename);

protected:
	//��ʼ������
	void BeginWeight(cv::Mat &dst, int type, double a, double b);
	//�����
	cv::Mat activationFunction(cv::Mat &x, std::string func_type);
	//����delta���
	void deltaError();
	//����Ȩ��
	void update_Weights();

public:
	vector<int> Layer_Neurons_nums;  //ÿһ����Ԫ��Ŀ��layerneuronnum��
	string Activation_Functions = "sigmoid"; //ǰ�򴫲��� �� �����
	int output_Neurons = 10;      //�������Ԫ�ĸ���
	float rates;        //ѧϰ��
	float accuracy = 0.;
	vector<double> loss_vec;
	float fine_tune_factor = 1.01;

protected:
	//�㣨layer��Ȩֵ����weights��ƫ���bias��
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