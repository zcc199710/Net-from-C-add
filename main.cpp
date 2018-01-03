#include "NN.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	//设置神经网络每一层的神经元数目   三层（784，100，10）给定数据集的固定格式
	vector<int> Layer_Neurons_nums = { 784,100,10 };

	// 神经网络结构  权重 偏执 
	NN nn;
	nn.BeginNN(Layer_Neurons_nums);
	nn.BeginWeights(0, 0., 0.01);
	nn.Begindross(Scalar(0.05));

	//获取训练和测试数据 
	Mat input, label, test_input, test_label;
	int sample_number = 200;
	nn.get_file_from_xml("data/input_label.xml", input, label, sample_number);
	nn.get_file_from_xml("data/input_label.xml", test_input, test_label, 200, 800);

	float loss_threshold = 0.5;
	nn.rates = 0.3;
	nn.output_Neurons = 2;
	nn.Activation_Functions = "sigmoid";

	nn.train(input, label, loss_threshold, true);
	nn.test(test_input, test_label);

	//保存模型 
	nn.save_model("models/model_sigmoid.xml");

	getchar();
	return 0;
}
