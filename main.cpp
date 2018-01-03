#include "NN.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	//����������ÿһ�����Ԫ��Ŀ   ���㣨784��100��10���������ݼ��Ĺ̶���ʽ
	vector<int> Layer_Neurons_nums = { 784,100,10 };

	// ������ṹ  Ȩ�� ƫִ 
	NN nn;
	nn.BeginNN(Layer_Neurons_nums);
	nn.BeginWeights(0, 0., 0.01);
	nn.Begindross(Scalar(0.05));

	//��ȡѵ���Ͳ������� 
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

	//����ģ�� 
	nn.save_model("models/model_sigmoid.xml");

	getchar();
	return 0;
}
