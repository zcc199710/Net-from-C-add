#include "NN.h"

//��ʼ��������  ����ÿһ��ľ���ÿһ��Ȩֵ�����ÿһ��ƫ�þ���
void NN::BeginNN(vector<int> layer_neuron_num_)
{
	Layer_Neurons_nums = layer_neuron_num_;
	//enerate every layer.
	layers.resize(Layer_Neurons_nums.size());
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i].create(Layer_Neurons_nums[i], 1, CV_32FC1);
	}
	cout << "Generate layers, successfully!" << endl;
	//Generate every weights matrix and bias
	weights.resize(layers.size() - 1);
	dross.resize(layers.size() - 1);
	for (int i = 0; i < (layers.size() - 1); ++i)
	{
		weights[i].create(layers[i + 1].rows, layers[i].rows, CV_32FC1);
		//bias[i].create(layer[i + 1].rows, 1, CV_32FC1);
		dross[i] = cv::Mat::zeros(layers[i + 1].rows, 1, CV_32FC1);
	}
	cout << "Generate weights matrices and bias, successfully!" << endl;
	cout << "Begin NN, over !" << endl;
}

//Ȩֵ��ʼ������initWeights()����initWeight()��������ʵ���ǳ�ʼ��һ���Ͷ��������
void NN::BeginWeight(Mat &dst, int type, double a, double b)
{
	if (type == 0)
	{
		randn(dst, a, b);
	}
	else
	{
		randu(dst, a, b);
	}
}
//initialise the weights matrix.
void NN::BeginWeights(int type, double a, double b)
{
	//Initialise weights cv::Matrices and bias
	for (int i = 0; i < weights.size(); ++i)
	{
		BeginWeight(weights[i], 0, 0., 0.1);
	}
}
//ƫ�ó�ʼ���Ǹ����е�ƫ�ø���ͬ��ֵ��������Scalar������������ֵ��
void NN::Begindross(Scalar& bias_)
{
	for (int i = 0; i < dross.size(); i++)
	{
		dross[i] = bias_;
	}
}

Mat  NN::sigmoid(Mat &x)
{
	Mat exp_x, fx;
	exp(-x, exp_x);
	fx = 1.0 / (1.0 + exp_x);
	return fx;
}

//΢�� ���� ��Derivative�� �����   
Mat NN::derivativeFunction(Mat& fx, string func_type)
{
	Mat dx;
	if (func_type == "sigmoid")
	{
		dx = sigmoid(fx).mul((1 - sigmoid(fx)));
	}
	return dx;
}

//�����������Ŀ�꺯��������������ƽ���͵ľ�ֵ��Ϊ��Ҫ��С����Ŀ�꺯�����ڷ��򴫲������ã�
void NN::calcLoss(Mat &output, Mat &target, Mat &output_error, float &loss)
{
	if (target.empty())
	{
		cout << "Can't find the target Matrix" << endl;
		return;
	}
	output_error = target - output;
	Mat err_sqrare;
	pow(output_error, 2., err_sqrare);
	Scalar err_sqr_sum = sum(err_sqrare);
	loss = err_sqr_sum[0] / (float)(output.rows);
}
//ǰ�򴫲�
void NN::Forward_Trans()
{
	for (int i = 0; i < Layer_Neurons_nums.size() - 1; ++i)
	{
		Mat product = weights[i] * layers[i] + dross[i];   //������˵Ľ�� 
		// ���ü���� �����һ�㶼�Ƿ����Ժ����������ڵļ�ֵ���Ǹ��������ṩ�����Խ�ģ����
		layers[i + 1] = activationFunction(product, Activation_Functions);  // ���ü���� �����һ�㶼�Ƿ����Ժ����������ڵļ�ֵ���Ǹ��������ṩ�����Խ�ģ����
	}
}

//����� func_type��ֵΪ����Ҫѡ��ļ����
Mat NN::activationFunction(Mat &x, string func_type)
{
	Activation_Functions = func_type;
	Mat fx;
	if (func_type == "sigmoid")
	{
		fx = sigmoid(x);
	}
	return fx;
}

//���򴫲�(����� �� Ѱ�����Ž� ����Ȩ�� )
void NN::Backward_Trans()
{
	calcLoss(layers[layers.size() - 1], target, output_error, loss);
	deltaError();  
	update_Weights();
}

//����delta��    ��Ҫע��ľ��Ǽ����ʱ�����������ز�ļ��㹫ʽ�ǲ�һ���ġ�
void NN::deltaError()
{
	delta_err.resize(layers.size() - 1);
	for (int i = delta_err.size() - 1; i >= 0; i--)
	{
		delta_err[i].create(layers[i + 1].size(), layers[i + 1].type());
		Mat dx = derivativeFunction(layers[i + 1], Activation_Functions);   //�� �� ΢�ֹ�ʽ ��
		//����㣨Output layer�� delta error
		if (i == delta_err.size() - 1)
		{
			delta_err[i] = dx.mul(output_error);
		}
		else  //���ز㣨Hidden layer�� delta error
		{
			Mat weight = weights[i];
			Mat weight_t = weights[i].t();
			Mat delta_err_1 = delta_err[i];
			delta_err[i] = dx.mul((weights[i + 1]).t() * delta_err[i + 1]);
		}
	}
}

//����Ȩ��
void NN::update_Weights()
{
	for (int i = 0; i < weights.size(); ++i)
	{
		Mat delta_weights = rates * (delta_err[i] * layers[i].t());
		weights[i] = weights[i] + delta_weights;
	}
}

//Train  �������������һ��ԭ�� ����һ����������һ�����о�����Ϊ����
//Train  ����loos��ֵ��Ϊ��ֹ����
void NN::train(Mat input, Mat target_, float loss_threshold, bool draw_loss_curve)
{
	if (input.empty())
	{
		cout << "Input is empty!,please put a true imput " << endl;
		return;
	}

	cout << "Train,begain! ..." <<endl;

	Mat sample;
	if (input.rows == (layers[0].rows) && input.cols == 1)
	{
		target = target_;
		sample = input;
		layers[0] = sample;
		Forward_Trans();
		int num_of_train = 0;
		while (loss > loss_threshold)
		{
			Backward_Trans();
			Forward_Trans();
			num_of_train++;
			if (num_of_train % 500 == 0)
			{
				cout << "Train " << num_of_train << " times" << endl;
				cout << "Loss: " << loss << endl;
			}
		}
		cout << endl << "Train " << num_of_train << " times" << endl;
		cout << "Loss is : " << loss << endl;
		cout << "Train sucessfully!.." << endl;
	}
	else if (input.rows == (layers[0].rows) && input.cols > 1)
	{
		double batch_loss = loss_threshold + 0.01;
		int epoch = 0;
		while (batch_loss > loss_threshold)
		{
			batch_loss = 0.;
			for (int i = 0; i < input.cols; ++i)
			{
				target = target_.col(i);
				sample = input.col(i);
				layers[0] = sample;

				Forward_Trans();
				Backward_Trans();
				batch_loss += loss;
			}

			loss_vec.push_back(batch_loss);

			if (loss_vec.size() >= 2 && draw_loss_curve)
			{
				//draw_curve(board, loss_vec);
			}
			epoch++;
			if (epoch % output_Neurons == 0)
			{
				cout << "Number of epoch: " << epoch << endl;
				cout << "Loss sum: " << batch_loss << endl;
			}
			if (epoch % 100 == 0)
			{
				rates *= fine_tune_factor;
			}
		}
		cout << endl << "Number of epoch: " << epoch << endl;
		cout << "Loss sum is : " << batch_loss << endl;
		cout << "Train sucessfully!..." << endl;
	}
	else
	{
		cout << "Rows of input don't cv::Match the number of input!" << endl;
	}
}

//Test   ����
void NN::test(Mat &input, Mat &target_)
{
	if (input.empty())
	{
		cout << "Input is empty!" << endl;
		return;
	}
	cout << endl << "Predict,begain!..." << endl;

	if (input.rows == (layers[0].rows) && input.cols == 1)
	{
		int predict_number = forward(input);

		cv::Point target_maxLoc;
		minMaxLoc(target_, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
		int target_number = target_maxLoc.y;

		cout << "Predict is : " << predict_number << endl;
		cout << "Target is :  " << target_number << endl;
		cout << "Loss is : " << loss << endl;
	}
	else if (input.rows == (layers[0].rows) && input.cols > 1)
	{
		double loss_sum = 0;
		int right_num = 0;
		cv::Mat sample;
		for (int i = 0; i < input.cols; ++i)
		{
			sample = input.col(i);
			int predict_number = forward(sample);
			loss_sum += loss;

			target = target_.col(i);
			Point target_maxLoc;
			minMaxLoc(target, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
			int target_number = target_maxLoc.y;

			cout << "Test sample: " << i << "   " << "Predict: " << predict_number << endl;
			cout << "Test sample: " << i << "   " << "Target:  " << target_number << endl << endl;
			if (predict_number == target_number)
			{
				right_num++;
			}
		}
		accuracy = (double)right_num / input.cols;
		cout << "Loss sum: " << loss_sum << endl;
		cout << "accuracy: " << accuracy << endl;
	}
	else
	{
		cout << "Rows of input don't cv::Match the number of input!" << endl;
		return;
	}
}

//Predict_one   
int NN::forward(Mat &input)
{
	if (input.empty())
	{
		cout << "Input is empty!" <<endl;
		return -1;
	}

	if (input.rows == (layers[0].rows) && input.cols == 1)
	{
		layers[0] = input;
		Forward_Trans();

		Mat layer_out = layers[layers.size() - 1];
		Point predict_maxLoc;
		//ǰ�򴫲��õ����һ�������layerout��Ȼ���layerout����ȡ���ֵ��λ�ã�������λ�õ�y���ꡣ
		minMaxLoc(layer_out, NULL, NULL, NULL, &predict_maxLoc, cv::noArray());//opencv��ĺ���
		return predict_maxLoc.y;
	}
	else
	{
		cout << "Please give one sample alone and ensure input.rows = layer[0].rows" << endl;
		return -1;
	}
}

//Predict  ѭ���ĵ�����һ������
vector<int> NN::predict(Mat &input)
{
	Mat sample;
	if (input.rows == (layers[0].rows) && input.cols > 1)
	{
		vector<int> predicted_labels;
		for (int i = 0; i < input.cols; ++i)
		{
			sample = input.col(i);
			int predicted_label = forward(sample);
			predicted_labels.push_back(predicted_label);
			return predicted_labels;
		}
	}
}

// ģ�Ͱ���  ������Ԫ��Ŀ ����Ȩ�� ����� ѧϰ��
void NN::save_model(string filename)
{
	FileStorage model(filename, cv::FileStorage::WRITE);
	model << "layer_neuron_num" << Layer_Neurons_nums;
	model << "learning_rate" << rates;
	model << "activation_function" << Activation_Functions;

	for (int i = 0; i < weights.size(); i++)
	{
		string weight_name = "weight_" + to_string(i);
		model << weight_name << weights[i];
	}
	model.release();
}

	//Load model
void NN::load_model(string filename)
{
	FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);
	Mat input_, target_;

	fs["layer_neuron_num"] >> Layer_Neurons_nums;
	BeginNN(Layer_Neurons_nums);

	for (int i = 0; i < weights.size(); i++)
	{
		string weight_name = "weight_" + to_string(i);
		fs[weight_name] >> weights[i];
	}

	fs["learning_rate"] >> rates;
	fs["activation_function"] >> Activation_Functions;

	fs.release();
}

//��xml�ļ��д�ָ�����п�ʼ��ȡһ����Ŀ�������ͱ�ǩ��Ĭ�ϴӵ�0�п�ʼ��ȡ 
void get_file_from_xml(string filename, Mat& input, Mat& label, int sample_num, int start)
{
	FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);
	Mat input_, target_;
	fs["input"] >> input_;
	fs["target"] >> target_;
	fs.release();
	input = input_(Rect(start, 0, sample_num, input_.rows));
	label = target_(Rect(start, 0, sample_num, target_.rows));
}