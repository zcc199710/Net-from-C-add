#include "NN.h"

//初始化神经网络  生成每一层的矩阵、每一个权值矩阵和每一个偏置矩阵
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

//权值初始化函数initWeights()调用initWeight()函数，其实就是初始化一个和多个的区别
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
//偏置初始化是给所有的偏置赋相同的值。这里用Scalar对象来给矩阵赋值。
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

//微分 函数 （Derivative） 简称求导   
Mat NN::derivativeFunction(Mat& fx, string func_type)
{
	Mat dx;
	if (func_type == "sigmoid")
	{
		dx = sigmoid(fx).mul((1 - sigmoid(fx)));
	}
	return dx;
}

//计算输出误差和目标函数，所有输出误差平方和的均值作为需要最小化的目标函数（在反向传播中运用）
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
//前向传播
void NN::Forward_Trans()
{
	for (int i = 0; i < Layer_Neurons_nums.size() - 1; ++i)
	{
		Mat product = weights[i] * layers[i] + dross[i];   //矩阵相乘的结果 
		// 调用激活函数 激活函数一般都是非线性函数。它存在的价值就是给神经网络提供非线性建模能力
		layers[i + 1] = activationFunction(product, Activation_Functions);  // 调用激活函数 激活函数一般都是非线性函数。它存在的价值就是给神经网络提供非线性建模能力
	}
}

//激活函数 func_type的值为你所要选择的激活函数
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

//反向传播(求误差 求导 寻找最优解 更新权重 )
void NN::Backward_Trans()
{
	calcLoss(layers[layers.size() - 1], target, output_error, loss);
	deltaError();  
	update_Weights();
}

//计算delta误差，    需要注意的就是计算的时候输出层和隐藏层的计算公式是不一样的。
void NN::deltaError()
{
	delta_err.resize(layers.size() - 1);
	for (int i = delta_err.size() - 1; i >= 0; i--)
	{
		delta_err[i].create(layers[i + 1].size(), layers[i + 1].type());
		Mat dx = derivativeFunction(layers[i + 1], Activation_Functions);   //求导 （ 微分公式 ）
		//输出层（Output layer） delta error
		if (i == delta_err.size() - 1)
		{
			delta_err[i] = dx.mul(output_error);
		}
		else  //隐藏层（Hidden layer） delta error
		{
			Mat weight = weights[i];
			Mat weight_t = weights[i].t();
			Mat delta_err_1 = delta_err[i];
			delta_err[i] = dx.mul((weights[i + 1]).t() * delta_err[i + 1]);
		}
	}
}

//更新权重
void NN::update_Weights()
{
	for (int i = 0; i < weights.size(); ++i)
	{
		Mat delta_weights = rates * (delta_err[i] * layers[i].t());
		weights[i] = weights[i] + delta_weights;
	}
}

//Train  跟下面的输入是一个原理 接受一个样本（即一个单列矩阵）作为输入
//Train  利用loos阙值作为终止条件
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

//Test   测试
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
		//前向传播得到最后一层输出层layerout，然后从layerout中提取最大值的位置，最后输出位置的y坐标。
		minMaxLoc(layer_out, NULL, NULL, NULL, &predict_maxLoc, cv::noArray());//opencv里的函数
		return predict_maxLoc.y;
	}
	else
	{
		cout << "Please give one sample alone and ensure input.rows = layer[0].rows" << endl;
		return -1;
	}
}

//Predict  循环的调用上一个函数
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

// 模型包括  各层神经元数目 超级权重 激活函数 学习率
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

//从xml文件中从指定的列开始提取一定数目的样本和标签。默认从第0列开始读取 
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