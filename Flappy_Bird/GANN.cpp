/*
#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
*/

float sigmoid(float x){
	return 1.0 / (1.0 + exp(-x));
}

float derivative_s(float x){
	return x * (1 - x);
}

struct NN
{
	deque<float>layers;
	deque<MatrixXf>weights;
	float lr;
	NN(){
		lr = 0.2;
		for(int i = 0; i < 3; i++){
			layers.push_back(Layers[i]);
		}
		for(int i = 0; i < layers.size() - 1; i++){
			MatrixXf w = (MatrixXf::Random(layers[i + 1], layers[i])) * (1 / sqrt(layers[i + 1]));
			
			weights.push_back(w);
		}
	}

	void train(ArrayXXf input_list, ArrayXXf target_list){
		MatrixXf inputs = input_list.matrix();
		MatrixXf targets = target_list.matrix();

		deque<MatrixXf>inp, out, error;
		inp.push_back(inputs);
		out.push_back(inputs);
		error.push_back(MatrixXf::Zero(1, 1));


		for(int i = 1; i < layers.size(); i++){
			//cout << weights[i - 1].cols() << " " << out[i - 1].rows() << "\n";
			inp.push_back(weights[i - 1] * out[i - 1]);
			out.push_back(inp[i].unaryExpr(&sigmoid));
			error.push_back(MatrixXf::Zero(1, 1));
		}

		for(int i = error.size() - 1; i >= 0; i--){
			if(i == error.size() - 1){
				error[i] = (targets - out[i]).cwiseProduct(out[i].unaryExpr(&derivative_s));
			}
			else{
				error[i] = (weights[i].transpose() * error[i + 1]).cwiseProduct(out[i].unaryExpr(&derivative_s));
			}
		}

		for(int i = weights.size() - 1; i >= 0; i--){
			weights[i] += lr * (error[i + 1] * out[i].transpose());
		}
		
	}

	ArrayXXf query(ArrayXXf input_list){
		MatrixXf inputs = input_list.matrix();

		deque<MatrixXf>inp, out;
		inp.push_back(inputs);
		out.push_back(inputs);

		for(int i = 1; i < layers.size(); i++){
			//cout << weights[i - 1].cols() << " " << out[i - 1].rows() << "\n";
			inp.push_back(weights[i - 1] * out[i - 1]);
			out.push_back(inp[i].unaryExpr(&sigmoid));
		}

		return out[layers.size() - 1].array();

	}

};

/*

int main(int argc, char const *argv[])
{	
	int layers[] = {784, 500, 10};
	NN n(layers, 0.2);

	string foo;

	int epoch = 2;

	for(int k = 0; k < epoch; k++){
		freopen("mnist_train.csv", "r", stdin);
		int c = 0;
		while(cin >> foo){
			ArrayXXf input_list(784, 1), target_list(10, 1);

			for(int i = 0; i < 10; i++)
				target_list(i, 0) = 0.01;

			replace(foo.begin(), foo.end(), ',', ' ');
			stringstream ss(foo);

			float temp;

			for(int i = 0; i < 785; i++){
				ss >> temp;
				if(i == 0)
					target_list(temp, 0) = 0.99;
				else{
					input_list(i - 1, 0) = (temp / 255.0 * 0.99) + 0.01;
					//cout << input_list(i - 1, 0) << " " << temp << "\n";
				}
			}
			n.train(input_list, target_list);

			c++;
			cout << c << "\n";
		}
	
		cin.clear();

	}

	cin.clear();

	freopen("mnist_test.csv", "r", stdin);

	float counter = 0.0, sum = 0.0;

	while(cin >> foo){
		ArrayXXf input_list(784, 1);
		ArrayXXf output(10, 1);

		replace(foo.begin(), foo.end(), ',', ' ');
		stringstream ss(foo);

		float temp, target;

		for(int i = 0; i < 785; i++){
			ss >> temp;
			if(i == 0)
				target = temp;
			else{
				input_list(i - 1, 0) = (temp / 255.0 * 0.99) + 0.01;
				//cout << input_list(i - 1, 0) << " " << temp << "\n";
			}
		}
		output = n.query(input_list);
		float biggest, biggestIndex;
		for(int i = 0; i < 10; i++){
			if(i == 0 or output(i, 0) > biggest){
				biggest = output(i, 0);
				biggestIndex = i;
			}
		}
		if(biggestIndex == target)sum++;
		counter++;
		cout << counter << "\n";
		//cout << biggestIndex << " " << target << "\n";
	}

	cout << sum / counter;

	return 0;
}
*/