// Thomas Klimek
// Neural Net Assignment 4
// neuralnet.cpp
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>     // std::string, std::stod
#include <stdlib.h>
#include <cmath>

using namespace std;

// CHANGE ETA AND ALPHA HERE
double eta = 0.15;
double alpha = 0.5;

/*************************** Explanation and sources ********************************/
/*           
	To run this code: g++ neuralnet.cpp
	./a.out

	(make sure you are in the same folder as testing.txt and training.txt)
	(program will prompt user how many training runs and how many testing runs to preform)


				This Neural Net is built using the following architectire

	1. Class NeuralNet holds the neural network represented as a vector of Layers
		There is 1 input layer, 1 hidden layer, and 1 output layer
		NeuralNet is also responsible for the inputting of values, forward propigation
		and backward propigation.
	2. Class Layer is a typedef of a vector of Neurons. The Neural net is arranged with
		Layer 1 - input layer containing 4 input neurons
		Layer 2 - hidden layer containing 2 hidden neurons
		Layer 3 - output layer containing 1 output neuron
	3. Class Neuron represents a single neuron in the net. A neuron holds an output value,
	a index (relative to the network) a gradient, and a vector of Connections. The neuron
	is used to calculate outputs, change weights, apply the transfer function, and 
	calculate gradients. 
	4. Struct connection is used to represent a single edge in a graph of neurons. The
	struct holds a weight and a delta weight for an edge. Neuron holds a vector of these
	structs to represent all the edges between it and any other neuron it touches.
	5. Class TrainingData is used to read in data from a file name line by line to
	feed it to the NeuralNet class. It is designed to interface with the neural net 
	class and fill in the containers needed by this class. It works specifically with
	the data files provided for this assignment.

	Data Representation:
	In Our Neural net there is a good amount of data transformation for each input and
	output. I will detail the transformations here...
	1. Inputs: inputs are normalized to a number between -1 and 1, fitted for our
	activation function of tanh
	2. Targets: the flower target types are transformed using a step function into
	the range of -1 and 1. An "Iris-setosa" has the value -1.0, an "Iris-versicolor" has
	a value of 0.0, and an "Iris-virginica" has a value of 1.0.
	3. Outputs: Outputs are normalized over the range of -1 to 1 divided into 3 parts.
	An output between -1 and -0.66 corresponds to an "Iris-setosa", an output in the
	range of -0.66 and 0.33 corresponds to an "Iris-versicolor", and an output in the
	range of 0.33 to 1 corresponds to an "Iris-virginica".

	Sources: Neural Network class implementation is based on David Miller's found at
	http://www.millermattson.com/dave/?p=54

	Normalization:
	https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

	More Neural Network Sources:
	https://en.wikipedia.org/wiki/Artificial_neural_network
	Slides
	Piazza

/************************************************************************************/

// Used to convert string to double when reading input
double string_to_double( const std::string& s )
 {
   std::istringstream i(s);
   double x;
   if (!(i >> x))
     return 0;
   return x;
 } 

// Class to read input from file
class TrainingData
{
public: 
	TrainingData(string filename);
	void getNextInput(vector<double> &inputVals);
	double getNextOutput();
	bool isEOF() { return datafile.eof(); }
private:
	ifstream datafile;

};
// Opens file and sets ifstream
TrainingData::TrainingData(string filename){
	datafile.open(filename.c_str());
}
// Reads in 4 doubles of inout into the vector inputVals
void TrainingData::getNextInput(vector<double> &inputVals){
	inputVals.clear();
	string input;
	double input_double;
	for (int i = 0; i < 4; i++){
		getline(datafile, input, ',');
		input_double = string_to_double(input);
		inputVals.push_back(input_double);
	}
}
// Reads in the target output flower, and transforms the data to our range
double TrainingData::getNextOutput(){
	string input;
	double input_double;
	getline(datafile, input);
	if (input.compare("Iris-setosa") == 0)
		input_double = -1.1;
	else if (input.compare("Iris-versicolor") == 0)
		input_double = 0.0;
	else if (input.compare("Iris-virginica") == 0)
		input_double = 1.0;
	return input_double;
}
// represents an edge in the neural network graph
struct Connection
{
	double weight;
	double delta_weight;
};
// forward declaration of neuron so we can typedef layers
class Neuron;
// represents a layer in the neural network by a vector of neurons
typedef vector<Neuron> Layer;
// our class for representing a neuron
class Neuron {
public: 
	Neuron(int connections, int index);
	void setValue(double val) { n_value = val; }
	double getValue() { return n_value; } 
	void feedForward(Layer &prevLayer);
	double transferFunction(double x);
	double transferFunctionDerivative(double x);
	void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
	double n_value;
	double sumError(const Layer &nextLayer) const;
	int n_index;
	vector<Connection> edges;
	double n_gradient;
};
// Class constructor that initializes all the edges, and
// the index of the neuron relative to the web
Neuron::Neuron(int connections, int index){
	for (int i = 0; i < connections; i++){
		Connection edge;
		edge.weight =  rand() / double(RAND_MAX);
		//edge.delta_weight = 0;
		edges.push_back(edge);
	}
	n_index = index;
}
// Forward propigation on neuron level, updates weights and values
// of other neurons connected to this one
void Neuron::feedForward(Layer &prevLayer){
	double sum = 0.0;

	for (int i = 0; i <prevLayer.size(); i++){
		sum += prevLayer[i].getValue() * prevLayer[i].edges[n_index].weight;
	}
	n_value = Neuron::transferFunction(sum);
}
// our transfer function of tanh
double Neuron::transferFunction(double x){
	return tanh(x);
}
// approximate the tanh derrivative using 1 - x^2
double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}
// Sums error from neurons in current layer
double Neuron::sumError(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (int n = 0; n < nextLayer.size() - 1; ++n) {
        sum += edges[n].weight * nextLayer[n].n_gradient;
    }

    return sum;
}
// calculates the gradients for the hidden layer
void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumError(nextLayer);
    n_gradient = dow * Neuron::transferFunctionDerivative(n_value);
}

//calculates the gradients for the output layer
void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - n_value;
    n_gradient = delta * Neuron::transferFunctionDerivative(n_value);
}

// used to update input weights during backwards propigation
void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (int n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.edges[n_index].delta_weight;
        double newDeltaWeight =
                eta
                * neuron.getValue()
                * n_gradient
                + alpha
                * oldDeltaWeight;

        neuron.edges[n_index].delta_weight = newDeltaWeight;
        neuron.edges[n_index].weight += newDeltaWeight;
    }
}


// our neuralnet class that represents the entire ANN
class NeuralNet
{
public:
	NeuralNet(vector<int> layers);
	void input(vector<double> &inputVals);
	void backProp(vector<double> &targetVals);
	void getResults(vector<double> &resultVals);
	void forwardProp();
	void print();
private:
	vector<Layer> net_layers;
	double n_error;
	double n_recentAverageError;
    static double n_recentAverageSmoothingFactor;
};
double NeuralNet::n_recentAverageSmoothingFactor = 120.0; // Number of training samples to average over

// contructor that takes a initialization vector as input. Each value inside this vector corresponds
// with how many neurons will be in the layer of the corresponding index i.e. 
// [3, 2, 1] - initializes a neural net with 3 input nodes, 2 hidden nodes, and 1 output node
NeuralNet::NeuralNet(vector<int> layerinit){
	int numLayers = layerinit.size();
	int connections;
	for (int i = 0; i < numLayers; i++){
		net_layers.push_back(Layer());
		if (i == numLayers - 1)
			connections = 0;
		else
			connections = layerinit[i + 1];
		for (int j = 0; j <= layerinit[i]; j++){
			net_layers.back().push_back(Neuron(connections, j));
		}
		net_layers.back().back().setValue(1.0);
	}
}
// normalization function for inputs to the nueral net to meet the tanh range
double normalize(double input, int index){
	if (index == 0){
		return -1 + 2 * ((input - 4.3) / (7.9 - 4.3));
	} else if (index == 1){
		return -1 + 2 * ((input - 2.0) / (4.4 - 2.0));
	} else if (index == 2){
		return -1 + 2 * ((input - 1.0) / (6.9 - 1.0));
	} else if (index == 3){
		return -1 + 2 * ((input - 0.1) / (2.5 - 0.1));
	} else { return 0; }
}
// input function for the neural net, feeds inputVals into the input layer
void NeuralNet::input(vector<double> &inputVals)
{
	double norminput;
	//cout << "input: ";
	for (int i = 0; i< inputVals.size(); i++){
		norminput = inputVals[i];
		norminput = normalize(norminput, i);
		net_layers[0][i].setValue(norminput);
		//cout << norminput << " ";
	}
	//cout << endl;
}
// forward propigation function, relies on the Neuron feedForward function
void NeuralNet::forwardProp(){
	for (int i = 1; i < net_layers.size(); i++){
		Layer &prevLayer = net_layers[i - 1];
		for (int j = 0; j < net_layers[i].size() - 1; j++){
			net_layers[i][j].feedForward(prevLayer);
		}
	}
}
// result conversion for the reverse transform from numbers to flower types
string convertResult(double num){
	if (num < -0.66){
		return "Iris-setosa";
	} else if (num < 0.33) {
		return "Iris-versicolor";
	} else if (num <= 1.0 ) {
		return "Iris-virginica";
	} else { return "Unexecpected result" ;}
}
// back propigation of the neural net
void NeuralNet::backProp(vector<double> &targetVals){
	//cout << "expected result: " << targetVals[0] << 
	//	"(" << convertResult(targetVals[0]) << ")" << endl;
	// calculate net error
	Layer &outputLayer = net_layers.back();
    n_error = 0.0;

    for (int n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getValue();
        n_error += delta * delta;
    }
    n_error /= outputLayer.size() - 1; // get average error squared
    n_error = sqrt(n_error); // RMS
	
	n_recentAverageError =
            (n_recentAverageError * n_recentAverageSmoothingFactor + n_error)
            / (n_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients

    for (int n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // calculate hidden gradients
   for (int i = net_layers.size() - 2; i > 0; --i) {
        Layer &hiddenLayer = net_layers[i];
        Layer &nextLayer = net_layers[i + 1];

        for (int n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

	// For all layers from outputs to first hidden layer,
    // update connection weights

    for (int i = net_layers.size() - 1; i > 0; --i) {
        Layer &layer = net_layers[i];
        Layer &prevLayer = net_layers[i - 1];

        for (int n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
	
}
// print function used for testing
void NeuralNet::print(){
	for (int i = 0; i < net_layers.size(); i++){
		for (int j = 0; j < net_layers[i].size(); j++){
			cout << net_layers[i][j].getValue() << " ";
		}
		cout << endl;
	}
}
// returns value of the output neuron
void NeuralNet::getResults(vector<double> &resultVals)
{
	double result;
    resultVals.clear();

    for (int n = 0; n < net_layers.back().size() - 1; ++n) {
    	result = net_layers.back()[n].getValue();
        resultVals.push_back(result);
        //cout << "Result val: " << result << 
		//" (" << convertResult(result) << ")"  << endl;
    }
}
// print function used for console input
void printInput(vector<double> &inputVals){
	cout << "gardener inputs: ";
	for (int i = 0; i < 4; i++){
		cout << inputVals[i] << " ";
	}
	cout << endl;
}
// print function used for console output
void printResults(vector<double> &resultVals){
	cout << "Neural Net predicts: " << resultVals[0]
	<< " (" << convertResult(resultVals[0]) << ")"  << endl;
}
// print function used for console output
void printTarget(double target){
	cout << "The actual plant type is: " << convertResult(target) << endl;
}
int main()
{
	int training_runs;
	int testing_runs;
	//cout << "Welcome to the Neural Net, Please input eta double: (x.x) [Range from 0.010 – 0.999]" << endl;
	//cin >> eta;
	//cout << "Please input alpha double: (x.x) [Range from  0.100 – 0.900] " << endl;
	//cin >> alpha;

	cout << "Training data set contains 120 data points, how many training runs would you like: " << endl;
	cin >> training_runs;
	// initialize neural net
	vector<int> layerinit;
	layerinit.push_back(4);
	layerinit.push_back(2);
	layerinit.push_back(1);

	NeuralNet network(layerinit);

	vector<double> inputVals;
	vector<double> targetVals;
	vector<double> resultVals;

	// TRAINING
	cout << "TRAINING DATA AND VALIDATING FOR " << training_runs << " RUNS..." << endl;
	for (int i = 0; i < training_runs; i++){
		TrainingData data("training.txt");
		while (!data.isEOF()) {

			data.getNextInput(inputVals);
			network.input(inputVals);
			network.forwardProp();

			network.getResults(resultVals);

			targetVals.clear();
			double targetVal = data.getNextOutput();
			targetVals.push_back(targetVal);
			network.backProp(targetVals);
		}
		//network.print();
	}
		
	cout << "Testing data set contains 30 data points, how many testing runs would you like: " << endl;
	cin >> testing_runs;
	cout << "------TESTING DATA ---------" << endl;
	double correct = 0.0;
	for (int i = 0; i < testing_runs; i++){
		TrainingData data("testing.txt");
			while (!data.isEOF()) {

				data.getNextInput(inputVals);
				printInput(inputVals);
				network.input(inputVals);
				network.forwardProp();

				network.getResults(resultVals);
				printResults(resultVals);

				targetVals.clear();
				double targetVal = data.getNextOutput();
				targetVals.push_back(targetVal);
				printTarget(targetVal);
			
				if (convertResult(targetVals[0]).compare(convertResult(resultVals[0])) == 0){
					correct++;
				}


				network.backProp(targetVals);
				cout << endl;
			}
	}
	cout << "Overall Accuracy: " << correct/(30*testing_runs) << " percent" << endl;
	cout << "Missed: " << (30*testing_runs) - correct 
		<< " out of : " << 30 * testing_runs << " total." << endl;

}