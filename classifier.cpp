#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Define the logistic function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Define the gradient of the logistic function
double sigmoid_gradient(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// Define the model parameters
double theta0 = 0.0, theta1 = 0.0, theta2 = 0.0;

// Define the learning rate and number of iterations
double alpha = 0.1;
int num_iterations = 10000;

// Generate the training and test datasets
vector<pair<vector<double>, int>> train_dataset, test_dataset;

void generate_datasets() {
    // Generate 500 points in class 0 for training
    for (int i = 0; i < 500; i++) {
        double x1 = (double) rand() / RAND_MAX;
        double x2 = (double) rand() / RAND_MAX;
        train_dataset.push_back(make_pair(vector<double>{x1, x2}, 0));
    }

    // Generate 500 points in class 1 for training
    for (int i = 0; i < 500; i++) {
        double x1 = (double) rand() / RAND_MAX + 1.0;
        double x2 = (double) rand() / RAND_MAX + 1.0;
        train_dataset.push_back(make_pair(vector<double>{x1, x2}, 1));
    }

    // Generate 500 points in class 0 for testing
    for (int i = 0; i < 500; i++) {
        double x1 = (double) rand() / RAND_MAX;
        double x2 = (double) rand() / RAND_MAX;
        test_dataset.push_back(make_pair(vector<double>{x1, x2}, 0));
    }

    // Generate 500 points in class 1 for testing
    for (int i = 0; i < 500; i++) {
        double x1 = (double) rand() / RAND_MAX + 1.0;
        double x2 = (double) rand() / RAND_MAX + 1.0;
        test_dataset.push_back(make_pair(vector<double>{x1, x2}, 1));
    }
}

// Train the logistic regression model
void train_model() {
    // Iterate over the training dataset
    for (int i = 0; i < num_iterations; i++) {
        // Randomly select a point from the training dataset
        int index = rand() % train_dataset.size();
        vector<double> x = train_dataset[index].first;
        int y = train_dataset[index].second;

        // Compute the hypothesis
        double z = theta0 + theta1 * x[0] + theta2 * x[1];
        double h = sigmoid(z);

        // Update the model parameters using gradient descent
        double error = h - y;
        theta0 -= alpha * error;
        theta1 -= alpha * error * x[0];
        theta2 -= alpha * error * x[1];
    }
}

// Predict the class of a new point
int predict(vector<double> x) {
    double z = theta0 + theta1 * x[0] + theta2 * x[1];
    double h = sigmoid(z);
    if (h >= 0.5) {
        return 1;
    } else {
        return 0;
    }
}

// Calculate the accuracy of the model on the test dataset
double calculate_accuracy() {
int num_correct = 0;
for (int i = 0; i < test_dataset.size(); i++) {
vector<double> x = test_dataset[i].first;
int y = test_dataset[i].second;
int y_pred = predict(x);
if (y_pred == y) {
num_correct++;
}
}
return (double) num_correct / test_dataset.size();
}

int main() {
// Generate the training and test datasets
generate_datasets();

// Train the logistic regression model
train_model();

// Calculate the accuracy of the model on the test dataset
double accuracy = calculate_accuracy();

// Print the accuracy of the model
cout << "Accuracy: " << accuracy << endl;

return 0;

}
