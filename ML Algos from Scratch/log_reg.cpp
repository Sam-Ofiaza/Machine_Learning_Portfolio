#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include "helpers.cpp"

using std::string, std::vector, std::cout, std::endl, std::to_string;

int main(int argc, char** argv) {
    std::ifstream inFS;
    string line, row_in, pclass_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1100;
    vector<double> pclass(MAX_LEN), survived(MAX_LEN), sex(MAX_LEN), age(MAX_LEN);

    inFS.open(R"(C:\Users\Sam\CLionProjects\ML_Algos_from_Scratch\titanic_project.csv)");
    if(!inFS.is_open()) {
        cout << "Could not open file titanic_project.csv" << endl;
        return 1;
    }

    getline(inFS, line);

    int numObservations = 0;
    while(inFS.good()) {
        getline(inFS, row_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');


        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }

    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    inFS.close();

    // Create train and test
    vector<double> train_sex, test_sex, train_survived, test_survived;
    for(int i = 0; i < numObservations; i++) {
        if (i < 800) {
            train_sex.push_back(sex[i]);
            train_survived.push_back(survived[i]);
        }
        else {
            test_sex.push_back(sex[i]);
            test_survived.push_back(survived[i]);
        }
    }

    // Set up weight vector, label vector, and data matrix
    vector<vector<double>> data_matrix(train_sex.size(), vector<double>(2, 1)), weights = {{1}, {1}}, labels = {train_survived};
    for(int i = 0; i < data_matrix.size(); i++)
        data_matrix[i][1] = train_sex[i] + 1;

    // Gradient descent
    double learning_rate = 0.001;
    vector<vector<double>> transposed_data_matrix = transpose(data_matrix);

    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < 50000; i++) {
        vector<vector<double>> prob_vector = scalar_op(vector_op(data_matrix, "*", weights), "sigmoid", 0);
        vector<vector<double>> error = vector_op(labels, "-", transpose(prob_vector));
        weights = vector_op(weights, "+", scalar_op(vector_op(transposed_data_matrix, "*", transpose(error)), "*", learning_rate));
    }
    auto end = std::chrono::steady_clock::now();

    cout << "Weights:\n";
    print_matrix(weights);

    // Predict with the generated weights
    vector<vector<double>> test_matrix(test_sex.size(), vector<double>(2, 1)), test_labels = {test_survived};
    for(int i = 0; i < test_matrix.size(); i++)
        test_matrix[i][1] = test_sex[i] + 1;

    vector<vector<double>> predicted = transpose(vector_op(test_matrix, "*", weights));

    vector<double> results = log_reg_metrics(predicted, test_labels);

    cout << "Accuracy = " << results[0] << endl;
    cout << "Sensitivity = " << results[1] << endl;
    cout << "Specificity = " << results[2] << endl;

    cout << "Elapsed training time in seconds: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s";

    return 0;
}
