#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

using std::string, std::vector, std::cout, std::endl, std::to_string;

void print_matrix(vector<vector<double>> m) {
    for (auto row: m) {
        for (auto val: row)
            cout << val << " ";
        cout << endl;
    }
}

double sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}

vector<vector<double>> vector_op(vector<vector<double>> m1, string op, vector<vector<double>> m2) {
    int r1 = m1.size(), c1 = m1[0].size(), r2 = m2.size(), c2 = m2[0].size();
    if (op == "*") {
        if (c1 != r2)
            throw std::runtime_error(
                    "Error: tried to multiply " + to_string(r1) + "x" + to_string(c1) + " matrix with a " +
                    to_string(r2) + "x" + to_string(c2) + "matrix.");
        vector<vector<double>> res(r1, vector<double>(c2, 0));
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                for (int k = 0; k < c1; k++) {
                    res[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
        return res;
    }
    if(r1 != r2 || c1 != c2)
        throw std::runtime_error("Error: tried to perform a vector operation on a " + to_string(r1) + "x" + to_string(c1) + " matrix and a " + to_string(r2) + "x" + to_string(c2) + "matrix.");
    for(int i = 0; i < r1; i++)
        for(int j = 0; j < c1; j++)
            if(op == "+")
                m1[i][j] += m2[i][j];
            else if(op == "-")
                m1[i][j] -= m2[i][j];
            else if(op == "/")
                m1[i][j] /= m2[i][j];
    return m1;
}

vector<vector<double>> transpose(vector<vector<double>> m) {
    int r = m.size(), c = m[0].size();
    vector<vector<double>> res(c, vector<double>(r, 0));
    for(int i = 0; i < r; i++)
        for(int j = 0; j < c; j++)
            res[j][i] = m[i][j];
    return res;
}

vector<vector<double>> scalar_op(vector<vector<double>> m, string op, double val) {
    int r = m.size(), c = m[0].size();
    if(op == "+") {
        for(int i = 0; i < r; i++)
            for(int j = 0; j < c; j++)
                m[i][j] += val;
    }
    else if(op == "-") {
        for(int i = 0; i < r; i++)
            for(int j = 0; j < c; j++)
                m[i][j] -= val;
    }
    else if(op == "*") {
        for(int i = 0; i < r; i++)
            for(int j = 0; j < c; j++)
                m[i][j] *= val;
    }
    else if(op == "/") {
        for(int i = 0; i < r; i++)
            for(int j = 0; j < c; j++)
                m[i][j] /= val;
    }
    else if(op == "sigmoid") {
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                m[i][j] = sigmoid(m[i][j]);
    }
    else if(op == "exp") {
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                m[i][j] = exp(m[i][j]);
    }
    return m;
}

vector<double> log_reg_metrics(vector<vector<double>> predicted, vector<vector<double>> test_labels) {
    vector<vector<double>> exp_predicted = scalar_op(predicted, "exp", 0);
    vector<vector<double>> probabilities = vector_op(exp_predicted, "/", scalar_op(exp_predicted, "+", 1));
    double correct = 0, sensitivity = 0, specificity = 0, total = (double)probabilities[0].size(), total_survived = 0, total_deceased = 0;
    for(int i = 0; i < test_labels[0].size(); i++) {
        if (test_labels[0][i])
            total_survived++;
        else
            total_deceased++;
    }
    for(int i = 0; i < probabilities[0].size(); i++) {
        if (probabilities[0][i] > 0.5 and test_labels[0][i]) {
            sensitivity++;
            correct++;
        } else if (probabilities[0][i] < 0.5 and !test_labels[0][i]) {
            specificity++;
            correct++;
        }
    }
    vector<double> res = {correct / total, sensitivity / total_survived, specificity / total_deceased};
    return res;
}

vector<double> naive_bayes_metrics(vector<double> predicted, vector<double> test_labels) {
    double correct = 0, sensitivity = 0, specificity = 0, total = (double)predicted.size(), total_survived = 0, total_deceased = 0;
    for(int i = 0; i < test_labels.size(); i++) {
        if (test_labels[i])
            total_survived++;
        else
            total_deceased++;
    }
    for(int i = 0; i < predicted.size(); i++) {
        if (predicted[i] and test_labels[i]) {
            sensitivity++;
            correct++;
        } else if (!predicted[i] and !test_labels[i]) {
            specificity++;
            correct++;
        }
    }

    vector<double> res = {correct / total, sensitivity / total_survived, specificity / total_deceased};
    return res;
}

double variance(double mean, vector<double> v) {
    auto add_square = [mean](double sum, int i) {
        auto d = i - mean;
        return sum + d*d;
    };
    double total = std::accumulate(v.begin(), v.end(), 0.0, add_square);
    return total / (v.size() - 1);
}

