#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

using std::string, std::ifstream, std::vector, std::cout, std::endl;

double sum(vector<double> v) {
    double sum = 0;
    for (double val : v)
        sum += val;
    return sum;
}

double mean(vector<double> v) {
    return sum(v) / (double)v.size();
}

double median(vector<double> v) {
    // v is assumed to be sorted
    int middle = v.size() / 2;
    if (v.size() % 2 == 0)
        return ((v[middle - 1] + v[middle]) / 2.0);
    return v[middle];
}

void print_stats(vector<double> v) {
    sort(v.begin(), v.end());
    cout << "Sum: " << sum(v) << endl;
    cout << "Mean: " << mean(v) << endl;
    cout << "Median: " << median(v) << endl;
    cout << "Range: " << v[0] << " " << v[v.size() - 1] << endl;
}

double covar(vector<double> x, vector<double> y) {
    int n = x.size();
    double meanX = mean(x), meanY = mean(y), numerator = 0;
    for (int i = 0; i < n; i++)
        numerator += (x[i] - meanX) * (y[i] - meanY);
    return numerator / (double)(n - 1);
}

double cor(vector<double> x, vector<double> y) {
    double covariance = covar(x, y), sigmaX = sqrt(covar(x, x)), sigmaY = sqrt(covar(y, y));
    return covariance / (sigmaX * sigmaY);
}


int main(int argc, char** argv) {
    ifstream inFS;
    string line, rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN), medv(MAX_LEN);

    cout << "Opening file Boston.csv." << endl;

    inFS.open(R"(C:\Users\Sam\CLionProjects\DataExploration\Boston.csv)");
    if(!inFS.is_open()) {
        cout << "Could not open file Boston.csv" << endl;
        return 1;
    }

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    cout << "heading: " << line << endl;

    int numObservations = 0;
    while(inFS.good()) {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;

    cout << "Closing file Boston.csv." << endl;
    inFS.close();

    cout << "Number of records: " << numObservations << endl;

    cout << "\nStats for rm" << endl;
    print_stats(rm);

    cout << "\nStats for medv" << endl;
    print_stats(medv);

    cout << "\n Covariance = " << covar(rm, medv) << endl;

    cout << "\n Correlation = " << cor(rm, medv) << endl;

    cout << "\nProgram terminated.";

    return 0;
}
