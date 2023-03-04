#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include "helpers.cpp"

using std::string, std::vector, std::cout, std::endl, std::to_string;

double calc_age_lh(double v, double mean_v, double var_v) {
    return 1 / sqrt(2 * atan(1)*4 * var_v) * exp(-(pow(v-mean_v, 2)/(2 * var_v)));
}

vector<double> calc_raw_prob(vector<double> apriori, vector<vector<double>> lh_pclass, vector<vector<double>> lh_sex, vector<double> age_mean, vector<double> age_var, double pclass, double sex, double age) {
    double num_s = lh_pclass[1][pclass - 1] * lh_sex[1][sex] * apriori[1] * calc_age_lh(age, age_mean[1], age_var[1]);
    double num_p = lh_pclass[0][pclass - 1] * lh_sex[0][sex] * apriori[0] * calc_age_lh(age, age_mean[0], age_var[0]);
    double denominator = num_s + num_p;
    return {num_s / denominator, num_p / denominator};
}

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
    vector<double> train_pclass, test_pclass, train_sex, test_sex, train_age, test_age, train_survived, test_survived;
    for(int i = 0; i < numObservations; i++) {
        if (i < 800) {
            train_pclass.push_back(pclass[i]);
            train_sex.push_back(sex[i]);
            train_age.push_back(age[i]);
            train_survived.push_back(survived[i]);
        }
        else {
            test_pclass.push_back(pclass[i]);
            test_sex.push_back(sex[i]);
            test_age.push_back(age[i]);
            test_survived.push_back(survived[i]);
        }
    }

    auto start = std::chrono::steady_clock::now();

    // Calculate priors
    vector<double> apriori(2);
    for(int i = 0; i < train_survived.size(); i++)
        apriori[(train_survived[i])]++;
    apriori[0] /= train_survived.size();
    apriori[1] /= train_survived.size();

    // Calculate likelihoods for qualitative data
    vector<double> count_survived(2);
    for(auto sv : train_survived)
        count_survived[(sv)]++;

    vector<vector<double>> lh_pclass(2, vector<double>(3, 0));
    for(int sv = 0; sv < 2; sv++)
        for(int pc = 1; pc < 4; pc++) {
            double cnt = 0;
            for(int i = 0; i < train_survived.size(); i++)
                if(train_pclass[i] == pc && train_survived[i] == sv)
                    cnt++;
            lh_pclass[sv][pc - 1] = cnt / count_survived[sv];
        }

    vector<vector<double>> lh_sex(2, vector<double>(2, 0));
    for(int sv = 0; sv < 2; sv++)
        for(int sx = 0; sx < 2; sx++) {
            double cnt = 0;
            for(int i = 0; i < train_survived.size(); i++)
                if(train_sex[i] == sx && train_survived[i] == sv)
                    cnt++;
            lh_sex[sv][sx] = cnt / count_survived[sv];
        }

    // Calculate likelihoods for quantitative data
    vector<double> age_mean(2), age_var(2);
    for(int sv = 0; sv < 2; sv++) {
        vector<double> selected_ages;
        double sum = 0;
        for(int i = 0; i < train_age.size(); i++)
            if(train_survived[i] == sv) {
                selected_ages.push_back(train_age[i]);
                sum += train_age[i];
            }
        age_mean[sv] = sum / selected_ages.size();
        age_var[sv] = variance(age_mean[sv], selected_ages);
    }

    auto end = std::chrono::steady_clock::now();

    cout << "Prior probability for not surviving and surviving, respectively:\n";
    cout << apriori[0] << "\n" << apriori[1] << "\n\n";

    cout << "Likelihood values for p(pclass|survived):\n";
    print_matrix(lh_pclass);
    cout << endl;

    cout << "Likelihood values for p(sex|survived):\n";
    print_matrix(lh_sex);
    cout << endl;

    // Apply to the first 5 test observations
    cout << "Perished Survived\n";
    for(int i = 0; i < 5; i++) {
        vector<double> raw = calc_raw_prob(apriori, lh_pclass, lh_sex, age_mean, age_var, test_pclass[i], test_sex[i], test_age[i]);
        cout << raw[1] << " " << raw[0] << "\n";
    }
    cout << endl;

    vector<double> predicted;
    for(int i = 0; i < test_survived.size(); i++) {
        vector<double> raw = calc_raw_prob(apriori, lh_pclass, lh_sex, age_mean, age_var, test_pclass[i], test_sex[i], test_age[i]);
        predicted.push_back(raw[0] > 0.5);
    }

    vector<double> results = naive_bayes_metrics(predicted, test_survived);

    cout << "Accuracy = " << results[0] << endl;
    cout << "Sensitivity = " << results[1] << endl;
    cout << "Specificity = " << results[2] << endl;

    cout << "Elapsed training time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "ns";

    return 0;
}
