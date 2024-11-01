#include <iostream>
#include "../include/dotlis.h"
#include "../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df = read_csv("wdbc.data");
    df = df.get_column_without({0});

    df = df.split_train_test({0.9}).second;

    vector<layer_info> layers = {
        {INPUT, nullptr, 30},
        {HIDDEN, sigmoid, 10},
        {HIDDEN, sigmoid, 10},
        {HIDDEN, sigmoid, 10},
        {OUTPUT, sigmoid, 2},
    };

    Dataframe X_test = df.get_column_without({0});
    Dataframe y_test = df.get_onehot(0);

    Parameter loaded_param = param_read("10-10-10.param");

    Network network(layers);
    network.setParam(new Parameter(loaded_param.get_weight_lys(), loaded_param.get_bias_lys()));

    Dataframe y_pred = network.predict(X_test);
    y_pred = markMax(y_pred);

    double sum_acc = calConfusionM(y_pred.get_column({0}), y_test.get_column({0}));

    cout << "---------------------------------------------------------" << endl;
    cout << "RMSE: " << sum_acc << endl;

    return 0;
}
