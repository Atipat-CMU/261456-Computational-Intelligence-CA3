#include <iostream>
#include "../include/dotlis.h"
#include "../include/mlp.h"
#include "../include/genea.h"

using namespace std;

using namespace genea;

int main(){
    srand(time(0));

    Dataframe df = read_csv("wdbc.data");
    df = df.get_column_without({0});

    df = df.split_train_test({0.9}).first;

    vector<layer_info> layers = {
        {INPUT, nullptr, 30},
        {HIDDEN, sigmoid, 10},
        {HIDDEN, sigmoid, 10},
        {OUTPUT, sigmoid, 2},
    };

    Dataframe X_train = df.get_column_without({0});
    Dataframe y_train = df.get_onehot(0);

    Population networks(layers, 200);

    networks.setData(X_train, y_train);

    double max = 0;
    double current = 0;

    for(int generation = 0; generation < 1000; generation++){
        current = networks.getFitness();
        if(current > max){
            max = current;
            networks.getBestParam().to_file("10-10.param");
        }
        cout << "generation " << generation << ": " << current << endl;
        networks.next();
    }
    
    return 0;
}
