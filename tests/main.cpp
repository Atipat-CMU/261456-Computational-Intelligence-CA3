#include <iostream>
#include "../include/dotlis.h"
#include "../include/mlp.h"
#include "../include/genea.h"

using namespace std;

using namespace genea;

int main(){

    Dataframe df = read_csv("wdbc.data");
    df = df.get_column_without({0});
    
    df.to_csv("dataset2.csv");
    df.get_onehot(0).to_csv("onehot.csv");

    vector<layer_info> layers = {
        {INPUT, nullptr, 30},
        {HIDDEN, sigmoid, 4},
        {OUTPUT, sigmoid, 2},
    };

    Dataframe X_train = df.get_column_without({0});
    Dataframe y_train = df.get_onehot(0);

    Population networks(layers, 100);

    networks.setData(X_train, y_train);

    cout << networks.getError() << endl;
    for(int generation = 0; generation < 1000; generation++){
        networks.next();
        cout << "generation: " << generation << " " << networks.getError() << endl;
    }
    
    return 0;
}
