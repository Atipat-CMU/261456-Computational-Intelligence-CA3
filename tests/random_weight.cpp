#include <iostream>
#include "../include/dotlis.h"
#include "../include/mlp.h"

using namespace std;

using namespace dotlis;

int main(){

    vector<layer_info> layers = {
        {INPUT, nullptr, 8},
        {HIDDEN, linear, 8},
        {OUTPUT, linear, 1},
    };

    Parameter init_parameter(layers, -1, 1);
    init_parameter.to_file("default.param");

    return 0;
}
