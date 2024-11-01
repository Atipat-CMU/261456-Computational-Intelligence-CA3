#ifndef POPULATION_H
#define POPULATION_H

#include <vector>

using namespace std;

#include "../../mlp/object/Network.h"
#include "../../genea/object/Individual.h"

namespace genea {
    class Population
    {
    private:
        int generation_count = 0;
        vector<Individual*> p;
        int number;
        Dataframe X, y;
        vector<layer_info> layers;

        double random(double min, double max);

        vector<Individual*> getGroupP(indiv_status st);
        vector<Individual*> getTop(vector<Individual*> p, int top);
        void setGroupP(indiv_status st, vector<Individual*> target);
        void setGroupP(indiv_status st, int start, int end);
        Parameter* mating(Parameter in1, Parameter in2);
        Parameter* mutate(Parameter in);
        void selectToP1();
        void createP2();
        void createP3();
        void createNP();
        void replaceP();
        
    public:
        Population(vector<layer_info> layers, int g);
        ~Population();

        void setData(Dataframe X, Dataframe y);
        double getFitness();
        Parameter getBestParam();
        void next();
    };
    
    Population::Population(vector<layer_info> layers, int n)
    {
        this->number = n;
        for(int i = 0; i < n; i++){
            Individual* indiv = new Individual(layers);
            p.push_back(indiv);
        }
        this->layers = layers;
    }
    
    Population::~Population()
    {
        for (Individual* indiv : p) {
            if (indiv != nullptr) {
                delete indiv;
                indiv = nullptr;
            }
        }
        p.clear();
    }

    double Population::random(double min, double max){
        float r1 = (float)rand() / (float)RAND_MAX;
        return min + r1 * (max - min);
    }

    vector<Individual*> Population::getGroupP(indiv_status st){
        vector<Individual*> id_indiv;

        for (Individual* indiv : p) {
            if(indiv->isStatus(st)){
                id_indiv.push_back(indiv);
            }
        }
        return id_indiv;
    }

    vector<Individual*> Population::getTop(vector<Individual*> p, int top){
        for (Individual* indiv : p) {
            indiv->fit(this->X, this->y);
        }
        sort(p.begin(), p.end(), [](Individual* a, Individual* b) {
            return a->getFitness() < b->getFitness(); // Sorting in descending order of fitness
        });

        vector<Individual*> target_indiv;

        for(int i = p.size()-top; i < p.size(); i++){
            target_indiv.push_back(p[i]);
        }
        return target_indiv;
    }

    void Population::setGroupP(indiv_status st, int start, int range){
        int end = min(start+range-1, int(p.size()));
        for(int i = start; i <= end; i++){
            p[i]->setStatus(st);
        }
    }

    void Population::setGroupP(indiv_status st, vector<Individual*> target){
        for (Individual* indiv : target) {
            indiv->setStatus(st);
        }
    }

    Parameter* Population::mating(Parameter in1, Parameter in2){
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> distrib(1, 2);

        vector<vector<int>> unit_all;
        for(int i = 0; i < layers.size(); i++){
            vector<int> unit_ly;
            for(int j = 0; j < layers[i].N_node; j++){
                unit_ly.push_back(distrib(gen));
            }
            unit_all.push_back(unit_ly);
        }

        vector<vector<double>> weight_lys;
        vector<vector<double>> bias_lys;
        vector<double> weight_ly;
        vector<double> bias_ly;

        weight_lys.push_back(weight_ly);
        bias_lys.push_back(bias_ly);
        for(int i = 1; i < layers.size(); i++){
            for(int j = 0; j < layers[i].N_node; j++){
                if(unit_all[i][j] == 1){
                    vector<double> weight_ls = in1.get_weight_unit(layers, i, j);
                    for(double weight : weight_ls){
                        weight_ly.push_back(weight);
                    }
                    bias_ly.push_back(in1.get_bias_unit(layers, i, j));
                }else{
                    vector<double> weight_ls = in2.get_weight_unit(layers, i, j);
                    for(double weight : weight_ls){
                        weight_ly.push_back(weight);
                    }
                    bias_ly.push_back(in2.get_bias_unit(layers, i, j));
                }
            }
            weight_lys.push_back(weight_ly);
            bias_lys.push_back(bias_ly);
            weight_ly.clear();
            bias_ly.clear();
        }

        return new Parameter(weight_lys, bias_lys);
    }

    Parameter* Population::mutate(Parameter in){
        double pm = 0.5;

        vector<vector<double>> weight_lys = in.get_weight_lys();
        vector<vector<double>> bias_lys = in.get_bias_lys();

        vector<vector<double>> weight_lys_new;
        vector<vector<double>> bias_lys_new;

        for(vector<double> weight_ls : weight_lys){
            vector<double> weight_ly_new;
            for(double weight : weight_ls){
                if(random(0,1) < pm){
                    weight_ly_new.push_back(weight + random(-1,1));
                }else{
                    weight_ly_new.push_back(weight);
                }
            }
            weight_lys_new.push_back(weight_ly_new);
        }
        for(vector<double> bias_ls : bias_lys){
            vector<double> bias_ly_new;
            for(double bias : bias_ls){
                if(random(0,1) < pm){
                    bias_ly_new.push_back(bias + random(-1,1));
                }else{
                    bias_ly_new.push_back(bias);
                }
            }
            bias_lys_new.push_back(bias_ly_new);
        }

        return new Parameter(weight_lys_new, bias_lys_new);
    }

    void Population::selectToP1(){
        vector<Individual*> p1 = this->getTop(p, int(this->number*0.8));
        this->setGroupP(P1, p1);
    }

    void Population::createP2(){
        vector<Individual*> p1 = this->getGroupP(P1);

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> distrib(0, p1.size() - 1);

        for(int i = 0; i < int(this->number*0.8); i++){
            int id1 = distrib(gen);
            int id2 = distrib(gen);
            while (id1 == id2) id2 = distrib(gen);

            Parameter in1 = p1[id1]->getParameter();
            Parameter in2 = p1[id2]->getParameter();
            Parameter* ofs = this->mating(in1, in2);

            Individual* indiv = new Individual(layers);
            indiv->setParameter(ofs);
            indiv->setStatus(P2);
            p.push_back(indiv);
        }
    }

    void Population::createP3(){
        vector<Individual*> p2 = this->getGroupP(P2);
        for (Individual* indiv : p2) {
            indiv->setParameter(this->mutate(indiv->getParameter()));
        }
        this->setGroupP(P3, p2);
    }

    void Population::createNP(){
        vector<Individual*> p3 = this->getGroupP(P3);
        this->setGroupP(NP, p3);

        vector<Individual*> p1 = this->getGroupP(P1);
        this->setGroupP(NP, this->getTop(p1, int(this->number*0.2)));

        vector<Individual*> p0 = this->getGroupP(NONE);

        random_device rd;
        mt19937 gen(rd());
        shuffle(p0.begin(), p0.end(), gen);
        
        vector<Individual*> np = this->getGroupP(NP);
        int need = this->number - np.size();
        for(int i = 0; i < need; i++){
            Individual* target_in = p0[i];
            target_in->setStatus(NP);
            np.push_back(target_in);
        }
    }

    void Population::replaceP(){
        vector<Individual*> np = this->getGroupP(NP);
        for (Individual* indiv : p) {
            if (indiv != nullptr) {
                if(!indiv->isStatus(NP)){
                    delete indiv;
                    indiv = nullptr;
                }
            }
        }
        p.clear();
        p = np;
        this->setGroupP(NONE, p);
    }

    void Population::next(){
        selectToP1();
        createP2();
        createP3();
        createNP();
        replaceP();
    }

    double Population::getFitness(){
        for (Individual* indiv : p) {
            indiv->fit(this->X, this->y);
        }
        sort(p.begin(), p.end(), [](Individual* a, Individual* b) {
            return a->getFitness() < b->getFitness();
        });
        return p[p.size()-1]->getFitness();
    }

    Parameter Population::getBestParam(){
        for (Individual* indiv : p) {
            indiv->fit(this->X, this->y);
        }
        sort(p.begin(), p.end(), [](Individual* a, Individual* b) {
            return a->getFitness() < b->getFitness();
        });
        return p[p.size()-1]->getParameter();
    }

    void Population::setData(Dataframe X, Dataframe y){
        this->X = X;
        this->y = y;
    }
}

#endif
