#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "../../mlp/object/Network.h"

namespace genea {
    enum indiv_status {
        P1,
        P2,
        P3,
        NP,
        DIED,
        NONE
    };

    class Individual
    {
    private:
        Network *neuralN;
        double fitness;
        indiv_status status = NONE;

    public:
        Individual(vector<layer_info> layers);
        ~Individual();

        void setStatus(indiv_status status);
        void clearStatus();
        bool isStatus(indiv_status status);

        void fit(Dataframe X, Dataframe y);
        double getFitness();
        double getError(Dataframe X, Dataframe y);
        Parameter getParameter();
        void setParameter(Parameter param);
        bool operator<(const Individual& obj) const;
    };
    
    Individual::Individual(vector<layer_info> layers)
    {
        this->neuralN = new Network(layers);
    }
    
    Individual::~Individual()
    {
        delete this->neuralN;
    }

    void Individual::setStatus(indiv_status status){
        this->status = status;
    }

    void Individual::clearStatus(){
        this->status = NONE;
    }

    bool Individual::isStatus(indiv_status status){
        return this->status == status;
    }

    void Individual::fit(Dataframe X, Dataframe y){
        double error = this->neuralN->getError(X, y);
        this->fitness = 1/(error+1);
    }

    double Individual::getError(Dataframe X, Dataframe y){
        return this->neuralN->getError(X, y);
    }

    double Individual::getFitness(){
        return this->fitness;
    }

    Parameter Individual::getParameter(){
        return this->neuralN->getParam();
    }

    void Individual::setParameter(Parameter param){
        this->neuralN->setParam(param);
    }

    bool Individual::operator<(const Individual& obj) const
    {
        return fitness < obj.fitness;
    }
    
}

#endif
