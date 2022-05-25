#include<vector>
#include<Eigen/Dense>
#include<sstream>
#include<fstream>
#include <stdlib.h>
#include<iostream>
using Eigen::RowVector3d;
using Eigen::RowVector4i;

RowVector3d getVector(int N,double t){
    std::string line, temp;
    std::ifstream file;
    RowVector3d a;
    RowVector3d b;
    file.open("points.txt");
    int counter=0;
    while (getline(file, line)){
        if(counter==N){
            std::stringstream ss1(line);
            ss1>>a[0];
            ss1>>a[1];
            ss1>>a[2];
            ss1>>b[0];
            ss1>>b[1];
            ss1>>b[2];
            //std::cout<<"ciao"<<std::endl;
            file.close();
            return (1-t)*a+t*b;
        }
        ++counter;
    }
    std::cout<<N<<" error"<<std::endl;
    file.close();
    RowVector3d miao;
    return miao;
}

double computeVolumeTetra(RowVector3d a, RowVector3d b, RowVector3d c, RowVector3d d){
    RowVector3d  v1=a-d;
    RowVector3d  v2=b-d;
    RowVector3d  v3=c-d;
    return abs(v1.dot(v2.cross(v3)))/6;
}

double computeVolumeMesh(double t){
    std::string line, word;
    std::ifstream file;
    file.open("table.txt");
    double vol=0;
    int i=0;
    while (getline(file, line))
    {
        std::stringstream ss1(line);
        int i1;
        int i2;
        int i3;
        int i4;
        ss1>>i1;
        ss1>>i2;
        ss1>>i3;
        ss1>>i4;
        RowVector3d a=getVector(i1,t);
        RowVector3d b=getVector(i2,t);
        RowVector3d c=getVector(i3,t);
        RowVector3d d=getVector(i4,t);
        vol=vol+computeVolumeTetra(a,b,c,d);
    }
    file.close();
    return vol;
}




int main(){
    std::vector<RowVector3d> start;
    std::vector<RowVector3d> end;
    std::vector<RowVector4i> indices;
    std::string line, word;
    std::ifstream file;
    int k=0;
    double temp;
    double max=0;
    for(int t=0; t<=100; t++){
        temp=computeVolumeMesh(0.01*t);
        std::cout<<temp<<std::endl;
        if(temp>max){
            max=temp;
        }
    }
    std::cout<<(computeVolumeMesh(0)-max)/(computeVolumeMesh(0))<<" "<<(computeVolumeMesh(0)-computeVolumeMesh(1))/(computeVolumeMesh(0))<<std::endl;
}


