#include<vector>
#include<Eigen/Dense>
#include<sstream>
#include<fstream>
#include <stdlib.h>
#include<iostream>
using Eigen::RowVector3d;
using Eigen::RowVector4i;


double computeVolumeTetra(RowVector3d a, RowVector3d b, RowVector3d c, RowVector3d d){
   RowVector3d  v1=a-d;
   RowVector3d  v2=b-d;
   RowVector3d  v3=c-d;
return abs(v1.dot(v2.cross(v3)))/6;
}

double computeVolumeMesh(std::vector<RowVector3d> points,std::vector<RowVector4i> tetra){
    double vol=0;
    int N=tetra.size();
    for(int i=0; i<N; i++){
        vol=vol+computeVolumeTetra(points[tetra[i][0]],points[tetra[i][1]],points[tetra[i][2]],points[tetra[i][3]]);
    }
    return vol;
}

std::vector<RowVector3d> transMesh(std::vector<RowVector3d> start, std::vector<RowVector3d> end, double t){
    int N=start.size();
    std::vector<RowVector3d> temp;
    for (int i=0; i<N; i++){
        temp.push_back((1-t)*start[i]+t*end[i]);
    }
    return temp;
}



int main(){
    std::vector<RowVector3d> start;
    std::vector<RowVector3d> end;
    std::vector<RowVector4i> indices;
    std::string line, word;
    std::ifstream file;
    file.open("morph.tet6");
    int counterlines=0;
    int k=0;
    while (getline(file, line))
    {
        //skip first two lines
        if (counterlines>2){
            std::stringstream ss1(line);
            k=0;
            while(ss1.rdbuf()->in_avail() != 0){
                ss1>>word;
                k=k+1;
            }
            //line cointains points
            if (k==7){
                RowVector3d temp0;
                RowVector3d temp1;
                std::stringstream ss2(line);
                for(int i=0; i<3; i++){
                    ss2>>temp0[i];
                }
                for(int i=0; i<3; i++){
                    ss2>>temp1[i];
                }
                start.push_back(temp0);
                end.push_back(temp1);
            }
            if (k==5){

                RowVector4i temp;
                std::stringstream ss2(line);
                ss2>>word;
                for(int i=0;i<4;i++){
                    ss2>>temp[i];
                }
                indices.push_back(temp);
            }
        }
        counterlines++;
    }
    double temp;
    double max=0;
    for(int t=0; t<=100; t++){
        temp=computeVolumeMesh(transMesh(start,end,0.01*t),indices);
        std::cout<<temp<<std::endl;
        if(temp>max){
            max=temp;
        }
    }
    std::cout<<"max relativa difference"<<(computeVolumeMesh(start,indices)-temp)/(computeVolumeMesh(start,indices))<<std::endl;
    std::cout<<"start end relative difference"<<(computeVolumeMesh(start,indices)-computeVolumeMesh(end,indices))/(computeVolumeMesh(start,indices))<<std::endl;
}


