// PCA using Eigen library by Tim Nugent 2014

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;

class PCA{

public:
	PCA(){}
	void load_data(const char* data, char sep = ',');
	void run();
	void print();
	void write();
private:
	MatrixXd X, Xcentered, C, eigenvectors, transformed;
	VectorXd eigenvalues, cumulative;
	double eigensum;

};

void PCA::load_data(const char* data, char sep){

	// Read data
	unsigned int row = 0;
	ifstream file(data);
	if(file.is_open()){
		string line,token;
		while(getline(file, line)){
			stringstream tmp(line);
			unsigned int col = 0;
			while(getline(tmp, token, sep)){
				if(X.rows() < row+1){
					X.conservativeResize(row+1,X.cols());
				}
				if(X.cols() < col+1){
					X.conservativeResize(X.rows(),col+1);
				}
				X(row,col) = atof(token.c_str());
				col++;
			}
			row++;
		}
		file.close();
	}else{
		cout << "Failed to read file " << data << endl;
	}

}

void PCA::run(){

	// Mean centre and calculate covariance matrix:
	// http://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
	Xcentered = X.rowwise() - X.colwise().mean();
	C = (Xcentered.adjoint() * Xcentered) / double(X.rows());
	EigenSolver<MatrixXd> edecomp(C);
	eigenvalues = edecomp.eigenvalues().real();
	eigenvectors = edecomp.eigenvectors().real();
	cumulative.resize(eigenvalues.rows());
	vector<pair<double,VectorXd> > eigen_pairs; 
	double c = 0.0; 
	eigensum = 0.0;  
	for(unsigned int i = 0; i < eigenvectors.cols(); i++){
		eigen_pairs.push_back(make_pair(eigenvalues(i),eigenvectors.col(i)));
		eigensum += eigenvalues(i);
	}
    	// http://stackoverflow.com/questions/5122804/sorting-with-lambda
    	sort(eigen_pairs.begin(),eigen_pairs.end(), [](const pair<double,VectorXd> a, const pair<double,VectorXd> b) -> bool {return (a.first > b.first);} );
    	for(unsigned int i = 0; i < eigen_pairs.size(); i++){
		eigenvalues(i) = eigen_pairs[i].first;
		c += eigenvalues(i);
		cumulative(i) = c;
		eigenvectors.col(i) = eigen_pairs[i].second;
    	}
    	transformed = Xcentered * eigenvectors;

}

void PCA::print(){

	cout << "Input data:" << endl << X << endl << endl;
	cout << "Centered data:"<< endl << Xcentered << endl << endl;
	cout << "Covariance matrix:" << endl << C << endl << endl;
	cout << "Eigenvalues:" << endl << eigenvalues << endl << endl;	
	cout << "Eigenvectors:" << endl << eigenvectors << endl << endl;	
	cout << "Sorted eigenvalues:" << endl;
    	for(unsigned int i = 0; i < eigenvalues.rows(); i++){
		cout << "PC " << i+1 << ": Eigenvalue: " << eigenvalues(i);
		printf("\t(%3.3f of variance, cumulative =  %3.3f)\n",eigenvalues(i)/eigensum,cumulative(i)/eigensum);
		//cout << eigenvectors.col(i) << endl << endl;
    	}
    	cout << endl;
    	cout << "Sorted eigenvectors:" << endl << eigenvectors << endl << endl;	
	cout << "Transformed data:" << endl << X * eigenvectors << endl << endl;	
	cout << "Transformed centred data:" << endl << transformed << endl << endl;	

}

void PCA::write(){

	ofstream outfile("eigenvectors.csv");
	for(unsigned int i = 0; i < eigenvectors.rows(); i++){
		for(unsigned int j = 0; j < eigenvectors.cols(); j++){
			outfile << eigenvectors(i,j);
			if(j != eigenvectors.cols()-1) outfile << ",";
		}	
		outfile << endl;
	}
	outfile.close();
	cout << "Written file eigenvectors.csv" << endl;

	outfile.open("transformed_data.csv");
	for(unsigned int i = 0; i < transformed.rows(); i++){
		for(unsigned int j = 0; j < transformed.cols(); j++){
			outfile << transformed(i,j);
			if(j != transformed.cols()-1) outfile << ",";
		}	
		outfile << endl;
	}
	outfile.close();
	cout << "Written file transformed_data.csv" << endl;

}	

int main(int argc, const char* argv[]){

	if(argc < 2){
		cout << "Usage:\n" << argv[0] << " <DATA>" << endl;
		cout << "File format:\nX1,X2, ... Xn\n";
		return(0);
	}
	PCA* P = new PCA();
	P->load_data(argv[1]);
	P->run();
	P->print();
	P->write();
	delete P;
	return(0);

}

