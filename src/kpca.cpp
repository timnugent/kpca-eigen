// Kernel PCA using the Eigen library, by Tim Nugent 2014

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;

class PCA{

public:
	PCA() : components(2), kernel_type(1), normalise(0), gamma(0.001), constant(1.0), order(2.0) {}
	explicit PCA(MatrixXd& d) : components(2), kernel_type(1), normalise(0), gamma(0.001), constant(1.0), order(2.0) {X = d;}
	void load_data(const char* data, char sep = ',');
	void set_components(const int i){components = i;};
	void set_kernel(const int i){kernel_type = i;};	
	void set_normalise(const int i){normalise = i;};
	void set_gamma(const double i){gamma = i;};
	void set_constant(const int i){constant = i;};
	void set_order(const int i){order = i;};
	MatrixXd& get_transformed(){return transformed;}	
	void run_pca();
	void run_kpca();
	void print();
	void write_transformed(string);
	void write_eigenvectors(string);
private:
	double kernel(const VectorXd& a, const VectorXd& b);
	MatrixXd X, Xcentered, C, K, eigenvectors, transformed;
	VectorXd eigenvalues, cumulative;
	unsigned int components, kernel_type, normalise;
	double gamma, constant, order;

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
		Xcentered.resize(X.rows(),X.cols());
	}else{
		cout << "Failed to read file " << data << endl;
	}

}

double PCA::kernel(const VectorXd& a, const VectorXd& b){

	/*
		Kernels
		1 = RBF
		2 = Polynomial
		TODO - add some of these these:
		http://crsouza.blogspot.co.uk/2010/03/kernel-functions-for-machine-learning.html	

	*/
	switch(kernel_type){
	    case 2  :
	    	return(pow(a.dot(b)+constant,order));
	    default : 
	    	return(exp(-gamma*((a-b).squaredNorm())));
	}

}

void PCA::run_kpca(){

	// Fill kernel matrix
	K.resize(X.rows(),X.rows());
	for(unsigned int i = 0; i < X.rows(); i++){
		for(unsigned int j = i; j < X.rows(); j++){
			K(i,j) = K(j,i) = kernel(X.row(i),X.row(j));
			//printf("k(%i,%i) = %f\n",i,j,K(i,j));
		}	
	}	
	//cout << endl << K << endl;

	EigenSolver<MatrixXd> edecomp(K);
	eigenvalues = edecomp.eigenvalues().real();
	eigenvectors = edecomp.eigenvectors().real();
	cumulative.resize(eigenvalues.rows());
	vector<pair<double,VectorXd> > eigen_pairs; 
	double c = 0.0; 
	for(unsigned int i = 0; i < eigenvectors.cols(); i++){
		if(normalise){
			double norm = eigenvectors.col(i).norm();
			eigenvectors.col(i) /= norm;
		}
		eigen_pairs.push_back(make_pair(eigenvalues(i),eigenvectors.col(i)));
	}
	// http://stackoverflow.com/questions/5122804/sorting-with-lambda
	sort(eigen_pairs.begin(),eigen_pairs.end(), [](const pair<double,VectorXd> a, const pair<double,VectorXd> b) -> bool {return (a.first > b.first);} );
	for(unsigned int i = 0; i < eigen_pairs.size(); i++){
		eigenvalues(i) = eigen_pairs[i].first;
		c += eigenvalues(i);
		cumulative(i) = c;
		eigenvectors.col(i) = eigen_pairs[i].second;
	}
	transformed.resize(X.rows(),components);

	for(unsigned int i = 0; i < X.rows(); i++){
		for(unsigned int j = 0; j < components; j++){
			for (int k = 0; k < K.rows(); k++){
                transformed(i,j) += K(i,k) * eigenvectors(k,j);
		 	}
		}
	}	

	/*
	cout << "Input data:" << endl << X << endl << endl;
	cout << "Centered data:"<< endl << Xcentered << endl << endl;
	cout << "Centered kernel matrix:" << endl << Kcentered << endl << endl;
	cout << "Eigenvalues:" << endl << eigenvalues << endl << endl;	
	cout << "Eigenvectors:" << endl << eigenvectors << endl << endl;	
	*/
	cout << "Sorted eigenvalues:" << endl;
	for(unsigned int i = 0; i < eigenvalues.rows(); i++){
		if(eigenvalues(i) > 0){
			cout << "PC " << i+1 << ": Eigenvalue: " << eigenvalues(i);
			printf("\t(%3.3f of variance, cumulative =  %3.3f)\n",eigenvalues(i)/eigenvalues.sum(),cumulative(i)/eigenvalues.sum());
		}
	}
	cout << endl;
	//cout << "Sorted eigenvectors:" << endl << eigenvectors << endl << endl;	
	//cout << "Transformed data:" << endl << transformed << endl << endl;	
}

void PCA::run_pca(){

	// http://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
	Xcentered = X.rowwise() - X.colwise().mean();	
	C = (Xcentered.adjoint() * Xcentered) / double(X.rows());
	EigenSolver<MatrixXd> edecomp(C);
	eigenvalues = edecomp.eigenvalues().real();
	eigenvectors = edecomp.eigenvectors().real();
	cumulative.resize(eigenvalues.rows());
	vector<pair<double,VectorXd> > eigen_pairs; 
	double c = 0.0; 
	for(unsigned int i = 0; i < eigenvectors.cols(); i++){
		if(normalise){
			double norm = eigenvectors.col(i).norm();
			eigenvectors.col(i) /= norm;
		}
		eigen_pairs.push_back(make_pair(eigenvalues(i),eigenvectors.col(i)));
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
		if(eigenvalues(i) > 0){
			cout << "PC " << i+1 << ": Eigenvalue: " << eigenvalues(i);
			printf("\t(%3.3f of variance, cumulative =  %3.3f)\n",eigenvalues(i)/eigenvalues.sum(),cumulative(i)/eigenvalues.sum());
		}
	}
	cout << endl;
	cout << "Sorted eigenvectors:" << endl << eigenvectors << endl << endl;	
	cout << "Transformed data:" << endl << X * eigenvectors << endl << endl;	
	//cout << "Transformed centred data:" << endl << transformed << endl << endl;	

}

void PCA::write_transformed(string file){

	ofstream outfile(file);
	for(unsigned int i = 0; i < transformed.rows(); i++){
		for(unsigned int j = 0; j < transformed.cols(); j++){
			outfile << transformed(i,j);
			if(j != transformed.cols()-1) outfile << ",";
		}	
		outfile << endl;
	}
	outfile.close();
	cout << "Written file " << file << endl;

}	

void PCA::write_eigenvectors(string file){

	ofstream outfile(file);
	for(unsigned int i = 0; i < eigenvectors.rows(); i++){
		for(unsigned int j = 0; j < eigenvectors.cols(); j++){
			outfile << eigenvectors(i,j);
			if(j != eigenvectors.cols()-1) outfile << ",";
		}	
		outfile << endl;
	}
	outfile.close();
	cout << "Written file " << file << endl;

}	

int main(int argc, const char* argv[]){

	/*
	if(argc < 2){
		cout << "Usage:\n" << argv[0] << " <DATA>" << endl;
		cout << "File format:\nX1,X2, ... Xn\n";
		return(0);
	}
	*/

	PCA* P = new PCA();
	P->load_data("data/test.data");
	P->run_pca();
	cout << "Regular PCA (data/test.data):" << endl;
	P->run_pca();
	P->print();
	delete P;

	P = new PCA();
	P->load_data("data/wikipedia.data");
	cout << "Kernel PCA (data/wikipedia.data) - RBF kernel, gamma = 0.001:" << endl;
	P->run_kpca();
	P->write_eigenvectors("data/eigenvectors_RBF_data.csv");
	P->write_transformed("data/transformed_RBF_data.csv");
	cout << endl;	
	delete P;

	P = new PCA();
	P->load_data("data/wikipedia.data");
	P->set_kernel(2);
	P->set_constant(1);
	P->set_order(2);
	cout << "Kernel PCA (data/wikipedia.data) - Polynomial kernel, order = 2, constant = 1:" << endl;
	P->run_kpca();
	P->write_eigenvectors("data/eigenvectors_Polynomial_data.csv");
	P->write_transformed("data/transformed_Polynomial_data.csv");	
	cout << endl;	
	delete P;

	return(0);

}

