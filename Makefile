CPP = g++
CFLAGS = -Wall -Wextra -Werror -O3 -std=c++11
INC = -I/usr/include/eigen3

all: pca

pca: src/pca.cpp
	$(CPP) $(CFLAGS) $(INC) src/pca.cpp ${LIBS} -o bin/pca

clean:
	rm bin/pca eigenvectors.csv transformed_data.csv &> /dev/null

test:
	bin/pca data/wikipedia.data

