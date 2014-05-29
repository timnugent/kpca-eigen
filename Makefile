CPP = g++
CFLAGS = -Wall -Wextra -Werror -Wno-unused-parameter -O3 -std=c++11
INC = -I/usr/include/eigen3

all: kpca

kpca: src/kpca.cpp
	$(CPP) $(CFLAGS) $(INC) src/kpca.cpp ${LIBS} -o bin/kpca

clean:
	rm bin/kpca data/transformed_*.csv data/eigenvectors_*.csv *.png &> /dev/null

test:
	bin/kpca
	@echo "Plotting data using R..."
	src/plot.R

