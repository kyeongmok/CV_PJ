CPP = g++

BLASOBJ = daxpy.o dscal.o ddot.o dnrm2.o
LIBLINOBJ = $(BLASOBJ) tron.o linear.o Liblinear.o
LIBLINSRC = daxpy.c dscal.c ddot.c dnrm2.c tron.cpp linear.cpp Liblinear.cpp
LBFSRC = CV_PJ.cpp DecisionTree.cpp LBF.cpp Shape.cpp
LBFOBJ = CV_PJ.o DecisionTree.o LBF.o Shape.o

TARGET = test
$(TARGET) : $(LIBLINOBJ) $(LBFOBJ)
	$(CPP) -o $(TARGET) $(LIBLINOBJ) $(LBFOBJ) `pkg-config opencv --libs`

$(BLAASOBJ) : blas.h
tron.o : tron.h 
linear.o : linear.h tron.h
Liblinear.o : Liblinear.h linear.h
CV_PJ.o : LBF.h Shape.h
DecisionTree.o : DecisionTree.h LBF.h
LBF.o : LBF.h Liblinear.h
Shape.o : Shape.h LBFDef.h

$(LBFOBJ) : $(LBFSRC)
	$(CPP) -c -std=c++11 $(LBFSRC) `pkg-config opencv --cflags`

$(LIBLINOBJ) : $(LIBLINSRC)
	$(CPP) -c $(LIBLINSRC)

clean:
	rm -rf $(LIBLINOBJ) $(LBFOBJ) $(TARGET)
