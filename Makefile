CXX := g++
SRCS := example_matmul.cpp tensor_mul.cpp tensor_display.cpp string_to_tensor.cpp
TARGET := tensorMul

all:
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)
clean:
	rm -f $(TARGET)
