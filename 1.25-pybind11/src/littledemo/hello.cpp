#include <pybind11/pybind11.h>
#include <iostream>

void f(){
	std::cout << "hello world \n" << std::endl;
}

PYBIND11_MODULE(handsome, m){
	m.doc() = "a stupid function, print hello world";
	m.def("hello", &f, "print hello world");
}
