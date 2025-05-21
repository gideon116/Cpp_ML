#include <iostream>
#include "tensor_display.h"

void display_matrix(std::vector<std::vector<std::vector<int>>> m) {
    for (std::vector<std::vector<int>> i : m){
        std::cout << "[ ";
        for (std::vector<int> j : i){
            std::cout << " [ ";
            for (int k : j){
                std::cout << k << " ";
            }
            std::cout << "] ";
        }
        std::cout << "] \n";
        
    }
}