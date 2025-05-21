#include <iostream>
#include "tensor_mul.h"
#include "tensor_display.h"
#include "string_to_tensor.h"

int main() {

    std::vector<std::vector<std::vector<int>>>* m1 = nullptr;
    m1 = new std::vector<std::vector<std::vector<int>>>;
    *m1 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};

    std::vector<std::vector<std::vector<int>>>* m2 = nullptr;
    m2 = new std::vector<std::vector<std::vector<int>>>;
    *m2 = {{{9, 8}, {7, 6}}, {{5, 4}, {3, 2}}};

    std::string s1 = "[[1,2,3][4,5,6]]";
    std::string s2 = "[[7,8][9,10][11,12]]";

    std::cout << "\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n";
    std::cout << "please input your matrices in the following format:\n\n";
    std::cout << "m1 = [[1,2,3][4,5,6]]\n";
    std::cout << "m2 = [[7,8][9,10][11,12]]\n\n";
    std::cout << "this means (1) NO spaces and (2) NO commas between matrices\n\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "\n";
    
    std::cout << "m1 = ";
    std::cin >> s1;
    std::cout << "m2 = ";
    std::cin >> s2;
    std::cout << "\n";

    *m1 = string_to_matrix(s1);
    *m2 = string_to_matrix(s2);

    std::vector<std::vector<std::vector<int>>>* m = nullptr;
    m = new std::vector<std::vector<std::vector<int>>>;
    *m = matmul(m1, m2);

    std::cout << "RESULT \n";
    display_matrix(*m);
    std::cout << "\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    delete m1; delete m2; delete m;

    return 0;
}