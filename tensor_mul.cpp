#include <iostream>
#include "tensor_mul.h"

std::vector<std::vector<std::vector<int>>> matmul(
    std::vector<std::vector<std::vector<int>>>* m1, 
    std::vector<std::vector<std::vector<int>>>* m2) {
    
    int batch = static_cast<int>((*m1).size());
    int n = static_cast<int>((*m1)[0][0].size());
    int n1 = static_cast<int>((*m1)[0].size());
    int n2 = static_cast<int>((*m2)[0][0].size());

    std::vector<std::vector<std::vector<int>>> m(batch, std::vector<std::vector<int>>(n1, std::vector<int>(n2, 0)));

    // ensure m1 cols == m2 rows
    if (n != static_cast<int>((*m2)[0].size())) {
        std::cout << "ERROR!" << std::endl;
        return m;
    }

    for (int b = 0; b < batch; b++){
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n1; k++) {
                    m[b][i][k] += (*m1)[b][i][j] * (*m2)[b][j][k];
                }
            }
        }
    }
    return m;
}
