#include <iostream>

void display_matrix(std::vector<std::vector<std::vector<int>>> m) {
    for (std::vector<std::vector<int>> i : m){
        std::cout << "[ ";
        for (std::vector<int> j : i){
            std::cout << " [ ";
            for (int k : j){
                std::cout << k << " ";
            }
            std::cout << " ] ";
        }
        std::cout << " ] \n";
        
    }
}

std::vector<std::vector<std::vector<int>>> string_to_matrix(std::string s) {
    std::vector<std::vector<std::vector<int>>> t;
    std::vector<std::vector<int>> m;
    std::vector<int> r;
    std::string* temp = nullptr;
    temp = new std::string;

    for (int c = 0; c < static_cast<int>(s.length()); c++){
        
        if (s[c] != ']' && s[c] != ' ' && s[c] != '[') {
            if  (s[c] == ',') {

                // convert string of to int
                r.push_back(std::stoi(*temp));
                delete temp;
                temp = new std::string;

            } else {*temp += s[c];}

        } else if (s[c] == ']') {

            if (s[c-1] == ']') {
                t.push_back(m);
                m = {};

            } else{
                r.push_back(std::stoi(*temp));
                delete temp;
                temp = new std::string;
                m.push_back(r);
                r = {};
            }
        }
    }
    delete temp;
    return t;
}

std::vector<std::vector<std::vector<int>>> matmul(
    std::vector<std::vector<std::vector<int>>> * m1, 
    std::vector<std::vector<std::vector<int>>> * m2) {
    
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
    std::cout << "[[1,2,3][4,5,6]]\n\n";
    std::cout << "this means (1) NO spaces and (2) NO commas between matrices:\n\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
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
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    delete m1; delete m2; delete m;

    return 0;
}
