#include <iostream>
#include "string_to_tensor.h"

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