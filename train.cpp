// TODO: mutithread tensor ops and maxpool
// TODO: add gpu support for linux and windows
// TODO: tile dw and dx shaders for conv2d backward pass
// TODO: make tensor class tempelatable
// TODO: add shaders for reduce sum / layer normalization
// TODO: use function pointers in shaders
// TODO: GPU version of transpose (should be simple flat gx=256, gy=1, gz=1)

#include "example_models.h"

int main()
{
    img_class();
    return 0;
}