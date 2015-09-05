#include <iostream>
#include "ICM.h"
#include <fstream>
#include <string>
int main()
{
    ImageDenoisingProcessor processor;
    processor.IteratedConditionalModes();
    processor.RecoverImage();
}