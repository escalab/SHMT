#include "cuda_utils.h"

int snapTransformSize(int dataSize)
{
    int hiBit;
    unsigned int lowPOT, hiPOT;
 
    dataSize = iAlignUp(dataSize, 16);
 
    for (hiBit = 31; hiBit >= 0; hiBit--)
        if (dataSize & (1U << hiBit))
        {
            break;
        }

    lowPOT = 1U << hiBit;
 
    if (lowPOT == (unsigned int)dataSize)
    {
        return dataSize;
    }

    hiPOT = 1U << (hiBit + 1);

    if (hiPOT <= 1024)
    {
        return hiPOT;
    }
    else
    {
        return iAlignUp(dataSize, 512);
    }
}
