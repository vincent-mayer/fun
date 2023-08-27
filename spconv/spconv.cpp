#include <vector>
#include <iostream>

int getValidOutPos(const int *input_pos,
                   const int *kernelSize,
                   const int *stride, const int *padding,
                   const int *dilation,
                   const int *outSpatialShape, int *out)
{
    int lowers[3];
    int uppers[3];
    int counter[3];
    int counterSize[3];
    int pointCounter = 0;
    int val;
    int numPoints = 1;
    int m, offset;
    bool valid = false;
    for (unsigned i = 0; i < 3; ++i)
    {
        lowers[i] = (input_pos[i] - (kernelSize[i] - 1) * dilation[i] - 1 +
                     stride[i] + padding[i]) /
                    stride[i];
        uppers[i] = (input_pos[i] + padding[i]) / stride[i];
    }

    for (unsigned i = 0; i < 3; ++i)
    {
        counterSize[i] = ((uppers[i] - lowers[i]) / dilation[i] + 1);
        numPoints *= counterSize[i];
    }

    for (unsigned i = 0; i < 3; ++i)
    {
        counter[i] = 0;
    }
    for (int i = 0; i < numPoints; ++i)
    {
        valid = true;
        m = 1;
        offset = 0;
        for (int j = 3 - 1; j >= 0; --j)
        {
            val = uppers[j] - counter[j];
            out[pointCounter * 3 + j] = val;
            if (val < 0 || (val > outSpatialShape[j] - 1))
            {
                valid = false;
                // break;
            }
            offset += m * (input_pos[j] - val * stride[j] + padding[j]);
            m *= kernelSize[j];
        }

        out[pointCounter * 3 + 3] = offset;
        if (valid)
            ++pointCounter;
        counter[3 - 1] += 1;
        for (int c = 3 - 1; c >= 0; --c)
        {
            if (counter[c] == counterSize[c] && c > 0)
            {
                counter[c - 1] += 1;
                counter[c] = 0;
            }
        }
    }
    return pointCounter;
}

int main(int argc, char *argv[])
{
    auto kernelSize = std::vector<int>{3, 3, 3};
    auto stride = std::vector<int>{2, 2, 2};
    auto padding = std::vector<int>{1, 1, 1};
    auto dilation = std::vector<int>{1, 1, 1};
    auto outSpatialShape = std::vector<int>{1440, 1440, 50};

    auto indicesIn = std::vector<int>{552, 342, 20};
    std::vector<int> validPoints(27 * 4);

    auto numValidPoints = getValidOutPos(indicesIn.data(), kernelSize.data(), stride.data(), padding.data(), dilation.data(), outSpatialShape.data(), validPoints.data());
    std::cout << validPoints[0] << validPoints[1] << validPoints[2] << std::endl;
}