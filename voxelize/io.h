#include <fstream>
#include <iostream>
#include <vector>

auto readNumpyFloatArray(std::string filename) -> std::vector<float>
{
    // Open the binary file
    std::ifstream file(filename, std::ios::binary);

    // Read the size of the array
    int size;
    file.read(reinterpret_cast<char *>(&size), sizeof(int));
    std::cout << "Total size of file " << filename << " " << size << std::endl;

    // Read the array elements
    std::vector<float> array(size);
    file.read(reinterpret_cast<char *>(array.data()), size * sizeof(float));

    // Close the file
    file.close();

    return array;
}