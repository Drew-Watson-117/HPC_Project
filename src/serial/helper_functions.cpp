#include "helper_functions.hpp"

std::vector<int16_t> getData() {
    std::ifstream file("data/srtm_14_04_6000x6000_short16.raw",std::ios::binary);
    std::vector<int16_t> data(6000*6000);
    file.read(reinterpret_cast<char*>(data.data()), 6000*6000 * sizeof(int16_t));
    file.close();
    return data;
}

bool writeFile(std::vector<int> data, char* fileName) {
    std::ofstream file;
    file.open(fileName, std::ios::binary);
    if (file.is_open()) {
        char* buffer = reinterpret_cast<char*>(data.data());
        size_t size = data.size() * sizeof(int);
        file.write(buffer, size);
        file.close();
        free(buffer);
        return true;
    } else {
        return false;
    }
}

int getIndex(int16_t x, int16_t y, int16_t xDim) {
    return xDim*y+x;
}

bool pixelInMap(int16_t pixelX, int16_t pixelY) {
    return !(pixelX > 5999 || pixelX < 0 || pixelY < 0 || pixelY > 5999);
}

bool inBoundary(int16_t centerX, int16_t centerY, int16_t considerX, int16_t considerY) {
    int16_t xDiff = std::abs(considerX - centerX);
    int16_t yDiff = std::abs(considerY - centerY);

    return (pixelInMap(considerX,considerY) && xDiff < 100 && yDiff < 100);
}