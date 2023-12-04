# include<iostream>
# include<vector>
# include<cstdint>
# include<fstream>

int main(int argc, char* argv[]){
    std::vector<int16_t> firstDataset(6000*6000);
    std::vector<int16_t> secondDataset(6000*6000);

    std::ifstream file1(argv[1], std::ios::binary);
    file1.read(reinterpret_cast<char*>(firstDataset.data()), 6000*6000 * sizeof(int16_t));
    file1.close();
    
    std::ifstream file2(argv[2], std::ios::binary);
    file2.read(reinterpret_cast<char*>(secondDataset.data()), 6000*6000 * sizeof(int16_t));
    file2.close();

    bool isEqual = true;
    for(int i = 0; i < firstDataset.size(); i++){
        if(firstDataset[i] != secondDataset[i]){
            std::cout << "disrepency detected at " << i << ": first dataset: " << firstDataset[i] << ", second dataset: " << secondDataset[i] << std::endl;
            isEqual = false;
            break;
        }
    }
    if (isEqual) {
        std::cout << "Implementation outputs match!" << std::endl;
    }
    else {
        std::cout << "Implementation outputs DO NOT match" << std::endl;
    }

    return 0;
}

