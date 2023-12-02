if [ ! -d build/cpu_distributed ]; then
    echo "build/cpu_distributed does not exist. Creating directory...";
    mkdir -p build/cpu_distributed;
fi
if [ ! -d build/cpu_distributed/CMakeFiles ]; then
    echo "No cmake files found. Running cmake...";
    cmake -B build/cpu_distributed -S src/cpu_distributed;
fi

cd build/cpu_distributed;
make;
if [ -f "cpu_distributed" ]; then
    cd ../..;
    if [ ! -z "$1" ]; then
        mpiexec -n $1 build/cpu_distributed/cpu_distributed;
    else echo "ERR: Must supply process count as command line argument";
    fi
fi
