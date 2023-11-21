if [ ! -d build/cpu_shared ]; then
    echo "build/cpu_shared does not exist. Creating directory...";
    mkdir -p build/cpu_shared;
fi
if [ ! -d build/cpu_shared/CMakeFiles ]; then
    echo "No cmake files found. Running cmake...";
    cmake -B build/cpu_shared -S src/cpu_shared;
fi
cd build/cpu_shared;
make;
if [ -f "cpu_shared" ]; then
    cd ../..;
    if [ ! -z "$1" ]; then
        build/cpu_shared/cpu_shared $1;
    else echo "ERR: Must supply thread count as a command line argument";
    fi
fi