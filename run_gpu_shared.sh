if [ ! -d build/gpu_shared ]; then
    echo "build/gpu_shared does not exist. Creating directory...";
    mkdir -p build/gpu_shared;
fi
if [ ! -d build/gpu_shared/CMakeFiles ]; then
    echo "No cmake files found. Running cmake...";
    cmake -B build/gpu_shared -S src/gpu_shared;
fi
cd build/gpu_shared;
make;
if [ -f "gpu_shared" ]; then
   cd ../..;
    build/gpu_shared/gpu_shared;
fi