if [ ! -d build/serial ]; then
    echo "build/serial does not exist. Creating directory...";
    mkdir -p build/serial;
fi
if [ ! -d build/serial/CMakeFiles ]; then
    echo "No cmake files found. Running cmake...";
    cmake -B build/serial -S src/serial;
fi
cd build/serial;
make;
if [ -f "serial" ]; then
    cd ../..;
    build/serial/serial;
fi