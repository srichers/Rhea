mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
cmake --build . --config Release
