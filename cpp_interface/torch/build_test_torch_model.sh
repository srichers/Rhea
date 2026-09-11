/home/srichers/.local/bin/cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ -S/mnt/scratch/srichers/software/Rhea/cpp_interface -B/mnt/scratch/srichers/software/Rhea/build -G Ninja

/home/srichers/.local/bin/cmake --build /mnt/scratch/srichers/software/Rhea/build --config Debug --target all --
