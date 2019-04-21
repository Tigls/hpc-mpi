cmake_minimum_required(VERSION 3.13)
project(test C)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(
        test
        main.c
)

target_link_libraries(test ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS} m)