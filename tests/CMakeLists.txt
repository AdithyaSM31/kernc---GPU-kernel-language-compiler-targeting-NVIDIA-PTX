find_package(GTest REQUIRED)

# 1. Unit Tests
add_executable(kernc_unit_tests
    unit/lexer_test.cpp
    unit/parser_test.cpp
    unit/sema_test.cpp
    unit/ir_pass_test.cpp
)
target_link_libraries(kernc_unit_tests PRIVATE GTest::GTest GTest::Main kernc_lib)
include(GoogleTest)
gtest_discover_tests(kernc_unit_tests)

# 2. End-to-End GPU Tests
if (CUDA_FOUND AND CUDA_DRIVER_LIB)
    add_executable(kernc_e2e_tests
        e2e/test_kernels.cpp
    )
    target_include_directories(kernc_e2e_tests PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(kernc_e2e_tests PRIVATE GTest::GTest kernc_lib ${CUDA_DRIVER_LIB})
    gtest_discover_tests(kernc_e2e_tests)
endif()

# 3. Diagnostic Tests
add_test(NAME diagnostic_tests
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/diagnostic/run_diagnostic_tests.py $<TARGET_FILE:kernc> ${CMAKE_CURRENT_SOURCE_DIR}/diagnostic
)

# 4. PTX Acceptance Tests (ptxas sm_86)
if (PTXAS_BIN)
    set(PTX_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx_accept")
    file(MAKE_DIRECTORY "${PTX_TEST_DIR}")

    set(KERNELS vecadd saxpy clamp relu)
    foreach(kern ${KERNELS})
        set(KERN_SRC "${CMAKE_SOURCE_DIR}/examples/${kern}.kern")
        set(OUT_PTX "${PTX_TEST_DIR}/${kern}.ptx")
        set(OUT_CUBIN "${PTX_TEST_DIR}/${kern}.cubin")

        add_test(NAME ptx_accept_${kern}
            COMMAND bash -c "\"$<TARGET_FILE:kernc>\" \"${KERN_SRC}\" -o \"${OUT_PTX}\" && \"${PTXAS_BIN}\" -arch=sm_86 \"${OUT_PTX}\" -o \"${OUT_CUBIN}\""
        )
    endforeach()

    # Regression kernels: compiled at both -O0 and -O1 so the optimizer cannot hide a codegen bug.
    file(GLOB REGRESS_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/regress/*.kern")
    foreach(KERN_SRC ${REGRESS_KERNELS})
        get_filename_component(kern ${KERN_SRC} NAME_WE)
        foreach(opt O0 O1)
            set(OUT_PTX "${PTX_TEST_DIR}/regress_${kern}_${opt}.ptx")
            add_test(NAME ptx_regress_${kern}_${opt}
                COMMAND bash -c "\"$<TARGET_FILE:kernc>\" \"${KERN_SRC}\" -${opt} -o \"${OUT_PTX}\" && \"${PTXAS_BIN}\" -arch=sm_86 \"${OUT_PTX}\" -o /dev/null"
            )
        endforeach()
    endforeach()
endif()
