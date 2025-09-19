# Modify test environment.
# See also https://cmake.org/cmake/help/latest/prop_test/ENVIRONMENT_MODIFICATION.html.
function(test_environment NAME VARIABLE OP VALUE)

    set_tests_properties(
        ${NAME}
        PROPERTIES
            ENVIRONMENT_MODIFICATION ${VARIABLE}=${OP}:${VALUE}
    )

endfunction()
