# Mark a test as skipped if all collected tests were marked as skipped.
#
# The test is marked as skipped if pytest outputs a message similar to:
#   === 1 skipped in 0.01s ===
#
# See also https://cmake.org/cmake/help/latest/prop_test/SKIP_REGULAR_EXPRESSION.html#prop_test:SKIP_REGULAR_EXPRESSION.
function(pytest_skip_mark TESTNAME)
    set_property(
        TEST ${TESTNAME}
        PROPERTY
            SKIP_REGULAR_EXPRESSION "=== [0-9]+ skipped in [0-9.]+s ==="
    )
endfunction()
