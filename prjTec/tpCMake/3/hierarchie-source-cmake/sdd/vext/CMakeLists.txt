add_executable(test-vext-1 test-vext-1.c)
target_link_libraries(test-vext-1 sdd bcb)
add_test(test-vext-1 test-vext-1)

add_executable(test-vext-2 test-vext-2.c)
target_link_libraries(test-vext-2 sdd bcb)
add_test(test-vext-2 test-vext-2)

add_executable(test-vext-3 test-vext-3.c)
target_link_libraries(test-vext-3 sdd bcb)
add_test(test-vext-3 test-vext-3)

add_executable(test-vext-4 test-vext-4.c)
target_link_libraries(test-vext-4 sdd bcb)
add_test(test-vext-4 test-vext-4)

add_executable(test-vext-5 test-vext-5.c)
target_link_libraries(test-vext-5 sdd bcb)
add_test(test-vext-5 test-vext-5)

add_executable(test-vext-6 test-vext-6.c)
target_link_libraries(test-vext-6 sdd bcb)
add_test(test-vext-6 test-vext-6)

set_tests_properties(test-vext-1 test-vext-2 test-vext-3 test-vext-4 test-vext-5 test-vext-6 PROPERTIES TIMEOUT 10)
