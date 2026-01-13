# 对于任意我做的测试，我可以在mlir中任意定义函数，通过类似test-opt可以指定申明某个函数，只需在链接时链接对应的共享库

# 类似于libruntime.so定义的test_print_i64(int64_t v)