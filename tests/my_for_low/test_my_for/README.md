# 执行流水线

mlir-opt --allow-unregistered-dialect --test-lower-to-llvm tests/my_for_low/test.mlir -o test_low.mlir

build/test-opt --allow-unregistered-dialect --test-lower-my-for tests/my_for_low/test_low.mlir -o test_low_for.mlir

mlir-translate tests/my_for_low/test_low_for.mlir --allow-umregistered-dialect  --mlir-to-llvmir  -o tests/my_for_low/test_low_for.ll

llvm-as tests/my_for_low/test_low_for.ll -o tests/my_for_low/test_low_for.bc

lli tests/my_for_low/test_low_for.bc

# 返回的应该是退出循环的临界值
echo $?