执行指令： build/test-opt --allow-unregistered-dialect --strip-test-print forall_low.mlir | mlir-translate -mlir-to-llvmir

lli -load=./libruntime.so -entry-function=forall forall_low_mul.bc
