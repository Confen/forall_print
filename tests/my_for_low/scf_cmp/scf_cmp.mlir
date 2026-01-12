//注意使用memref,防止优化将scf直接优化为
//func.func @func{
//    return
// }

module {
  func.func @func() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %st = arith.constant 1 : index

    %c0 = arith.constant 0 : i32
    %mem = memref.alloca() : memref<i32>
    memref.store %c0, %mem[] : memref<i32>

    scf.for %a = %lb to %ub step %st{
        %one = arith.constant 1 : index
        %tmp = arith.addi %a, %one : index
        %v = arith.index_cast %tmp : index to i32
        memref.store %v, %mem[] : memref<i32>
        scf.yield
    }
    %r = memref.load %mem[] : memref<i32>
    func.return
}
}