module {
  llvm.func @forall() {
    %0 = llvm.mlir.constant(5 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(10 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.icmp "slt" %4, %2 : i64
    llvm.cond_br %6, ^bb2, ^bb3(%1 : i64)
  ^bb2:  // pred: ^bb1
    %7 = "test.print"(%5) : (index) -> index
    %8 = llvm.add %4, %3 : i64
    llvm.br ^bb1(%8 : i64)
  ^bb3(%9: i64):  // 2 preds: ^bb1, ^bb4
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %11 = llvm.icmp "slt" %9, %0 : i64
    llvm.cond_br %11, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %12 = "test.print"(%10) : (index) -> index
    %13 = llvm.add %9, %3 : i64
    llvm.br ^bb3(%13 : i64)
  ^bb5:  // pred: ^bb3
    llvm.return
  }
}

