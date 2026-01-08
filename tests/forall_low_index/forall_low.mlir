module {
  llvm.func @forall() {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(10 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
    %4 = builtin.unrealized_conversion_cast %3 : i64 to index
    %5 = llvm.icmp "slt" %3, %1 : i64
    llvm.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    "test.print"(%4) : (index) -> (index)
    %6 = llvm.add %3, %2 : i64
    llvm.br ^bb1(%6 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
}

