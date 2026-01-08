module {
  llvm.func @test_print_i64(i64)
  llvm.func @forall() {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(10 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
    %4 = llvm.icmp "slt" %3, %1 : i64
    llvm.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.add %3, %5 : i64
    llvm.call @test_print_i64(%6) : (i64) -> ()
    %7 = llvm.add %3, %2 : i64
    llvm.br ^bb1(%7 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
}

