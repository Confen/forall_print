module {
  func.func @forall() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c10 step %c1 {
      "test.print"(%arg0) : (index) -> ()
    }
    %c0_0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1_1 = arith.constant 1 : index
    scf.for %arg0 = %c0_0 to %c5 step %c1_1 {
      "test.print"(%arg0) : (index) -> ()
    }
    return
  }
}

