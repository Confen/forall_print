
func.func @forall() {
  scf.forall (%i) in (10) {
    // %i = 0..9，各迭代可并行
    "test.print"(%i) : (index) -> ()
  }
  scf.forall (%j) in (5) {
    "test.print"(%j) : (index) -> ()
  }
  return
}

