// RUN: test-opt %s -test-lower-my-for -split-input-file | FileCheck %s

// ----- case 1: high-level input (func+arith) -----
// 这段用于验证：高层 IR 能 parse，my.for 结构正确。
// 如果你的 pass 目前只处理 llvm.func，这里应该保持不变（my.for 还在）。

module {
  func.func @main() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %st = arith.constant 1 : index

    "my.for"(%lb, %ub, %st) ({
    ^bb0(%iv: index):
      // 做点简单计算，确保 iv 被用到
      %one = arith.constant 1 : index
      %tmp = arith.addi %iv, %one : index
      "my.yield"() : () -> ()
    }) : (index, index, index) -> ()

    return
  }
}

// CHECK-LABEL: func.func @main
// CHECK: "my.for"(
// CHECK-SAME: : (index, index, index) -> ()
// CHECK: ^bb0(%{{.*}}: index):
// CHECK: arith.addi
// CHECK: "my.yield"
// CHECK: return

// ----- case 2: ill-formed region (no iv) -----
// 用来验证你的 verifier/你的 pass 的结构检查是否能报错（可选）。
// 你如果不想要失败用例，可以删掉这段。
// -----
// RUN: not test-opt %s -test-lower-my-for 2>&1 | FileCheck %s --check-prefix=ERR

// ----- split marker -----
// -----
// ERR: my.for region must have induction var block argument
