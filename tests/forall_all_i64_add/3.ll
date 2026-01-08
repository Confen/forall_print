; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @test_print_i64(i64)

define void @forall() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %6, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 10
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = add i64 %2, 1
  call void @test_print_i64(i64 %5)
  %6 = add i64 %2, 1
  br label %1

7:                                                ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
