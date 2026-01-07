; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @hello_forall() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 10
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  %5 = add i64 %2, 1
  br label %1

6:                                                ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
