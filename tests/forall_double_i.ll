; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@fmt_i64 = internal constant [5 x i8] c"%ld\0A\00", align 1

declare i32 @printf(ptr, ...)

define void @forall() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %6, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 10
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = call i32 (ptr, ...) @printf(ptr @fmt_i64, i64 %2)
  %6 = add i64 %2, 1
  br label %1

7:                                                ; preds = %10, %1
  %8 = phi i64 [ %12, %10 ], [ 0, %1 ]
  %9 = icmp slt i64 %8, 5
  br i1 %9, label %10, label %13

10:                                               ; preds = %7
  %11 = call i32 (ptr, ...) @printf(ptr @fmt_i64, i64 %8)
  %12 = add i64 %8, 1
  br label %7

13:                                               ; preds = %7
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
