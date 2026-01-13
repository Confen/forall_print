#include "include/dialect.h"
#include "include/dialect_op.h"
#include "include/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

namespace {
struct StripTestPrintPass
    : PassWrapper<StripTestPrintPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripTestPrintPass)

  StringRef getArgument() const final { return "strip-test-print"; }
  StringRef getDescription() const final {
    return "Remove test.print ops before translation";
  }

  void runOnOperation() override {
    SmallVector<Operation *, 4> toErase;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "test.print")
        toErase.push_back(op);
    });
    for (Operation *op : toErase)
      op->erase();

    SmallVector<Operation *, 4> deadCasts;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() ==
              "builtin.unrealized_conversion_cast" &&
          op->use_empty()) {
        deadCasts.push_back(op);
      }
    });
    for (Operation *op : deadCasts)
      op->erase();
  }
};

struct LowerTestPrintPass
    : PassWrapper<LowerTestPrintPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTestPrintPass)

  StringRef getArgument() const final { return "lower-test-print"; }
  StringRef getDescription() const final {
    return "Lower test.print to llvm.call @printf";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder moduleBuilder(module.getBodyRegion());

    auto i8Type = IntegerType::get(context, 8);
    auto i8PtrType = LLVM::LLVMPointerType::get(context);
    auto i32Type = IntegerType::get(context, 32);
    auto i64Type = IntegerType::get(context, 64);

    std::string format = "%ld\n";
    format.push_back('\0');
    auto formatType = LLVM::LLVMArrayType::get(i8Type, format.size());

    auto formatName = StringRef("fmt_i64");
    auto formatGlobal = module.lookupSymbol<LLVM::GlobalOp>(formatName);
    if (!formatGlobal) {
      moduleBuilder.setInsertionPointToStart(module.getBody());
      formatGlobal = moduleBuilder.create<LLVM::GlobalOp>(
          module.getLoc(), formatType, /*isConstant=*/true,
          LLVM::Linkage::Internal, formatName,
          moduleBuilder.getStringAttr(format), /*alignment=*/1);
    }

    auto printfName = StringRef("printf");
    auto printfFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(printfName);
    if (!printfFunc) {
      moduleBuilder.setInsertionPointToStart(module.getBody());
      auto printfType =
          LLVM::LLVMFunctionType::get(i32Type, {i8PtrType}, /*isVarArg=*/true);
      printfFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          module.getLoc(), printfName, printfType);
    }

    SmallVector<Operation *, 4> toErase;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "test.print")
        return;

      Value value = op->getOperand(0);
      if (value.getType().isIndex()) {
        Operation *def = value.getDefiningOp();
        if (def && def->getName().getStringRef() ==
                       "builtin.unrealized_conversion_cast" &&
            def->getNumOperands() == 1) {
          value = def->getOperand(0);
        } else {
          op->emitError("expected index from unrealized_conversion_cast");
          signalPassFailure();
          return;
        }
      }

      if (!value.getType().isInteger(64)) {
        op->emitError("expected i64 value for test.print");
        signalPassFailure();
        return;
      }

      OpBuilder builder(op);
      auto c0 = builder.create<LLVM::ConstantOp>(
          op->getLoc(), i64Type, builder.getI64IntegerAttr(0));
      auto addr = builder.create<LLVM::AddressOfOp>(op->getLoc(), formatGlobal);
      auto formatPtr = builder.create<LLVM::GEPOp>(
          op->getLoc(), i8PtrType, formatType, addr, ArrayRef<Value>{c0, c0});
      builder.create<LLVM::CallOp>(op->getLoc(), printfFunc,
                                   ValueRange{formatPtr, value});
      toErase.push_back(op);
    });

    for (Operation *op : toErase)
      op->erase();

    SmallVector<Operation *, 4> deadCasts;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() ==
              "builtin.unrealized_conversion_cast" &&
          op->use_empty()) {
        deadCasts.push_back(op);
      }
    });
    for (Operation *op : deadCasts)
      op->erase();
  }
};

struct algo_for_pass : PassWrapper<algo_for_pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(algo_for_pass)

  StringRef getArgument() const final { return "test-lower-my-for"; }
  StringRef getDescription() const final {
    return "Lower my.for (scf-like) to LLVM CFG using block arguments";
  }

  // Helper: accept i64 directly; or index produced by unrealized_cast(i64)
  static FailureOr<Value> getAsI64(Value v, Operation *anchor) {
    MLIRContext *ctx = anchor->getContext();

    if (v.getType().isInteger(64))
      return v;

    if (v.getType().isIndex()) {
      if (Operation *def = v.getDefiningOp()) {
        if (def->getName().getStringRef() ==
                "builtin.unrealized_conversion_cast" &&
            def->getNumOperands() == 1 &&
            def->getOperand(0).getType().isInteger(64)) {
          return def->getOperand(0);
        }
      }
    }
    return failure();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    auto i64Type = IntegerType::get(context, 64);

    // 只在 llvm.func 里做 CFG 改写（因为我们会插 llvm.br/llvm.cond_br）
    SmallVector<LLVM::LLVMFuncOp, 8> funcs;
    module.walk([&](LLVM::LLVMFuncOp f) { funcs.push_back(f); });

    for (LLVM::LLVMFuncOp func : funcs) {
      SmallVector<Operation *, 8> loops;
      func.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "my.for")
          loops.push_back(op);
      });

      for (Operation *op : loops) {
        Location loc = op->getLoc();

        // ---- (1) 校验 my.for 结构：scf.for 风格 ----
        if (op->getNumRegions() != 1) {
          op->emitError("my.for must have exactly 1 region");
          signalPassFailure();
          continue;
        }
        if (op->getNumOperands() != 3) {
          op->emitError("my.for expects 3 operands: lb, ub, step");
          signalPassFailure();
          continue;
        }

        Region &loopRegion = op->getRegion(0);
        if (!llvm::hasSingleElement(loopRegion)) {
          op->emitError("my.for region must have a single block");
          signalPassFailure();
          continue;
        }
        Block &loopBlock = loopRegion.front();
        if (loopBlock.getNumArguments() < 1) {
          op->emitError("my.for region must have induction var block argument");
          signalPassFailure();
          continue;
        }

        // operands: lb, ub, step
        Value lb = op->getOperand(0);
        Value ub = op->getOperand(1);
        Value step = op->getOperand(2);

        auto lbI64 = getAsI64(lb, op);
        auto ubI64 = getAsI64(ub, op);
        auto stepI64 = getAsI64(step, op);
        if (failed(lbI64) || failed(ubI64) || failed(stepI64)) {
          op->emitError("expected lb/ub/step to be i64, or index from unrealized_cast(i64)");
          signalPassFailure();
          continue;
        }

        // ---- (2) CFG 重写：split 当前 block ----
        Block *preheader = op->getBlock();
        Region *parentRegion = preheader->getParent();

        // 把 my.for 及其后的 op 切到 exitBlock
        Block *exitBlock = preheader->splitBlock(Block::iterator(op));

        // 在 exitBlock 之前插入 cond/body blocks
        Block *condBlock = new Block();
        Block *bodyBlock = new Block();
        parentRegion->getBlocks().insert(exitBlock->getIterator(), condBlock);
        parentRegion->getBlocks().insert(exitBlock->getIterator(), bodyBlock);

        OpBuilder builder(context);

        // 用 condBlock 的 block argument 表示 “phi iv”
        condBlock->addArgument(i64Type, loc);
        Value iv = condBlock->getArgument(0);

        // ---- (3) preheader: br (lb) -> cond ----
        builder.setInsertionPointToEnd(preheader);
        builder.create<LLVM::BrOp>(loc, ValueRange(*lbI64), condBlock);
        // 注意：你的 MLIR-21 里 BrOp build 签名是 (ValueRange operands, Block* dest)

        // ---- (4) cond: cmp + cond_br -> body/exit ----
        builder.setInsertionPointToStart(condBlock);
        auto cmp = builder.create<LLVM::ICmpOp>(
          //slt: signed less-than
            loc, LLVM::ICmpPredicate::slt, iv, *ubI64);

        // 你这套版本通常存在无参数的 cond_br builder
        builder.create<LLVM::CondBrOp>(loc, cmp, bodyBlock, exitBlock);

        // ---- (5) body: 搬运 region body，替换 iv，末尾回边 ----
        builder.setInsertionPointToStart(bodyBlock);

        // 替换 induction var
        Value ivArg = loopBlock.getArgument(0);
        if (ivArg.getType().isInteger(64)) {
          ivArg.replaceAllUsesWith(iv);
        } else if (ivArg.getType().isIndex()) {
          // 如果你坚持 my.for 的 iv 是 index：插 i64->index 的 unrealized cast
          auto cast = builder.create<UnrealizedConversionCastOp>(
              loc, TypeRange{IndexType::get(context)}, ValueRange(iv));
          ivArg.replaceAllUsesWith(cast.getResult(0));
        } else {
          op->emitError("unsupported induction var type; expected i64 or index");
          signalPassFailure();
          continue;
        }

        // terminator 必须是 my.yield
        Operation *term = loopBlock.getTerminator();
        if (!term || term->getName().getStringRef() != "my.yield") {
          op->emitError("expected my.yield terminator in my.for region");
          signalPassFailure();
          continue;
        }

        // 把 terminator 之前的 ops splice 到 bodyBlock
        auto &srcOps = loopBlock.getOperations();
        auto termIt = Block::iterator(term);
        bodyBlock->getOperations().splice(bodyBlock->end(), srcOps,
                                          srcOps.begin(), termIt);

        // 删除 my.yield
        term->erase();

        // body 末尾：next = iv + step; br(next) -> cond
        builder.setInsertionPointToEnd(bodyBlock);
        auto next = builder.create<LLVM::AddOp>(loc, iv, *stepI64);
        builder.create<LLVM::BrOp>(loc, ValueRange(next.getResult()), condBlock);

        // ---- (6) 删除原 my.for op（它现在在 exitBlock 开头）----
        op->erase();

        // ---- (7) 清理死的 unrealized casts（可选） ----
        SmallVector<Operation *, 8> deadCasts;
        func.walk([&](Operation *cand) {
          if (cand->getName().getStringRef() ==
                  "builtin.unrealized_conversion_cast" &&
              cand->use_empty())
            deadCasts.push_back(cand);
        });
        for (Operation *c : deadCasts)
          c->erase();
      }
    }
  }
};


struct unique_mul_pass : PassWrapper<unique_mul_pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(unique_mul_pass)

  StringRef getArgument() const final { return "lower-test-print-llvm-mul"; }
  StringRef getDescription() const final {
    return "Lower test.print to llvm.call @test_print_i64 and llvm.mul";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder moduleBuilder(module.getBodyRegion());

    auto i64Type = IntegerType::get(context, 64);
    auto voidType = LLVM::LLVMVoidType::get(context);

    auto printName = StringRef("test_print_i64");
    auto printFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(printName);
    if (!printFunc) {
      moduleBuilder.setInsertionPointToStart(module.getBody());
      auto printType =
          LLVM::LLVMFunctionType::get(voidType, {i64Type}, /*isVarArg=*/false);
      printFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          module.getLoc(), printName, printType);
    }

    SmallVector<Operation *, 4> toErase;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "test.print")
        return;

      Value value = op->getOperand(0);
      if (value.getType().isIndex()) {
        Operation *def = value.getDefiningOp();
        if (def && def->getName().getStringRef() ==
                       "builtin.unrealized_conversion_cast" &&
            def->getNumOperands() == 1) {
          value = def->getOperand(0);
        } else {
          op->emitError("expected index from unrealized_conversion_cast");
          signalPassFailure();
          return;
        }
      }

      if (!value.getType().isInteger(64)) {
        op->emitError("expected i64 value for test.print");
        signalPassFailure();
        return;
      }

      OpBuilder builder(op);
      auto c2 = builder.create<LLVM::ConstantOp>(
          op->getLoc(), i64Type, builder.getI64IntegerAttr(2));
      auto muled = builder.create<LLVM::MulOp>(op->getLoc(), value, c2);
      builder.create<LLVM::CallOp>(op->getLoc(), printFunc,
                                   ValueRange{muled});
      toErase.push_back(op);
    });

    for (Operation *op : toErase)
      op->erase();

    SmallVector<Operation *, 4> deadCasts;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() ==
              "builtin.unrealized_conversion_cast" &&
          op->use_empty()) {
        deadCasts.push_back(op);
      }
    });
    for (Operation *op : deadCasts)
      op->erase();
  }
};


struct unique_add_pass : PassWrapper<unique_add_pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(unique_add_pass)

  StringRef getArgument() const final { return "lower-test-print-llvm-add"; }
  StringRef getDescription() const final {
    return "Lower test.print to llvm.call @test_print_i64 and llvm.add";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder moduleBuilder(module.getBodyRegion());

    auto i64Type = IntegerType::get(context, 64);
    auto voidType = LLVM::LLVMVoidType::get(context);

    auto printName = StringRef("test_print_i64");
    auto printFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(printName);
    if (!printFunc) {
      moduleBuilder.setInsertionPointToStart(module.getBody());
      auto printType =
          LLVM::LLVMFunctionType::get(voidType, {i64Type}, /*isVarArg=*/false);
      printFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          module.getLoc(), printName, printType);
    }

    SmallVector<Operation *, 4> toErase;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "test.print")
        return;

      Value value = op->getOperand(0);
      if (value.getType().isIndex()) {
        Operation *def = value.getDefiningOp();
        if (def && def->getName().getStringRef() ==
                       "builtin.unrealized_conversion_cast" &&
            def->getNumOperands() == 1) {
          value = def->getOperand(0);
        } else {
          op->emitError("expected index from unrealized_conversion_cast");
          signalPassFailure();
          return;
        }
      }

      if (!value.getType().isInteger(64)) {
        op->emitError("expected i64 value for test.print");
        signalPassFailure();
        return;
      }

      OpBuilder builder(op);
      auto c1 = builder.create<LLVM::ConstantOp>(
          op->getLoc(), i64Type, builder.getI64IntegerAttr(1));
      auto added = builder.create<LLVM::AddOp>(op->getLoc(), value, c1);
      builder.create<LLVM::CallOp>(op->getLoc(), printFunc,
                                   ValueRange{added});
      toErase.push_back(op);
    });

    for (Operation *op : toErase)
      op->erase();

    SmallVector<Operation *, 4> deadCasts;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() ==
              "builtin.unrealized_conversion_cast" &&
          op->use_empty()) {
        deadCasts.push_back(op);
      }
    });
    for (Operation *op : deadCasts)
      op->erase();
  }
};
} // namespace

namespace mlir {
namespace test {

void registerTestPasses() {
  PassRegistration<StripTestPrintPass>();
  PassRegistration<LowerTestPrintPass>();
  PassRegistration<unique_add_pass>();
  PassRegistration<unique_mul_pass>();
  PassRegistration<algo_for_pass>();
}

} // namespace test
} // namespace mlir
