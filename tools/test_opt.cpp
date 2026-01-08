#include "include/dialect.h"
#include "include/dialect_op.h"

#include "llvm/ADT/SmallVector.h"
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

static void registerTestPasses() {
  PassRegistration<StripTestPrintPass>();
  PassRegistration<LowerTestPrintPass>();
  PassRegistration<unique_add_pass>();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::test::testDialect, mlir::LLVM::LLVMDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect>();
  registerTestPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "test-opt", registry));
}
