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

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::test::testDialect, mlir::LLVM::LLVMDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect>();
  mlir::test::registerTestPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "test-opt", registry));
}
