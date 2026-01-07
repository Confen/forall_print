#include "include/dialect.h"
#include "include/dialect_op.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::test;

void testDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "build/dialect_op.cpp.inc"
      >();
}

#define GET_DIALECT_DEFINITION
#include "build/dialect.cpp.inc"

#define GET_OP_CLASSES
#include "build/dialect_op.cpp.inc"
