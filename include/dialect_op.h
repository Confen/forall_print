#ifndef TEST_DIALECT_OP_H
#define TEST_DIALECT_OP_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "include/dialect.h"

#define GET_OP_CLASSES
#include "build/dialect_op.h.inc"

#endif
