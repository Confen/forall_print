#ifndef DIALECT_OP_
#define DIALECT_OP_

include "IR/dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"


def PrintOp : Test_Op<"print", [Pure]>
{
	let summary = "print";
	let arguments = (ins Index:$value);
	let results = (outs Index:$type);
}

#endif



