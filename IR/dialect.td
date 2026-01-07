#ifndef TEST_DIALECT_
#define TEST_DIALECT_

include "mlir/IR/OpBase.td"

def test_Dialect : Dialect {
	let name = "test";
	let cppNamespace = "::mlir::test";
	let summary = [{
	a test my dialect.	
}];

}

class Test_Op<string mnemonic, list<Trait> traits = []> :
	Op<test_Dialect, mnemonic, traits>;

#endif
