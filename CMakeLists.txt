cmake_minimum_required(VERSION 3.20)
project(test)
set(CMAKE_CXX_STANDARD 20)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(LLVM_TARGET_DEFINITIONS IR/dialect.td)
mlir_tablegen(dialect.h.inc -gen-dialect-decls)
mlir_tablegen(dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(DialectIncGen)
set(LLVM_TARGET_DEFINITIONS IR/dialect_op.td)
mlir_tablegen(dialect_op.h.inc -gen-op-decls)
mlir_tablegen(dialect_op.cpp.inc -gen-op-defs)
add_public_tablegen_target(DialectOpIncGen)

add_custom_target(generate-inc ALL
	DEPENDS DialectIncGen DialectOpIncGen)

add_mlir_library(TestDialect
	lib/dialect.cpp
	PARTIAL_SOURCES_INTENDED
	DEPENDS DialectIncGen DialectOpIncGen
	LINK_LIBS PUBLIC MLIRIR)

add_llvm_executable(test-opt
	tools/test_opt.cpp
	PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(test-opt)
target_link_libraries(test-opt PRIVATE
	TestDialect
	MLIROptLib
	MLIRLLVMDialect
	MLIRIR
	MLIRFuncDialect
	MLIRSCFDialect
	MLIRParser
	MLIRPass
	MLIRSupport
	MLIRTransforms)
