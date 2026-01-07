/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: dialect.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace test {

/// a test my dialect.
class testDialect : public ::mlir::Dialect {
  explicit testDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~testDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("test");
  }
};
} // namespace test
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::test::testDialect)
