#include "lib/Conversion/CGGIToJaxite/CGGIToJaxite.h"

#include <cstdint>
#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/Jaxite/IR/JaxiteOps.h"
#include "lib/Dialect/Jaxite/IR/JaxiteTypes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_CGGITOJAXITE
#include "lib/Conversion/CGGIToJaxite/CGGIToJaxite.h.inc"

class CGGIToJaxiteTypeConverter : public TypeConverter {
 public:
  CGGIToJaxiteTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    // TODO(gshruthi): Is this conversion needed for jaxite?
    // addConversion([ctx](lwe::LWECiphertextType type) -> Type {
    //   int width = widthFromEncodingAttr(type.getEncoding());
    //   return encrytpedUIntTypeFromWidth(ctx, width);
    // });
    addConversion([this](ShapedType type) -> Type {
      return type.cloneWith(type.getShape(),
                            this->convertType(type.getElementType()));
    });
  }
};

/// Returns true if the func's body contains any CGGI ops.
bool containsCGGIJaxiteOps(func::FuncOp func) {
  auto walkResult = func.walk([&](Operation *op) {
    if (llvm::isa<cggi::CGGIDialect>(op->getDialect()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

/// Returns the Value corresponding to a server key in the FuncOp containing
/// this op.
FailureOr<Value> getContextualJaxiteServerKeySet(Operation *op) {
  int num_inputs = op->getParentOfType<func::FuncOp>()
                       .getBody()
                       .getBlocks()
                       .front()
                       .getNumArguments();
  if (num_inputs < 2) {
    return op->emitOpError() << "Found less than 2 ops in function input. Did "
                                "the AddServerKeyArg pattern fail to run?";
  }
  auto block_args_list = op->getParentOfType<func::FuncOp>()
                             .getBody()
                             .getBlocks()
                             .front()
                             .getArguments();
  auto it = block_args_list.begin();
  while (!mlir::isa<jaxite::ServerKeySetType>(it->getType())) {
    it++;
  }
  if (it == block_args_list.end() ||
      !mlir::isa<jaxite::ServerKeySetType>(it->getType())) {
    return op->emitOpError() << "Cannot find Params argument. Did the "
                                "AddServerKeyArg pattern fail to run?";
  }
  return *it;
}

/// Returns the Value corresponding to params in the FuncOp containing
/// this op.
FailureOr<Value> getContextualJaxiteParams(Operation *op) {
  int num_inputs = op->getParentOfType<func::FuncOp>()
                       .getBody()
                       .getBlocks()
                       .front()
                       .getNumArguments();
  if (num_inputs < 2) {
    return op->emitOpError() << "Found less than 2 ops in function input. Did "
                                "the AddServerKeyArg pattern fail to run?";
  }
  auto block_args_list = op->getParentOfType<func::FuncOp>()
                             .getBody()
                             .getBlocks()
                             .front()
                             .getArguments();
  auto it = block_args_list.begin();
  while (!mlir::isa<jaxite::ParamsType>(it->getType())) {
    it++;
  }
  if (it == block_args_list.end() ||
      !mlir::isa<jaxite::ParamsType>(it->getType())) {
    return op->emitOpError() << "Cannot find Params argument. Did the "
                                "AddServerKeyArg pattern fail to run?";
  }
  return *it;
}

template <class Op>
struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

/// Convert a func by adding a server key argument. Converted ops in other
/// patterns need a server key SSA value available, so this pattern needs a
/// higher benefit.
struct AddJaxiteServerKeyArg : public OpConversionPattern<func::FuncOp> {
  AddJaxiteServerKeyArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 1000) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!containsCGGIJaxiteOps(op)) {
      return failure();
    }

    auto serverKeyType = jaxite::ServerKeySetType::get(getContext());
    auto paramsType = jaxite::ParamsType::get(getContext());
    FunctionType originalType = op.getFunctionType();
    llvm::SmallVector<Type, 4> newTypes;
    newTypes.reserve(originalType.getNumInputs() + 2);
    for (auto t : originalType.getInputs()) {
      newTypes.push_back(t);
    }
    newTypes.push_back(serverKeyType);
    newTypes.push_back(paramsType);
    auto newFuncType =
        FunctionType::get(getContext(), newTypes, originalType.getResults());
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(newFuncType);
      // In addition to updating the type signature, we need to update the
      // entry block's arguments to match the type signature
      Block &block = op.getBody().getBlocks().front();
      block.addArguments(serverKeyType, op.getLoc());
      block.addArguments(paramsType, op.getLoc());
    });
    return success();
  }
};

struct ConvertCGGIToJaxiteLut3Op : public OpConversionPattern<cggi::Lut3Op> {
  ConvertCGGIToJaxiteLut3Op(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::Lut3Op>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::Lut3Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    FailureOr<Value> result_server_key =
        getContextualJaxiteServerKeySet(op.getOperation());
    if (failed(result_server_key)) return result_server_key;
    Value serverKey = result_server_key.value();

    FailureOr<Value> result_params =
        getContextualJaxiteParams(op.getOperation());
    if (failed(result_params)) return result_params;
    Value params = result_params.value();

    int64_t truth_table_value = op.getLookupTableAttr().getUInt();
    int8_t truth_table_value_int8 = static_cast<int8_t>(truth_table_value);
    Value tt = b.create<arith::ConstantOp>(
        op.getLoc(), b.getIntegerAttr(b.getI8Type(), truth_table_value_int8));

    // The ciphertext parameters (a, b, c) are passed in reverse order from cggi
    // to jaxite to mirror jaxite API
    auto createLut3Op = rewriter.create<jaxite::Lut3Op>(
        op.getLoc(), op.getOutput().getType(), op.getC(), op.getB(), op.getA(),
        tt, serverKey, params);
    rewriter.replaceOp(op, createLut3Op);
    return success();
  }
};

struct ConvertCGGIToJaxiteTrivialEncryptOp
    : public OpConversionPattern<lwe::TrivialEncryptOp> {
  ConvertCGGIToJaxiteTrivialEncryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::TrivialEncryptOp>(context, /*benefit=*/2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::TrivialEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualJaxiteParams(op.getOperation());
    if (failed(result)) return result;
    Value serverParams = result.value();
    lwe::EncodeOp encodeOp = op.getInput().getDefiningOp<lwe::EncodeOp>();
    if (!encodeOp) {
      return op.emitError() << "Expected input to TrivialEncrypt to be the "
                               "result of an EncodeOp, but it was "
                            << op.getInput().getDefiningOp()->getName();
    }
    auto createConstantOp = rewriter.create<jaxite::ConstantOp>(
        op.getLoc(), op.getOutput().getType(), encodeOp.getPlaintext(),
        serverParams);
    rewriter.replaceOp(op, createConstantOp);
    return success();
  }
};

struct ConvertCGGIToJaxiteEncodeOp : public OpConversionPattern<lwe::EncodeOp> {
  ConvertCGGIToJaxiteEncodeOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::EncodeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::EncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class CGGIToJaxite : public impl::CGGIToJaxiteBase<CGGIToJaxite> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *op = getOperation();

    CGGIToJaxiteTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addLegalDialect<jaxite::JaxiteDialect>();
    target.addIllegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();
    // FuncOp is marked legal by the default structural conversion patterns
    // helper, just based on type conversion. We need more, but because the
    // addDynamicallyLegalOp is a set-based method, we can add this after
    // calling addStructuralConversionPatterns and it will overwrite the
    // legality condition set in that function.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      int num_inputs = op.getFunctionType().getNumInputs();
      if (num_inputs < 2) {
        return false;
      }
      bool hasServerKeyArg = op.getFunctionType().getNumInputs() > 0 &&
                             mlir::isa<jaxite::ServerKeySetType>(
                                 op.getFunctionType().getInput(num_inputs - 2));
      bool hasParamsArg = op.getFunctionType().getNumInputs() > 0 &&
                          mlir::isa<jaxite::ParamsType>(
                              op.getFunctionType().getInput(num_inputs - 1));
      bool result = typeConverter.isSignatureLegal(op.getFunctionType()) &&
                    typeConverter.isLegal(&op.getBody()) && hasServerKeyArg &&
                    hasParamsArg;
      return result;
    });

    target.addDynamicallyLegalOp<tensor::FromElementsOp, tensor::ExtractOp>(
        [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });

    // FIXME: still need to update callers to insert the new server key arg, if
    // needed and possible.
    patterns.add<AddJaxiteServerKeyArg, ConvertCGGIToJaxiteEncodeOp,
                 ConvertCGGIToJaxiteLut3Op, ConvertCGGIToJaxiteTrivialEncryptOp,
                 GenericOpPattern<tensor::FromElementsOp>,
                 GenericOpPattern<tensor::ExtractOp>>(typeConverter, context);
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
