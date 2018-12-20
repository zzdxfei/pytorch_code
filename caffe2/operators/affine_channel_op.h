#ifndef CAFFE2_OPERATORS_AFFINE_CHANNEL_OP_H_
#define CAFFE2_OPERATORS_AFFINE_CHANNEL_OP_H_

#include <string>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// 前向传播操作
template <typename T, class Context>
class AffineChannelOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AffineChannelOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(bool, "is_learnable", is_learnable_, false) {
    // 存储类型必须已知
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  }

  // 执行前向传播
  bool RunOnDevice() override {
    // 不同的存储方式调用不同的函数
    return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                        : RunOnDeviceWithOrderNHWC();
  }

  // NCHW
  bool RunOnDeviceWithOrderNCHW() {
    // 获取输入张量
    const auto& X = Input(0);
    const auto& scale = Input(1);
    const auto& bias = Input(2);

    // 输出张量
    auto* Y = Output(0);

    // 该层学习，则不能In-place，Detectron中为In-place
    if (is_learnable_) {
      CAFFE_ENFORCE_NE(
          Y,
          &X,
          "In-place affine_channel_op is not supported when "
          "is_learnable = true.");
    }

    const int N = X.dim32(0);
    const int C = X.dim32(1);
    const int HxW = X.numel() / (N * C);

    // 输出和输入相同形状
    Y->ResizeLike(X);

    // 执行affine channel操作
    math::AffineChannel<T, Context, StorageOrder::NCHW>(
        N,
        C,
        HxW,
        X.template data<T>(),
        scale.template data<T>(),
        bias.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() {
    const auto& X = Input(0);
    const auto& scale = Input(1);
    const auto& bias = Input(2);
    auto* Y = Output(0);
    if (is_learnable_) {
      CAFFE_ENFORCE_NE(
          Y,
          &X,
          "In-place affine_channel_op is not supported when "
          "is_learnable = true.");
    }
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = X.dim32(ndim - 1);
    const int HxW = X.numel() / (N * C);
    Y->ResizeLike(X);
    math::AffineChannel<T, Context, StorageOrder::NHWC>(
        N,
        C,
        HxW,
        X.template data<T>(),
        scale.template data<T>(),
        bias.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

 private:
  const StorageOrder order_;
  const bool is_learnable_;
};

// 反向传播操作
template <typename T, class Context>
class AffineChannelGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AffineChannelGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(bool, "is_learnable", is_learnable_, false) {
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  }

  bool RunOnDevice() override {
    return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                        : RunOnDeviceWithOrderNHWC();
  }

  bool RunOnDeviceWithOrderNCHW();

  bool RunOnDeviceWithOrderNHWC();

 private:
  const StorageOrder order_;
  const bool is_learnable_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_AFFINE_CHANNEL_OP_H_
