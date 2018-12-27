// conv_op_impl.h is the templated implementation of the conv_op.h file.
#ifndef CAFFE2_OPERATORS_DEFORM_CONV_OP_IMPL_H_
#define CAFFE2_OPERATORS_DEFORM_CONV_OP_IMPL_H_

#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/deform_conv_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// 前向传播代码
template <typename T, class Context>
bool DeformConvOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const Tensor& X = Input(INPUT);
  const Tensor& offset = Input(OFFSET);
  auto& filter = Input(FILTER);
  Tensor* Y = Output(0);

  const int N = X.dim32(0), C = X.dim32(1);
  CAFFE_ENFORCE_EQ(X.dim(), filter.ndim());

  // output的channels
  const int M = filter.dim32(0);
  // 输入的通道数 = filter.dim32(1) * 组数
  CAFFE_ENFORCE(
      C == filter.dim32(1) * group_,
      "Convolution op: input channels does not match: # of input channels ",
      C,
      " is not equal to kernel channels * group:",
      filter.dim32(1),
      "*",
      group_);
  // 输出的通道数能被group整除
  CAFFE_ENFORCE(
      M % group_ == 0,
      "The number of output channels is not divisible by group.");
  // 仅支持kernel为(kh, kw)的形式
  CAFFE_ENFORCE(
      kernel_.size() == 2,
      "Deformable convolution only supports 2d kernel, has ",
      kernel_.size(),
      "d kernel.");
  // offset的轴必须为4维
  CAFFE_ENFORCE(
      offset.dim() == 4,
      "Deformable convolution only supports 4d offset, has ",
      offset.dim(),
      "d offset.");
  // offset的第0维为batch size
  CAFFE_ENFORCE_EQ(offset.dim32(0), N);
  // 输入的通道数必须被可变卷积组数整除
  CAFFE_ENFORCE(
      C % deformable_group_ == 0,
      "The number of input channels ",
      C,
      " is not divisible by deformable group ",
      deformable_group_);
  // 输出的通道数必须被可变卷积组数整除
  CAFFE_ENFORCE(
      M % deformable_group_ == 0,
      "The number of output channels ",
      M,
      " is not divisible by deformable group ",
      deformable_group_);
  // offset的通道数必须如下
  CAFFE_ENFORCE(
      offset.dim32(1) == 2 * kernel_h() * kernel_w() * deformable_group_,
      "Deformable convolution: offset 1st dimension must equal "
      "2 * kernel_h * kernel_w * deformable_group: 2 * ",
      kernel_h(),
      " * ",
      kernel_w(),
      " * ",
      deformable_group_);

  // offset的高和宽必须和输出一致
  CAFFE_ENFORCE_EQ(
      offset.dim32(2),
      (X.dim32(2) + pad_t() + pad_b() - (dilation_h() * (kernel_h() - 1) + 1)) /
              stride_h() +
          1);
  CAFFE_ENFORCE_EQ(
      offset.dim32(3),
      (X.dim32(3) + pad_l() + pad_r() - (dilation_w() * (kernel_w() - 1) + 1)) /
              stride_w() +
          1);

  // kernel_h * kernel_w
  int kernel_dims_size = 1;
  for (int i = 0; i < kernel_.size(); ++i) {
    CAFFE_ENFORCE(filter.dim32(i + 2) == kernel_[i]);
    kernel_dims_size *= kernel_[i];
  }

  // 计算Y的形状
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, filter.dim32(0));

  // (h, w)
  const vector<int> input_dims = GetDims(X);
  // (h, w)
  const vector<int> output_dims = GetDims(*Y);
  // 返回图片的尺寸 h x w
  const int input_image_size = this->GetDimsSize(X);
  const int output_image_size = this->GetDimsSize(*Y);

  vector<int> img_shape;
  img_shape.assign(X.sizes().begin() + 1, X.sizes().end());

  vector<int> buffer_shape;
  // (C * kernel_h * kernel_w, output_h, output_w)
  buffer_shape.push_back(C / group_ * kernel_dims_size);
  buffer_shape.insert(
      buffer_shape.end(), output_dims.begin(), output_dims.end());

  // The dimension of each kernel
  const int kernel_dim = C / group_ * kernel_dims_size;

  // The offset corresponding to a single input image, and a single output
  // image.

  // offset为batch间的间隔
  const int input_offset = C / group_ * input_image_size;
  const int output_offset = M / group_ * output_image_size;
  const int offset_offset = offset.numel() / offset.dim32(0);
  const int filter_offset = filter.size() / group_;

  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
  const T* Xdata = X.template data<T>();
  const T* offset_data = offset.template data<T>();

  if (InputSize() == 4) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(bias.ndim() == 1);
    CAFFE_ENFORCE(bias.dim32(0) == M);
    if (bias_multiplier_.size() != output_image_size) {
      // If the helper bias multiplier is not image size, reshape and fill it
      // with
      // one.
      // 方便后面的计算
      bias_multiplier_.Resize(vector<int64_t>(1, output_image_size));
      math::Set<T, Context>(
          output_image_size,
          static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
  }


  T* Ydata = Y->template mutable_data<T>();
  const T* bias_data = nullptr;
  if (InputSize() == 4) {
    bias_data = Input(BIAS).template data<T>();
  }

  auto f = [&](Tensor* col_buffer) {
    col_buffer->Resize(buffer_shape);
    T* col_buffer_data = col_buffer->template mutable_data<T>();

    // Im2col, followed by gemm.
    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        // 使用offset_data对X进行矩阵化
        DeformableIm2col(
            Xdata + group_id * input_offset,
            offset_data,
            X.sizes(),
            col_buffer->sizes(),
            col_buffer_data);
        // Weight term
        // 矩阵相乘 filter * X
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            M / group_,
            output_image_size,
            kernel_dim,
            1,
            filter.template data<T>() + group_id * filter_offset,
            col_buffer_data,
            0,
            Ydata + group_id * output_offset,
            &context_);
      }

      // 添加bias
      if (bias_data) {
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            output_image_size,
            1,
            1,
            bias_data,
            bias_multiplier_.template data<T>(),
            1,
            Ydata,
            &context_);
      }
      Xdata += input_offset * group_;
      Ydata += output_offset * group_;
      offset_data += offset_offset;
    }
  };

  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, f);
  } else {
    f(&col_buffer_);
  }
  return true;
}

template <typename T, class Context>
bool DeformConvGradientOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  // 已知项
  auto& X = Input(INPUT);
  auto& offset = Input(OFFSET);
  auto& filter = Input(FILTER);

  auto& dY = Input(OUTPUT_GRAD);

  // 求这两项
  auto* dfilter = Output(FILTER_GRAD);
  auto* doffset = Output(OFFSET_GRAD);

  const int N = X.dim32(0), C = X.dim32(1);

  // (input_h, input_w)
  const vector<int> input_dims = this->GetDims(X);
  // (input_h x input_w)
  const int input_image_size = this->GetDimsSize(X);

  // (output_h, output_w)
  const vector<int> output_dims = this->GetDims(dY);
  // The output image size is the spatial size of the output.
  // (output_h x output_w)
  const int output_image_size = this->GetDimsSize(dY);

  ConvPoolOpBase<Context>::ComputePads(input_dims);
  CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());

  // 输出通道个数
  const int M = filter.dim32(0);

  // 检查输入是否合法
  CAFFE_ENFORCE(filter.dim32(1) * group_ == C);

  CAFFE_ENFORCE(
      kernel_.size() == 2,
      "Deformable convolution only supports 2d kernel, has ",
      kernel_.size(),
      "d kernel.");
  CAFFE_ENFORCE(
      offset.ndim() == 4,
      "Deformable convolution only supports 4d offset, has ",
      offset.ndim(),
      "d offset.");
  CAFFE_ENFORCE_EQ(offset.dim32(0), N);
  CAFFE_ENFORCE(
      C % deformable_group_ == 0,
      "The number of input channels ",
      C,
      " is not divisible by deformable group ",
      deformable_group_);
  CAFFE_ENFORCE(
      M % deformable_group_ == 0,
      "The number of output channels ",
      M,
      " is not divisible by deformable group ",
      deformable_group_);
  CAFFE_ENFORCE(
      offset.dim32(1) == 2 * kernel_h() * kernel_w() * deformable_group_,
      "Deformable convolution: offset 1st dimension must equal "
      "2 * kernel_h * kernel_w * deformable_group: 2 * ",
      kernel_h(),
      " * ",
      kernel_w(),
      " * ",
      deformable_group_);

  CAFFE_ENFORCE_EQ(
      offset.dim32(2),
      (X.dim32(2) + pad_t() + pad_b() - (dilation_h() * (kernel_h() - 1) + 1)) /
              stride_h() +
          1);
  CAFFE_ENFORCE_EQ(
      offset.dim32(3),
      (X.dim32(3) + pad_l() + pad_r() - (dilation_w() * (kernel_w() - 1) + 1)) /
              stride_w() +
          1);

  // 3 x 3
  int kernel_dims_size = 1;
  for (int i = 0; i < kernel_.size(); ++i) {
    CAFFE_ENFORCE(filter.dim32(i + 2) == kernel_[i]);
    kernel_dims_size *= kernel_[i];
  }

  CAFFE_ENFORCE(M % group_ == 0);
  // 导数张量分配空间
  dfilter->ResizeLike(filter);
  doffset->ResizeLike(offset);

  // The dimension of each kernel
  const int kernel_dim = C / group_ * kernel_dims_size;

  // The offset corresponding to a single input image, and a single output
  // image.
  // batch中每个图像的长度
  const int input_offset = C / group_ * input_image_size;
  const int output_offset = M / group_ * output_image_size;
  const int offset_offset = offset.size() / offset.dim32(0);
  const int filter_offset = filter.size() / group_;

  // The col buffer is stored in CHW order as well - kernel_dim, and the
  // height and width.
  // (C, input_h, input_w)
  vector<int64_t> img_shape;
  img_shape.assign(X.dims().begin() + 1, X.dims().end());

  // (C x kernel_h x kernel_w, output_h, output_w)
  vector<int64_t> col_buffer_shape;
  col_buffer_shape.push_back(C * kernel_dims_size);
  col_buffer_shape.insert(
      col_buffer_shape.end(), output_dims.begin(), output_dims.end());
  col_buffer_.Resize(col_buffer_shape);

  const int col_buffer_offset = col_buffer_.size() / group_;

  const T* Xdata = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* offset_data = offset.template data<T>();
  const T* dYdata = dY.template data<T>();

  T* col_buffer_data = col_buffer_.template mutable_data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();
  T* doffset_data = doffset->template mutable_data<T>();

  // Pre-setting the gradients to zero.
  math::Set<T, Context>(dfilter->size(), 0, dfilter_data, &context_);

  T* dbias_data = nullptr;
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD);
    dbias->Resize(M);
    if (bias_multiplier_.size() != output_image_size) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(vector<int64_t>(1, output_image_size));
      math::Set<T, Context>(
          output_image_size,
          static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    dbias_data = dbias->template mutable_data<T>();
    math::Set<T, Context>(dbias->size(), 0, dbias_data, &context_);
  }

  // dX分配空间
  T* dXdata = nullptr;
  if (OutputSize() == 4 || (no_bias_ && (OutputSize() == 3))) {
    auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
    dX->ResizeLike(X);
    dXdata = dX->template mutable_data<T>();
    math::Set<T, Context>(dX->size(), 0, dXdata, &context_);
  }

  for (int image_id = 0; image_id < N; ++image_id) {
    // group_id = 0
    for (int group_id = 0; group_id < group_; ++group_id) {
      // 获得转变矩阵的导数
      math::Gemm<T, Context>(
          CblasTrans,
          CblasNoTrans,
          kernel_dim,
          output_image_size,
          M / group_,
          1,
          filter_data + group_id * filter_offset,
          dYdata + group_id * output_offset,
          0,
          col_buffer_data + group_id * col_buffer_offset,
          &context_);
    }

    // Gradient with respect to offsets
    // 求offset的导数
    DeformableCol2imCoord(
        col_buffer_data,
        Xdata,
        offset_data,
        X.dims(),
        col_buffer_shape,
        doffset_data);

    // Gradient with respect to input data
    // 求input的导数
    if (dXdata) {
      DeformableCol2im(
          col_buffer_data, offset_data, X.dims(), col_buffer_shape, dXdata);
      dXdata += input_offset * group_;
    }

    // Gradient with respect to filter
    // 求filter的导数
    // 计算转变矩阵
    DeformableIm2col(
        Xdata, offset_data, X.dims(), col_buffer_shape, col_buffer_data);
    // 求filter的导数
    for (int group_id = 0; group_id < group_; ++group_id) {
      math::Gemm<T, Context>(
          CblasNoTrans,
          CblasTrans,
          M / group_,
          kernel_dim,
          output_image_size,
          1,
          dYdata + group_id * output_offset,
          col_buffer_data + group_id * col_buffer_offset,
          1,
          dfilter_data + group_id * filter_offset,
          &context_);
    }

    // Gradient with respect to bias
    // 计算bias的导数
    if (dbias_data) {
      math::Gemv<T, Context>(
          CblasNoTrans,
          M,
          output_image_size,
          1,
          dYdata,
          bias_multiplier_.template data<T>(),
          1,
          dbias_data,
          &context_);
    }

    Xdata += input_offset * group_;
    dYdata += output_offset * group_;
    offset_data += offset_offset;
    doffset_data += offset_offset;
  }

  return true;
}
} // namespace caffe2

#endif // CAFFE2_OPERATORS_DEFORM_CONV_OP_IMPL_H_
