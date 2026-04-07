#pragma once

#include <torch/extension.h>

namespace space_filling_curves {

template <int XIndexValue, int YIndexValue, int ZIndexValue>
struct AxisOrder3DIndices {
  static constexpr int XIndex = XIndexValue;
  static constexpr int YIndex = YIndexValue;
  static constexpr int ZIndex = ZIndexValue;
};

template <typename Dispatch>
inline void dispatch_axis_order_3d(const std::string& axis_order, Dispatch&& dispatch) {
  if (axis_order == "xyz") {
    dispatch(AxisOrder3DIndices<0, 1, 2>{});
  } else if (axis_order == "xzy") {
    dispatch(AxisOrder3DIndices<0, 2, 1>{});
  } else if (axis_order == "yxz") {
    dispatch(AxisOrder3DIndices<1, 0, 2>{});
  } else if (axis_order == "yzx") {
    dispatch(AxisOrder3DIndices<1, 2, 0>{});
  } else if (axis_order == "zxy") {
    dispatch(AxisOrder3DIndices<2, 0, 1>{});
  } else if (axis_order == "zyx") {
    dispatch(AxisOrder3DIndices<2, 1, 0>{});
  } else {
    TORCH_CHECK(
      false,
      "axis_order must be one of {'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'}");
  }
}

}  // namespace space_filling_curves
