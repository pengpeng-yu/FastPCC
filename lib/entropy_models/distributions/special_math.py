# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# The "ndtr" function is derived from calculations made in:
# https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
# In the following email exchange, the author gives his consent to redistribute
# derived works under an Apache 2.0 license.
#
# From: Stephen Moshier <steve@moshier.net>
# Date: Sat, Jun 9, 2018 at 2:36 PM
# Subject: Re: Licensing cephes under Apache (BSD-like) license.
# To: rif <rif@google.com>
#
#
#
# Hello Rif,
#
# Yes, Google may distribute Cephes files under the Apache 2 license.
#
# If clarification is needed, I do not favor BSD over other free licenses.
# I would agree that Apache 2 seems to cover the concern you mentioned
# about sublicensees.
#
# Best wishes for good luck with your projects!
# Steve Moshier
#
#
#
# On Thu, 31 May 2018, rif wrote:
#
# > Hello Steve.
# > My name is Rif. I work on machine learning software at Google.
# >
# > Your cephes software continues to be incredibly useful and widely used. I
# > was wondering whether it would be permissible for us to use the Cephes code
# > under the Apache 2.0 license, which is extremely similar in permissions to
# > the BSD license (Wikipedia comparisons). This would be quite helpful to us
# > in terms of avoiding multiple licenses on software.
# >
# > I'm sorry to bother you with this (I can imagine you're sick of hearing
# > about this by now), but I want to be absolutely clear we're on the level and
# > not misusing your important software. In former conversation with Eugene
# > Brevdo (ebrevdo@google.com), you wrote "If your licensing is similar to BSD,
# > the formal way that has been handled is simply to add a statement to the
# > effect that you are incorporating the Cephes software by permission of the
# > author." I wanted to confirm that (a) we could use the Apache license, (b)
# > that we don't need to (and probably you don't want to) keep getting
# > contacted about individual uses, because your intent is generally to allow
# > this software to be reused under "BSD-like" license, and (c) you're OK
# > letting incorporators decide whether a license is sufficiently BSD-like?
# >
# > Best,
# >
# > rif
# >
# >
# >

"""Special Math Ops."""

import numpy as np
import torch

# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.) See `NdtrTest` in
# special_math_test.py.
LOGNDTR_FLOAT64_LOWER = -20.
LOGNDTR_FLOAT32_LOWER = -10.

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
LOGNDTR_FLOAT64_UPPER = 8.
LOGNDTR_FLOAT32_UPPER = 5.


def ndtr(x):
    """Normal distribution function.

    Returns the area under the Gaussian probability density function, integrated
    from minus infinity to x:

    ```
                      1       / x
       ndtr(x)  = ----------  |    exp(-0.5 t**2) dt
                  sqrt(2 pi)  /-inf

                = 0.5 (1 + erf(x / sqrt(2)))
                = 0.5 erfc(x / sqrt(2))
    ```

    Args:
      x: `Tensor` of type `float32`, `float64`.

    Returns:
      ndtr: `Tensor` with `dtype=x.dtype`.

    Raises:
      TypeError: if `x` is not floating-type.
    """

    if x.dtype not in [torch.float32, torch.float64]:
        raise TypeError(
            "x.dtype=%s is not handled, see docstring for supported types."
            % x.dtype)
    return _ndtr(x)


def _ndtr(x):
    """Implements ndtr core logic."""
    half_sqrt_2 = torch.tensor(0.5 * np.sqrt(2.), dtype=x.dtype, device=x.device)
    w = x * half_sqrt_2
    z = torch.abs(w)
    # noinspection PyTypeChecker
    y = torch.where(
        z < half_sqrt_2,
        1. + torch.erf(w),
        torch.where(w > 0., 2. - torch.erfc(z), torch.erfc(z)))
    return 0.5 * y


def log_ndtr(x, series_order=3):
    """Log Normal distribution function.

    For details of the Normal distribution function see `ndtr`.

    This function calculates `(log o ndtr)(x)` by either calling `log(ndtr(x))` or
    using an asymptotic series. Specifically:
    - For `x > upper_segment`, use the approximation `-ndtr(-x)` based on
      `log(1-x) ~= -x, x << 1`.
    - For `lower_segment < x <= upper_segment`, use the existing `ndtr` technique
      and take a log.
    - For `x <= lower_segment`, we use the series approximation of erf to compute
      the log CDF directly.

    The `lower_segment` is set based on the precision of the input:

    ```
    lower_segment = { -20,  x.dtype=float64
                    { -10,  x.dtype=float32
    upper_segment = {   8,  x.dtype=float64
                    {   5,  x.dtype=float32
    ```

    When `x < lower_segment`, the `ndtr` asymptotic series approximation is:

    ```
       ndtr(x) = scale * (1 + sum) + R_N
       scale   = exp(-0.5 x**2) / (-x sqrt(2 pi))
       sum     = Sum{(-1)^n (2n-1)!! / (x**2)^n, n=1:N}
       R_N     = O(exp(-0.5 x**2) (2N+1)!! / |x|^{2N+3})
    ```

    where `(2n-1)!! = (2n-1) (2n-3) (2n-5) ...  (3) (1)` is a
    [double-factorial](https://en.wikipedia.org/wiki/Double_factorial).


    Args:
      x: `Tensor` of type `float32`, `float64`.
      series_order: Positive Python `integer`. Maximum depth to
        evaluate the asymptotic expansion. This is the `N` above.

    Returns:
      log_ndtr: `Tensor` with `dtype=x.dtype`.

    Raises:
      TypeError: if `x.dtype` is not handled.
      TypeError: if `series_order` is a not Python `integer.`
      ValueError:  if `series_order` is not in `[0, 30]`.
    """
    if not isinstance(series_order, int):
        raise TypeError("series_order must be a Python integer.")
    if series_order < 0:
        raise ValueError("series_order must be non-negative.")
    if series_order > 30:
        raise ValueError("series_order must be <= 30.")

    if x.dtype == torch.float64:
        lower_segment = torch.tensor(
            LOGNDTR_FLOAT64_LOWER, dtype=torch.float64, device=x.device)
        upper_segment = torch.tensor(
            LOGNDTR_FLOAT64_UPPER, dtype=torch.float64, device=x.device)
    elif x.dtype == torch.float32:
        lower_segment = torch.tensor(
            LOGNDTR_FLOAT32_LOWER, dtype=torch.float32, device=x.device)
        upper_segment = torch.tensor(
            LOGNDTR_FLOAT32_UPPER, dtype=torch.float32, device=x.device)
    else:
        raise TypeError("x.dtype=%s is not supported." % x.dtype)

    # The basic idea here was ported from:
    #   https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
    # We copy the main idea, with a few changes
    # * For x >> 1, and X ~ Normal(0, 1),
    #     Log[P[X < x]] = Log[1 - P[X < -x]] approx -P[X < -x],
    #     which extends the range of validity of this function.
    # * We use one fixed series_order for all of 'x', rather than adaptive.
    # * Our docstring properly reflects that this is an asymptotic series, not a
    #   Taylor series. We also provided a correct bound on the remainder.
    # * We need to use the max/min in the _log_ndtr_lower arg to avoid nan when
    #   x=0. This happens even though the branch is unchosen because when x=0
    #   the gradient of a select involves the calculation 1*dy+0*(-inf)=nan
    #   regardless of whether dy is finite. Note that the minimum is a NOP if
    #   the branch is chosen.

    # noinspection PyTypeChecker
    return torch.where(
        x > upper_segment,
        -_ndtr(-x),  # log(1-x) ~= -x, x << 1
        torch.where(
            x > lower_segment,
            torch.log(_ndtr(torch.maximum(x, lower_segment))),
            _log_ndtr_lower(torch.minimum(x, lower_segment), series_order)))


def _log_ndtr_lower(x, series_order):
    """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
    x_2 = torch.square(x)
    # Log of the term multiplying (1 + sum)
    log_scale = (-0.5 * x_2 - torch.log(-x)
                 - torch.tensor(0.5 * np.log(2. * np.pi), dtype=x.dtype, device=x.device))
    return log_scale + torch.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
    """Calculates the asymptotic series used in log_ndtr."""
    if series_order <= 0:
        return torch.tensor(1, dtype=x.dtype, device=x.device)
    x_2 = torch.square(x)
    even_sum = torch.zeros_like(x)
    odd_sum = torch.zeros_like(x)
    x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.
    for n in range(1, series_order + 1):
        y = torch.tensor(
            _double_factorial(2 * n - 1), dtype=x.dtype, device=x.device
        ) / x_2n
        if n % 2:
            odd_sum += y
        else:
            even_sum += y
        x_2n = x_2n * x_2
    return 1. + even_sum - odd_sum


def _double_factorial(n):
    """The double factorial function for small Python integer `n`."""
    return torch.prod(torch.arange(n, 1, -2)).item()
