#ifndef SIMPLE_RANS_WRAPPER_H_
#define SIMPLE_RANS_WRAPPER_H_

#include <torch/extension.h>
#include <pybind11/numpy.h>
#include "rans_byte.h"


#define DEFAULT_ENC_BUF_SIZE (32 * 1024 * 1024)
#define PRECISION (16u)
#define PROB_SCALE (1u << 16u)
typedef py::array_t<uint16_t, py::array::c_style | py::array::forcecast> PyArrayUint16;
typedef py::array_t<bool, py::array::c_style | py::array::forcecast> PyArrayBool;


class RansEncoder
{
public:
    RansEncoder(
        size_t enc_buf_size = DEFAULT_ENC_BUF_SIZE);
    ~RansEncoder();

    int init_with_quantized_cdfs(
        const PyArrayUint16 &cdf_arr);
    uint64_t encode_with_precomp(
        const PyArrayUint16 &cdf_arr,
        const PyArrayUint16 &symbol_arr);
    uint64_t encode(
        const PyArrayUint16 &cdf_arr,
        const PyArrayUint16 &symbol_arr);
    uint64_t encode_bin(
        const PyArrayUint16 &cdf_arr,
        const PyArrayBool &symbol_arr);
    py::bytes flush();

private:
    RansState rans;
    uint8_t * enc_buf_ptr;
    size_t enc_buf_size;
    uint8_t * enc_ptr;
    std::vector<std::vector<RansEncSymbol>> esyms_list;
};


class RansDecoder
{
public:
    RansDecoder();

    int flush(
        const py::bytes &encoded);
    int init_with_quantized_cdfs(
        const PyArrayUint16 &cdf_arr);
    int decode_with_precomp(
        PyArrayUint16 &cdf_arr,
        PyArrayUint16 &symbol_array);
    int decode(
        PyArrayUint16 &cdf_arr,
        PyArrayUint16 &symbol_array);
    int decode_bin(
        PyArrayUint16 &cdf_arr,
        PyArrayBool &symbol_arr);

private:
    RansState rans;
    uint8_t * ptr;
    std::vector<std::vector<RansDecSymbol>> dsyms_list;
};

#endif  // SIMPLE_RANS_WRAPPER_H_
