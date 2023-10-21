#ifndef RANS_CODER_WRAPPER_H_
#define RANS_CODER_WRAPPER_H_

#include <torch/extension.h>
#include <pybind11/numpy.h>
#include "rans_byte.h"


#define DEFAULT_ENC_BUF_SIZE (8 * 1024 * 1024)
#define PRECISION (16u)
#define PROB_SCALE (1u << 16u)
typedef std::vector<uint32_t> VecUint32;
typedef std::vector<int32_t> VecInt32;
typedef std::vector<VecUint32> VecVecUint32;
typedef std::vector<VecInt32> VecVecInt32;
typedef py::array_t<int32_t, py::array::c_style | py::array::forcecast> PyArrayInt32;
typedef py::array_t<uint32_t, py::array::c_style | py::array::forcecast> PyArrayUint32;
typedef py::array_t<bool, py::array::c_style | py::array::forcecast> PyArrayBool;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> PyArrayDouble;


class IndexedRansCoder
{
public:
    IndexedRansCoder(
        bool overflow_coding,
        uint32_t batch_size,
        size_t enc_buf_size = DEFAULT_ENC_BUF_SIZE);
    ~IndexedRansCoder();

    int init_with_pmfs(
        PyArrayDouble &pmf_array, PyArrayInt32 &offset_array
    );
    int init_with_quantized_cdfs(
        VecVecUint32 &quantized_cdfs, PyArrayInt32 &offset_array
    );
    VecVecUint32 get_cdfs();
    PyArrayInt32 get_offset_array();

    template <bool OVERFLOW_CODING, bool WITH_INDEXES>
    std::vector<py::bytes> _encode_with_indexes(
        const PyArrayInt32 &symbol_array,
        const PyArrayInt32 &index_array
    );
    std::vector<py::bytes> encode(
        const PyArrayInt32 &symbol_array
    );
    std::vector<py::bytes> encode_with_indexes(
        const PyArrayInt32 &symbol_array,
        const PyArrayInt32 &index_array
    );

    template <bool OVERFLOW_CODING, bool WITH_INDEXES>
    int _decode_with_indexes(
        const std::vector<std::string> &encoded_list,
        const PyArrayInt32 &index_array,
        PyArrayInt32 &symbol_array
    );
    int decode(
        const std::vector<std::string> &encoded_list,
        PyArrayInt32 &symbol_array
    );
    int decode_with_indexes(
        const std::vector<std::string> &encoded_list,
        const PyArrayInt32 &index_array,
        PyArrayInt32 &symbol_array
    );

private:
    const bool overflow_coding;
    const uint32_t batch_size;
    std::vector<uint8_t *> enc_buf_ptrs;
    std::vector<size_t> enc_buf_sizes;
    std::vector<uint8_t *> enc_ptrs;
    
    VecVecUint32 cdfs;
    std::vector<std::vector<RansEncSymbol>> esyms_list;
    std::vector<std::vector<RansDecSymbol>> dsyms_list;
    PyArrayInt32 offset_array;
    std::vector<RansEncSymbol> bin_esyms;
    std::vector<RansDecSymbol> bin_dsyms;
};

class BinaryRansCoder
{
public:
    BinaryRansCoder(
        uint32_t batch_size,
        size_t enc_buf_size = DEFAULT_ENC_BUF_SIZE);
    ~BinaryRansCoder();

    std::vector<py::bytes> encode(
        const PyArrayBool &symbol_array,
        const PyArrayUint32 &prob_array
    );
    int decode(
        const std::vector<std::string> &encoded_list,
        const PyArrayUint32 &prob_array,
        PyArrayBool &symbol_array
    );

private:
    const uint32_t batch_size;
    std::vector<uint8_t *> enc_buf_ptrs;
    std::vector<size_t> enc_buf_sizes;
    std::vector<uint8_t *> enc_ptrs;
};

#endif  // RANS_CODER_WRAPPER_H_
