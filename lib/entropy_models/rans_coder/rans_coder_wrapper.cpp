#include <torch/extension.h>
#include "rans_coder_wrapper.h"
#include "cdf_ops.cpp"


IndexedRansCoder::IndexedRansCoder(
    bool overflow_coding, uint32_t batch_size, size_t enc_buf_size
) : overflow_coding(overflow_coding), batch_size(batch_size)
{
    assert(batch_size > 0);
    bin_esyms.resize(2);
    bin_dsyms.resize(2);
    RansEncSymbolInit(&bin_esyms[0], 0, 1, 1);
    RansDecSymbolInit(&bin_dsyms[0], 0, 1);
    RansEncSymbolInit(&bin_esyms[1], 1, 1, 1);
    RansDecSymbolInit(&bin_dsyms[1], 1, 1);
    assert(enc_buf_size > 0);
    enc_buf_ptrs.resize(batch_size);
    enc_buf_sizes.resize(batch_size);
    enc_ptrs.resize(batch_size);
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
        enc_buf_ptrs[idx] = new uint8_t[enc_buf_size];
        enc_buf_sizes[idx] = enc_buf_size;
    }
}

IndexedRansCoder::~IndexedRansCoder()
{
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
        delete[] enc_buf_ptrs[idx];
    }
}

int IndexedRansCoder::init_with_pmfs(
    PyArrayDouble &pmf_array,
    PyArrayInt32 &offset_array)
{
    VecVecUint32 &&cdfs = batched_pmf_to_quantized_cdf(pmf_array, offset_array, overflow_coding);
    return init_with_quantized_cdfs(cdfs, offset_array);
}

int IndexedRansCoder::init_with_quantized_cdfs(
    VecVecUint32 &cdfs,
    PyArrayInt32 &offset_array)
{
    size_t cdfs_num = cdfs.size();
    esyms_list.resize(cdfs_num);
    dsyms_list.resize(cdfs_num);

    for (size_t cdf_idx = 0; cdf_idx < cdfs.size(); ++cdf_idx)
    {
        const VecUint32 &cdf = cdfs[cdf_idx];
        std::vector<RansEncSymbol> &esyms = esyms_list[cdf_idx];
        std::vector<RansDecSymbol> &dsyms = dsyms_list[cdf_idx];

        assert(cdf[0] == 0 && cdf.back() == PROB_SCALE);
        assert((cdf.size() - 1) < (1u << PRECISION));
        const uint16_t symbols_num = uint16_t(cdf.size() - 1);

        esyms.resize(symbols_num);
        dsyms.resize(symbols_num);
        for (uint16_t sym = 0; sym < symbols_num; ++sym)
        {
            RansEncSymbolInit(&esyms[sym], cdf[sym], cdf[sym + 1] - cdf[sym], PRECISION);
            RansDecSymbolInit(&dsyms[sym], cdf[sym], cdf[sym + 1] - cdf[sym]);
        }
    }

    this->offset_array = std::move(offset_array);
    this->cdfs = std::move(cdfs);
    return 0;
}

VecVecUint32 IndexedRansCoder::get_cdfs()
{
    return this->cdfs;
}

PyArrayInt32 IndexedRansCoder::get_offset_array()
{
    return this->offset_array;
}

template <bool OVERFLOW_CODING, bool WITH_INDEXES>
std::vector<py::bytes> IndexedRansCoder::_encode_with_indexes(
    const PyArrayInt32 &symbol_array,
    const PyArrayInt32 &index_array)
{
    const py::buffer_info symbol_arr_buf = symbol_array.request();
    const py::buffer_info index_arr_buf = index_array.request();
    const int32_t * const offset_ptr = reinterpret_cast<int32_t *>(offset_array.request().ptr);
    const size_t symbols_num = symbol_arr_buf.shape[1];
    const size_t preserved_size = 4 * symbols_num;
    assert(symbol_arr_buf.ndim == 2);
    assert(batch_size == symbol_arr_buf.shape[0]);
    if (WITH_INDEXES)
    {
        assert(index_arr_buf.ndim == 2);
        assert(batch_size == index_arr_buf.shape[0]);
        assert(symbol_arr_buf.shape == index_arr_buf.shape);
    }
    std::vector<py::bytes> encoded_list;
    encoded_list.reserve(batch_size);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for collapse(1)
    for (size_t unit_idx = 0; unit_idx < batch_size; ++unit_idx)
    {
        const int32_t * const symbol_ptr = reinterpret_cast<int32_t *>(
            static_cast<uint8_t *>(symbol_arr_buf.ptr) + unit_idx * symbol_arr_buf.strides[0]);
        const int32_t * const index_ptr = WITH_INDEXES ? reinterpret_cast<int32_t *>(
            static_cast<uint8_t *>(index_arr_buf.ptr) + unit_idx * index_arr_buf.strides[0]) : nullptr;
        if (preserved_size > enc_buf_sizes[unit_idx])
        {
            delete[] enc_buf_ptrs[unit_idx];
            enc_buf_ptrs[unit_idx] = new uint8_t[preserved_size];
            enc_buf_sizes[unit_idx] = preserved_size;
        }
        uint8_t* const &enc_buf_ptr = enc_buf_ptrs[unit_idx];
        const size_t &enc_buf_size = enc_buf_sizes[unit_idx];

        uint8_t *ptr = enc_buf_ptr + enc_buf_size;
        RansState rans;
        RansEncInit(&rans);

        for (size_t forward_sym_idx = 0; forward_sym_idx < symbols_num; ++forward_sym_idx)
        {
            const size_t sym_idx = symbols_num - 1 - forward_sym_idx;
            const size_t &cdf_idx = WITH_INDEXES ? index_ptr[sym_idx] : sym_idx % cdfs.size();
            const std::vector<RansEncSymbol> &esyms = esyms_list[cdf_idx];
            const int32_t &offset = offset_ptr[cdf_idx];
            int32_t value = symbol_ptr[sym_idx] - offset;

            if (OVERFLOW_CODING)
            {
                const int32_t sign = value < 0;
                const int32_t max_value = esyms.size() - 1;
                int32_t gamma;
                if (sign)
                {
                    gamma = -value;
                    value = max_value;
                }
                else if (value >= max_value)
                {
                    gamma = value - max_value + 1;
                    value = max_value;
                }
                if (value == max_value)
                {
                    RansEncPutSymbol(&rans, &ptr, &bin_esyms[sign]);
                    int32_t n = 0;
                    while (gamma != 0)
                    {
                        RansEncPutSymbol(&rans, &ptr, &bin_esyms[gamma & 1]);
                        gamma >>= 1;
                        ++n;
                    }
                    while (--n > 0)
                    {
                        RansEncPutSymbol(&rans, &ptr, &bin_esyms[0]);
                    }
                }
            }

            RansEncPutSymbol(&rans, &ptr, &esyms[value]);
        }
        RansEncFlush(&rans, &ptr);
        assert(enc_buf_ptr <= ptr);
        enc_ptrs[unit_idx] = ptr;
    }
    Py_END_ALLOW_THREADS
    for (size_t unit_idx = 0; unit_idx < batch_size; ++unit_idx)
    {
        encoded_list.emplace_back(
            (const char *)enc_ptrs[unit_idx],
            (py::size_t)(enc_buf_ptrs[unit_idx] + enc_buf_sizes[unit_idx] - enc_ptrs[unit_idx]));
    }
    return encoded_list;
}

std::vector<py::bytes> IndexedRansCoder::encode(
    const PyArrayInt32 &symbol_array)
{
    if (overflow_coding)
        return _encode_with_indexes<true, false>(symbol_array, PyArrayInt32());
    else
        return _encode_with_indexes<false, false>(symbol_array, PyArrayInt32());
}

std::vector<py::bytes> IndexedRansCoder::encode_with_indexes(
    const PyArrayInt32 &symbol_array,
    const PyArrayInt32 &index_array)
{
    if (overflow_coding)
        return _encode_with_indexes<true, true>(symbol_array, index_array);
    else
        return _encode_with_indexes<false, true>(symbol_array, index_array);
}

template <bool OVERFLOW_CODING, bool WITH_INDEXES>
int IndexedRansCoder::_decode_with_indexes(
    const std::vector<std::string> &encoded_list,
    const PyArrayInt32 &index_array,
    PyArrayInt32 &symbol_array)
{
    const py::buffer_info index_arr_buf = index_array.request();
    const py::buffer_info symbol_arr_buf = symbol_array.request();
    const int32_t * const offset_ptr = reinterpret_cast<int32_t *>(offset_array.request().ptr);
    assert(batch_size == encoded_list.size());
    const size_t symbols_num = symbol_array.request().shape[1];
    if (WITH_INDEXES)
    {
        assert(symbol_array.request().shape == index_arr_buf.shape);
        assert(batch_size == index_arr_buf.shape[0]);
    }
    assert(symbol_array.writeable());
    assert(symbol_arr_buf.ndim == 2);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for collapse(1)
    for (size_t i = 0; i < batch_size; ++i)
    {
        const std::string &encoded = encoded_list[i];
        const int32_t * const index_ptr = WITH_INDEXES ? reinterpret_cast<int32_t *>(
            static_cast<uint8_t *>(index_arr_buf.ptr) + i * index_arr_buf.strides[0]) : nullptr;
        int32_t * const symbol_ptr = reinterpret_cast<int32_t *>(
            static_cast<uint8_t *>(symbol_arr_buf.ptr) + i * symbol_arr_buf.strides[0]);

        RansState rans;
        uint8_t *ptr = (uint8_t *)encoded.data();
        RansDecInit(&rans, &ptr);

        for (size_t j = 0; j < symbols_num; ++j)
        {
            const size_t &cdf_idx = WITH_INDEXES ? index_ptr[j] : j % cdfs.size();
            const int32_t &offset = offset_ptr[cdf_idx];
            const std::vector<RansDecSymbol> &dsyms = dsyms_list[cdf_idx];
            const VecUint32 &cdf = cdfs[cdf_idx];
            uint32_t cf = RansDecGet(&rans, PRECISION);
            int32_t value = std::upper_bound(cdf.begin() + 1, cdf.end(), cf) - cdf.begin() - 1;

            RansDecAdvanceSymbol(&rans, &ptr, &dsyms[value], PRECISION);

            if (OVERFLOW_CODING)
            {
                const int32_t max_value = dsyms.size() - 1;
                if (value == max_value)
                {
                    int32_t n = 0;
                    while (RansDecGet(&rans, 1) == 0)
                    {
                        ++n;
                        RansDecAdvanceSymbol(&rans, &ptr, &bin_dsyms[0], 1);
                    }
                    RansDecAdvanceSymbol(&rans, &ptr, &bin_dsyms[1], 1);
                    value = 1 << n;
                    while (--n >= 0)
                    {
                        int32_t bit = RansDecGet(&rans, 1);
                        RansDecAdvanceSymbol(&rans, &ptr, &bin_dsyms[bit], 1);
                        value |= bit << n;
                    }
                    int32_t sign = RansDecGet(&rans, 1);
                    RansDecAdvanceSymbol(&rans, &ptr, &bin_dsyms[sign], 1);
                    value = sign ? -value : value + max_value - 1;
                }
            }
            symbol_ptr[j] = value + offset;
        }
    }
    Py_END_ALLOW_THREADS
    return 0;
}

int IndexedRansCoder::decode(
    const std::vector<std::string> &encoded_list,
    PyArrayInt32 &symbol_array)
{
    if (overflow_coding)
        return _decode_with_indexes<true, false>(encoded_list, PyArrayInt32(), symbol_array);
    else
        return _decode_with_indexes<false, false>(encoded_list, PyArrayInt32(), symbol_array);
}

int IndexedRansCoder::decode_with_indexes(
    const std::vector<std::string> &encoded_list,
    const PyArrayInt32 &index_array,
    PyArrayInt32 &symbol_array)
{
    if (overflow_coding)
        return _decode_with_indexes<true, true>(encoded_list, index_array, symbol_array);
    else
        return _decode_with_indexes<false, true>(encoded_list, index_array, symbol_array);
}

BinaryRansCoder::BinaryRansCoder(
    uint32_t batch_size, size_t enc_buf_size
) : batch_size(batch_size)
{
    assert(batch_size > 0);
    assert(enc_buf_size > 0);
    enc_buf_ptrs.resize(batch_size);
    enc_buf_sizes.resize(batch_size);
    enc_ptrs.resize(batch_size);
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
        enc_buf_ptrs[idx] = new uint8_t[enc_buf_size];
        enc_buf_sizes[idx] = enc_buf_size;
    }
}

BinaryRansCoder::~BinaryRansCoder()
{
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
        delete[] enc_buf_ptrs[idx];
    }
}

std::vector<py::bytes> BinaryRansCoder::encode(
    const PyArrayBool &symbol_array,
    const PyArrayUint32 &prob_array)
{
    const py::buffer_info symbol_arr_buf = symbol_array.request();
    const py::buffer_info prob_array_buf = prob_array.request();
    const size_t symbols_num = symbol_arr_buf.shape[1];
    const size_t preserved_size = symbols_num / 4;
    assert(symbol_arr_buf.ndim == 2);
    assert(batch_size == symbol_arr_buf.shape[0]);
    assert(symbol_arr_buf.shape == prob_array_buf.shape);
    std::vector<py::bytes> encoded_list;
    encoded_list.reserve(batch_size);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for collapse(1)
    for (size_t unit_idx = 0; unit_idx < batch_size; ++unit_idx)
    {
        const bool * const symbol_ptr = reinterpret_cast<bool *>(
            static_cast<uint8_t *>(symbol_arr_buf.ptr) + unit_idx * symbol_arr_buf.strides[0]);
        const uint32_t * const prob_ptr = reinterpret_cast<uint32_t *>(
            static_cast<uint8_t *>(prob_array_buf.ptr) + unit_idx * prob_array_buf.strides[0]);

        if (preserved_size > enc_buf_sizes[unit_idx])
        {
            delete[] enc_buf_ptrs[unit_idx];
            enc_buf_ptrs[unit_idx] = new uint8_t[preserved_size];
            enc_buf_sizes[unit_idx] = preserved_size;
        }
        uint8_t* const &enc_buf_ptr = enc_buf_ptrs[unit_idx];
        const size_t &enc_buf_size = enc_buf_sizes[unit_idx];

        uint8_t *ptr = enc_buf_ptr + enc_buf_size;
        RansState rans;
        RansEncInit(&rans);

        for (size_t forward_sym_idx = 0; forward_sym_idx < symbols_num; ++forward_sym_idx)
        {
            const size_t sym_idx = symbols_num - 1 - forward_sym_idx;
            if (symbol_ptr[sym_idx] == 0)
                RansEncPut(&rans, &ptr, 0, PROB_SCALE - prob_ptr[sym_idx], PRECISION);
            else
                RansEncPut(&rans, &ptr, PROB_SCALE - prob_ptr[sym_idx], prob_ptr[sym_idx], PRECISION);
        }
        RansEncFlush(&rans, &ptr);
        assert(enc_buf_ptr <= ptr);
        enc_ptrs[unit_idx] = ptr;
    }
    Py_END_ALLOW_THREADS
    for (size_t unit_idx = 0; unit_idx < batch_size; ++unit_idx)
    {
        encoded_list.emplace_back(
            (const char *)enc_ptrs[unit_idx],
            (py::size_t)(enc_buf_ptrs[unit_idx] + enc_buf_sizes[unit_idx] - enc_ptrs[unit_idx]));
    }
    return encoded_list;
}


int BinaryRansCoder::decode(
    const std::vector<std::string> &encoded_list,
    const PyArrayUint32 &prob_array,
    PyArrayBool &symbol_array)
{
    const py::buffer_info prob_array_buf = prob_array.request();
    const py::buffer_info symbol_arr_buf = symbol_array.request();
    const size_t symbols_num = symbol_arr_buf.shape[1];
    assert(symbol_array.writeable());
    assert(symbol_arr_buf.ndim == 2);
    assert(batch_size == encoded_list.size());
    assert(symbol_arr_buf.shape == prob_array_buf.shape);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for collapse(1)
    for (size_t i = 0; i < batch_size; ++i)
    {
        const std::string &encoded = encoded_list[i];
        const uint32_t * const prob_ptr = reinterpret_cast<uint32_t *>(
            static_cast<uint8_t *>(prob_array_buf.ptr) + i * prob_array_buf.strides[0]);
        bool * const symbol_ptr = reinterpret_cast<bool *>(
            static_cast<uint8_t *>(symbol_arr_buf.ptr) + i * symbol_arr_buf.strides[0]);

        RansState rans;
        uint8_t *ptr = (uint8_t *)encoded.data();
        RansDecInit(&rans, &ptr);

        for (size_t j = 0; j < symbols_num; ++j)
        {
            if (RansDecGet(&rans, PRECISION) < PROB_SCALE - prob_ptr[j])
            {
                symbol_ptr[j] = 0;
                RansDecAdvance(&rans, &ptr, 0, PROB_SCALE - prob_ptr[j], PRECISION);
            }
            else
            {
                symbol_ptr[j] = 1;
                RansDecAdvance(&rans, &ptr, PROB_SCALE - prob_ptr[j], prob_ptr[j], PRECISION);
            }
        }
    }
    Py_END_ALLOW_THREADS
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batched_pmf_to_quantized_cdf", &batched_pmf_to_quantized_cdf,
          "Return batched quantized CDFs for given PMFs");
    py::class_<IndexedRansCoder>(m, "IndexedRansCoder")
        .def(py::init<bool, uint32_t, size_t>(),
            py::arg("overflow_coding"), py::arg("batch_size"),
            py::arg_v("enc_buf_size", DEFAULT_ENC_BUF_SIZE))
        .def("init_with_pmfs", &IndexedRansCoder::init_with_pmfs)
        .def("init_with_quantized_cdfs", &IndexedRansCoder::init_with_quantized_cdfs)
        .def("get_cdfs", &IndexedRansCoder::get_cdfs)
        .def("get_offset_array", &IndexedRansCoder::get_offset_array)
        .def("encode", &IndexedRansCoder::encode)
        .def("decode", &IndexedRansCoder::decode)
        .def("encode_with_indexes", &IndexedRansCoder::encode_with_indexes)
        .def("decode_with_indexes", &IndexedRansCoder::decode_with_indexes);
    py::class_<BinaryRansCoder>(m, "BinaryRansCoder")
        .def(py::init<uint32_t, size_t>(),
            py::arg("batch_size"), py::arg_v("enc_buf_size", DEFAULT_ENC_BUF_SIZE))
        .def("encode", &BinaryRansCoder::encode)
        .def("decode", &BinaryRansCoder::decode);
}
