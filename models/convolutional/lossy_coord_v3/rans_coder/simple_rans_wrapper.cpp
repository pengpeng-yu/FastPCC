#include <torch/extension.h>
#include "simple_rans_wrapper.h"


RansEncoder::RansEncoder(
    size_t enc_buf_size
) : enc_buf_size(enc_buf_size)
{
    assert(enc_buf_size > 0);
    enc_buf_ptr = new uint8_t[enc_buf_size];
    enc_ptr = enc_buf_ptr + enc_buf_size;
    RansEncInit(&rans);
}

RansEncoder::~RansEncoder()
{
    delete[] enc_buf_ptr;
}

int RansEncoder::init_with_quantized_cdfs(
    const PyArrayUint16 &cdf_arr)
{
    const py::buffer_info cdf_arr_buf = cdf_arr.request();
    const size_t cdfs_num = cdf_arr_buf.shape[0];
    const size_t symbols_num = cdf_arr_buf.shape[1];
    assert(symbols_num <= (1u << PRECISION));
    esyms_list.resize(cdfs_num);

    for (size_t cdf_idx = 0; cdf_idx < cdfs_num; ++cdf_idx)
    {
        const uint16_t * const cdf_arr_ptr = reinterpret_cast<uint16_t *>(
            static_cast<uint8_t *>(cdf_arr_buf.ptr) + cdf_idx * cdf_arr_buf.strides[0]);

        std::vector<RansEncSymbol> &esyms = esyms_list[cdf_idx];
        esyms.resize(symbols_num);

        for (size_t sym = 0; sym < symbols_num; ++sym)
        {
            const uint32_t &cur_cd = sym == 0 ? 0 : cdf_arr_ptr[sym - 1];
            const uint32_t &next_cd = sym == symbols_num - 1 ? PROB_SCALE : cdf_arr_ptr[sym];
            RansEncSymbolInit(&esyms[sym], cur_cd, next_cd - cur_cd, PRECISION);
        }
    }

    return 0;
}

uint64_t RansEncoder::encode_with_precomp(
    const PyArrayUint16 &cdf_arr, const PyArrayUint16 &symbol_arr)
{
    init_with_quantized_cdfs(cdf_arr);
    const py::buffer_info symbol_arr_buf = symbol_arr.request();
    const size_t symbols_num = symbol_arr_buf.shape[0];
    assert(symbols_num == esyms_list.size() || (esyms_list.size() == 1));
    const uint16_t * const symbol_ptr = reinterpret_cast<uint16_t *>(symbol_arr_buf.ptr);

    for (int32_t sym_idx = symbols_num - 1; sym_idx >= 0; --sym_idx)
    {
        const std::vector<RansEncSymbol> &esyms = esyms_list[sym_idx % esyms_list.size()];
        RansEncPutSymbol(&rans, &enc_ptr, &esyms[symbol_ptr[sym_idx]]);
    }

    esyms_list.clear();
    return enc_buf_ptr + enc_buf_size - enc_ptr;
}

uint64_t RansEncoder::encode(
    const PyArrayUint16 &cdf_arr, const PyArrayUint16 &symbol_arr)
{
    const py::buffer_info cdf_arr_buf = cdf_arr.request();
    const size_t cdfs_num = cdf_arr_buf.shape[0];
    const auto max_symbol_num = cdf_arr_buf.shape[1];
    assert(max_symbol_num <= PROB_SCALE);
    uint16_t * cdf_arr_ptr = reinterpret_cast<uint16_t *>(cdf_arr_buf.ptr);
    if (cdfs_num != 1)
        cdf_arr_ptr += cdfs_num * max_symbol_num;

    const py::buffer_info symbol_arr_buf = symbol_arr.request();
    const size_t symbols_num = symbol_arr_buf.shape[0];
    assert(symbols_num == cdfs_num || (cdfs_num == 1));
    const uint16_t * symbol_ptr = reinterpret_cast<uint16_t *>(symbol_arr_buf.ptr) + symbols_num;

    for (size_t idx = 0; idx < symbols_num; ++idx)
    {
        if (cdfs_num != 1)
            cdf_arr_ptr -= max_symbol_num;
        --symbol_ptr;
        const uint16_t &sym = *symbol_ptr;
        const uint32_t &cur = sym == 0 ? 0 : cdf_arr_ptr[sym - 1];
        const uint32_t &next = sym == max_symbol_num - 1 ? PROB_SCALE : cdf_arr_ptr[sym];
        RansEncPut(&rans, &enc_ptr, cur, next - cur, PRECISION);
    }

    return enc_buf_ptr + enc_buf_size - enc_ptr;
}

py::bytes RansEncoder::flush()
{
    RansEncFlush(&rans, &enc_ptr);
    assert(enc_buf_ptr <= enc_ptr);
    py::bytes encoded((const char *)enc_ptr, (py::size_t)(enc_buf_ptr + enc_buf_size - enc_ptr));
    enc_ptr = enc_buf_ptr + enc_buf_size;
    RansEncInit(&rans);
    return encoded;
}

RansDecoder::RansDecoder()
{}

int RansDecoder::flush(
    const py::bytes &encoded)
{
    ptr = reinterpret_cast<uint8_t *>(PyBytes_AS_STRING(encoded.ptr()));
    RansDecInit(&rans, &ptr);
    return 0;
}

int RansDecoder::init_with_quantized_cdfs(
    const PyArrayUint16 &cdf_arr)
{
    const py::buffer_info cdf_arr_buf = cdf_arr.request();
    const size_t cdfs_num = cdf_arr_buf.shape[0];
    const size_t symbols_num = cdf_arr_buf.shape[1];
    assert(symbols_num <= (1u << PRECISION));
    dsyms_list.resize(cdfs_num);

    for (size_t cdf_idx = 0; cdf_idx < cdfs_num; ++cdf_idx)
    {
        const uint16_t * const cdf_arr_ptr = reinterpret_cast<uint16_t *>(
            static_cast<uint8_t *>(cdf_arr_buf.ptr) + cdf_idx * cdf_arr_buf.strides[0]);

        std::vector<RansDecSymbol> &dsyms = dsyms_list[cdf_idx];
        dsyms.resize(symbols_num);

        for (size_t sym = 0; sym < symbols_num; ++sym)
        {
            const uint32_t &cur_cd = sym == 0 ? 0 : cdf_arr_ptr[sym - 1];
            const uint32_t &next_cd = sym == symbols_num - 1 ? PROB_SCALE : cdf_arr_ptr[sym];
            RansDecSymbolInit(&dsyms[sym], cur_cd, next_cd - cur_cd);
        }
    }

    return 0;
}

int RansDecoder::decode_with_precomp(
    PyArrayUint16 &cdf_arr,
    PyArrayUint16 &symbol_arr)
{
    init_with_quantized_cdfs(cdf_arr);
    const py::buffer_info symbol_arr_buf = symbol_arr.request();
    const size_t symbols_num = symbol_arr_buf.shape[0];
    assert(symbol_arr.writeable());
    assert(symbols_num == dsyms_list.size() || (dsyms_list.size() == 1));
    const auto max_symbol_num = dsyms_list[0].size();
    assert(max_symbol_num <= PROB_SCALE);
    uint16_t * const symbol_ptr = reinterpret_cast<uint16_t *>(symbol_arr_buf.ptr);
    const py::buffer_info cdf_arr_buf = cdf_arr.request();

    for (size_t idx = 0; idx < symbols_num; ++idx)
    {
        const std::vector<RansDecSymbol> &dsyms = dsyms_list[idx % dsyms_list.size()];
        const uint16_t * const cdf_arr_ptr = reinterpret_cast<uint16_t *>(
            static_cast<uint8_t *>(cdf_arr_buf.ptr) + (idx % dsyms_list.size()) * cdf_arr_buf.strides[0]);
        auto cf = RansDecGet(&rans, PRECISION);
        auto sym = std::upper_bound(cdf_arr_ptr, cdf_arr_ptr + max_symbol_num, cf) - cdf_arr_ptr;
        if (sym > max_symbol_num - 1)
            sym = max_symbol_num - 1;
        RansDecAdvanceSymbol(&rans, &ptr, &dsyms[sym], PRECISION);
        symbol_ptr[idx] = sym;
    }

    dsyms_list.clear();
    return 0;
}

int RansDecoder::decode(
    PyArrayUint16 &cdf_arr,
    PyArrayUint16 &symbol_arr)
{
    const py::buffer_info cdf_arr_buf = cdf_arr.request();
    const size_t cdfs_num = cdf_arr_buf.shape[0];
    const auto max_symbol_num = cdf_arr_buf.shape[1];
    assert(max_symbol_num <= PROB_SCALE);
    uint16_t * cdf_arr_ptr = reinterpret_cast<uint16_t *>(cdf_arr_buf.ptr);

    const py::buffer_info symbol_arr_buf = symbol_arr.request();
    const size_t symbols_num = symbol_arr_buf.shape[0];
    assert(symbol_arr.writeable());
    assert(symbols_num == cdfs_num || (cdfs_num == 1));
    uint16_t * symbol_ptr = reinterpret_cast<uint16_t *>(symbol_arr_buf.ptr);

    for (size_t idx = 0; idx < symbols_num; ++idx)
    {
        auto cf = RansDecGet(&rans, PRECISION);
        auto sym = std::upper_bound(cdf_arr_ptr, cdf_arr_ptr + max_symbol_num, cf) - cdf_arr_ptr;
        if (sym > max_symbol_num - 1)
            sym = max_symbol_num - 1;

        const uint32_t &cur = sym == 0 ? 0 : cdf_arr_ptr[sym - 1];
        const uint32_t &next = sym == max_symbol_num - 1 ? PROB_SCALE : cdf_arr_ptr[sym];
        RansDecAdvance(&rans, &ptr, cur, next - cur, PRECISION);
        *symbol_ptr = sym;
        ++symbol_ptr;
        if (cdfs_num != 1)
            cdf_arr_ptr += max_symbol_num;
    }

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<RansEncoder>(m, "RansEncoder")
        .def(py::init<size_t>(), py::arg_v("enc_buf_size", DEFAULT_ENC_BUF_SIZE))
        .def("encode_with_precomp", &RansEncoder::encode_with_precomp)
        .def("encode", &RansEncoder::encode)
        .def("flush", &RansEncoder::flush);
    py::class_<RansDecoder>(m, "RansDecoder")
        .def(py::init())
        .def("flush", &RansDecoder::flush)
        .def("decode_with_precomp", &RansDecoder::decode_with_precomp)
        .def("decode", &RansDecoder::decode);
}
