#include <torch/extension.h>
#include "indexed_rans_coder.h"
#include "cdf_ops.cpp"


IndexedRansCoder::IndexedRansCoder(
    uint32_t precision, bool overflow_coding
) : precision(precision), overflow_coding(overflow_coding)
{
    assert(precision <= 16 && precision > 0);
    bin_esyms.resize(2);
    bin_dsyms.resize(2);
    RansEncSymbolInit(&bin_esyms[0], 0, 1, 1);
    RansDecSymbolInit(&bin_dsyms[0], 0, 1);
    RansEncSymbolInit(&bin_esyms[1], 1, 1, 1);
    RansDecSymbolInit(&bin_dsyms[1], 1, 1);
}

int IndexedRansCoder::init_with_pmfs(
    std::vector<std::vector<double>> &pmfs,
    std::vector<int32_t> &offsets)
{
    auto &&ret = batched_pmf_to_quantized_cdf(pmfs, offsets, precision, overflow_coding);
    return init_with_quantized_cdfs(std::get<0>(ret), std::get<1>(ret));
}

int IndexedRansCoder::init_with_quantized_cdfs(
    std::vector<std::vector<uint32_t>> &cdfs,
    std::vector<int32_t> &offsets)
{
    const uint32_t prob_scale = 1u << precision;
    size_t cdfs_num = cdfs.size();
    esyms_list.resize(cdfs_num);
    dsyms_list.resize(cdfs_num);
#ifdef USE_CUM2SYM_LIST
    cum2sym_list.resize(cdfs_num);
#endif
    assert(cdfs_num == offsets.size());

    for (size_t cdf_idx = 0; cdf_idx < cdfs.size(); ++cdf_idx)
    {
        const auto &cdf = cdfs[cdf_idx];
        auto &esyms = esyms_list[cdf_idx];
        auto &dsyms = dsyms_list[cdf_idx];

        assert(cdf[0] == 0 && cdf.back() == prob_scale);
        assert((cdf.size() - 1) < (1u << precision));
        const uint16_t symbols_num = uint16_t(cdf.size() - 1);

        esyms.resize(symbols_num);
        dsyms.resize(symbols_num);
#ifdef USE_CUM2SYM_LIST
        auto &cum2sym = cum2sym_list[cdf_idx];
        cum2sym.resize(prob_scale);
#endif
        for (uint16_t sym = 0; sym < symbols_num; ++sym)
        {
            RansEncSymbolInit(&esyms[sym], cdf[sym], cdf[sym + 1] - cdf[sym], precision);
            RansDecSymbolInit(&dsyms[sym], cdf[sym], cdf[sym + 1] - cdf[sym]);
#ifdef USE_CUM2SYM_LIST
            for (size_t cum_idx = cdf[sym]; cum_idx < cdf[sym + 1]; ++cum_idx)
            {
                cum2sym[cum_idx] = sym;
            }
#endif
        }
    }

    this->offsets = std::move(offsets);
#ifndef USE_CUM2SYM_LIST
    this->cdfs = std::move(cdfs);
#endif
    return 0;
}

std::vector<py::bytes> IndexedRansCoder::encode_with_indexes(
    const std::vector<std::vector<int32_t>> &symbols_list,
    const std::vector<std::vector<int32_t>> &indexes_list)
{
    std::vector<py::bytes> encoded_list;
    size_t coding_units_num = symbols_list.size();
    assert(coding_units_num == indexes_list.size());

    size_t max_symbols_num = 0;
    for (size_t unit_idx = 0; unit_idx < coding_units_num; ++unit_idx)
    {
        const size_t symbols_num = symbols_list[unit_idx].size();
        if (symbols_num > max_symbols_num)
        {
            max_symbols_num = symbols_num;
        }
    }
    size_t out_max_size = 4 * max_symbols_num;
    uint8_t *out_buf = new uint8_t[out_max_size];

    for (size_t unit_idx = 0; unit_idx < coding_units_num; ++unit_idx)
    {
        const auto &symbols = symbols_list[unit_idx];
        const auto &indexes = indexes_list[unit_idx];
        const size_t symbols_num = symbols.size();
        assert(symbols_num == indexes.size());

        uint8_t *ptr = out_buf + out_max_size;
        uint8_t *rans_begin;
        RansState rans;
        RansEncInit(&rans);

        for (size_t forward_sym_idx = 0; forward_sym_idx < symbols_num; ++forward_sym_idx)
        {
            const size_t sym_idx = symbols_num - 1 - forward_sym_idx;
            const auto &cdf_idx = indexes[sym_idx];
            const auto &esyms = esyms_list[cdf_idx];
            const auto &offset = offsets[cdf_idx];
            int32_t value = symbols[sym_idx] - offset;
            assert(cdf_idx >= 0);
            assert(cdf_idx < esyms_list.size());

            if (overflow_coding)
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
        rans_begin = ptr;
        assert(out_buf <= rans_begin);
        encoded_list.emplace_back((const char *)ptr, (py::size_t)(out_buf + out_max_size - rans_begin));
    }
    delete[] out_buf;
    return encoded_list;
}

std::vector<std::vector<int32_t>> IndexedRansCoder::decode_with_indexes(
    const std::vector<std::string> &encoded_list,
    const std::vector<std::vector<int32_t>> &indexes_list)
{
    size_t coding_units_num = encoded_list.size();
    assert(coding_units_num == indexes_list.size());
    std::vector<std::vector<int32_t>> symbols_list(coding_units_num);

    for (size_t i = 0; i < coding_units_num; ++i)
    {
        const auto &encoded = encoded_list[i];
        const auto &indexes = indexes_list[i];
        const size_t symbols_num = indexes.size();
        auto &symbols = symbols_list[i];
        symbols.resize(symbols_num);

        RansState rans;
        uint8_t *ptr = (uint8_t *)encoded.data();
        RansDecInit(&rans, &ptr);

        for (size_t j = 0; j < symbols_num; ++j)
        {
            const auto &cdf_idx = indexes[j];
            const auto &offset = offsets[cdf_idx];
            const auto &dsyms = dsyms_list[cdf_idx];
#ifdef USE_CUM2SYM_LIST
            const auto &cum2sym = cum2sym_list[cdf_idx];
            int32_t value = cum2sym[RansDecGet(&rans, precision)];
#else
            const auto &cdf = cdfs[cdf_idx];
            int32_t value = -1;
            uint32_t cf = RansDecGet(&rans, precision);
            for (size_t k = 1; k < cdf.size(); ++k)
            {
                if (cdf[k] > cf)
                {
                    value = k - 1;
                    break;
                }
            }
#endif
            RansDecAdvanceSymbol(&rans, &ptr, &dsyms[value], precision);

            if (overflow_coding)
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
            symbols[j] = value + offset;
        }
    }
    return symbols_list;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf,
          "Return quantized CDF for a given PMF");
    m.def("batched_pmf_to_quantized_cdf", &batched_pmf_to_quantized_cdf,
          "Return batched quantized CDF for a given PMF");
    py::class_<IndexedRansCoder>(m, "IndexedRansCoder")
        .def(py::init<uint32_t, bool>())
        .def("init_with_pmfs", &IndexedRansCoder::init_with_pmfs)
        .def("init_with_quantized_cdfs", &IndexedRansCoder::init_with_quantized_cdfs)
        .def("encode_with_indexes", &IndexedRansCoder::encode_with_indexes)
        .def("decode_with_indexes", &IndexedRansCoder::decode_with_indexes);
}
