#ifndef INDEXED_RANS_CODER_H_
#define INDEXED_RANS_CODER_H_

#include <torch/extension.h>
#include "rans_byte.h"


class IndexedRansCoder
{
public:
    IndexedRansCoder(uint32_t precision);
    int init_with_pmfs(
        std::vector<std::vector<double>> &pmfs,
        std::vector<int32_t> &offset_list
    );
    int init_with_quantized_cdfs(
        std::vector<std::vector<uint32_t>> &quantized_cdfs,
        std::vector<int32_t> &offset_list
    );
    std::vector<py::bytes> encode_with_indexes(
        const std::vector<std::vector<int32_t>> &symbols_list, 
        const std::vector<std::vector<int32_t>> &indexes_list
    );
    std::vector<std::vector<int32_t>> decode_with_indexes(
        const std::vector<std::string> &encoded_list, 
        const std::vector<std::vector<int32_t>> &indexes_list
    );

private:
    const uint32_t precision;
    
    std::vector<std::vector<RansEncSymbol>> esyms_list;
    std::vector<std::vector<RansDecSymbol>> dsyms_list;
    std::vector<int32_t> offset_list;
    std::vector<std::vector<uint16_t>> cum2sym_list;
    std::vector<RansEncSymbol> bin_esyms;
    std::vector<RansDecSymbol> bin_dsyms;
};

#endif  // INDEXED_RANS_CODER_H_
