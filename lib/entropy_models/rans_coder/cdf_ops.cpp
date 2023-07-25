#include <torch/extension.h>


std::vector<uint32_t>
pmf_to_quantized_cdf(std::vector<double> &pmf, int32_t &offset, bool overflow_coding)
{
    std::vector<uint32_t> cdf;
    if (overflow_coding)
        cdf.resize(pmf.size() + 2);
    else
        cdf.resize(pmf.size() + 1);
    cdf[0] = 0u;

#ifndef	NDEBUG
    for (auto &p : pmf)
    {
        assert(p >= 0 && std::isfinite(p));
    }
#endif
    double total = std::accumulate(pmf.begin(), pmf.end(), 0.);
    if (overflow_coding)
    {
        double overflow = std::max(1. - total, 0.);
        total += overflow;
        pmf.push_back(overflow);
    }

    std::partial_sum(pmf.begin(), pmf.end(), pmf.begin());
    std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                   [total](auto p)
                   { return static_cast<uint32_t>(std::round(PROB_SCALE * (p / total))); });
    cdf.back() = PROB_SCALE;

    assert(cdf.size() >= 3);
    if (overflow_coding)
    {
        size_t cdf_start = 0, cdf_end = 0;
        for (size_t i = 0; i < cdf.size() - 1; i++)
        {
            if (cdf[i + 1] != cdf[i])
            {
                cdf_start = i;
                break;
            }
        }
        for (size_t i = cdf.size() - 2; i > 0; i--)
        {
            if (cdf[i - 1] != cdf[i])
            {
                cdf_end = i;
                break;
            }
        }
        offset += cdf_start;

        if (cdf_start > cdf_end)
        {
            assert(cdf_start = cdf.size() - 2);
            cdf_start = cdf.size() - 3;
            cdf_end = cdf_start + 1;
        }
        size_t cdf_size =  cdf_end - cdf_start + 1 + 1;
        for (size_t i = 0; i < cdf_size - 1; i++)
        {
            cdf[i] = cdf[i + cdf_start];
        }
        cdf.resize(cdf_size);
        cdf.back() = PROB_SCALE;
    }

    for (size_t i = 0; i < cdf.size() - 1; i++)
    {
        if (cdf[i + 1] == cdf[i])  // symbol i was set to zero freq
        {
            // find best symbol to steal frequency from (try to steal from low-freq ones)
            uint32_t best_freq = ~0u;
            size_t best_steal = 0;
            bool found_steal = false;
            for (size_t j = 0; j < cdf.size() - 1; j++)
            {
                uint32_t freq = cdf[j + 1] - cdf[j];
                if (freq > 1 && freq < best_freq)
                {
                    best_freq = freq;
                    best_steal = j;
                    found_steal = true;
                }
            }
            assert(found_steal);

            // and steal from it!
            if (best_steal < i)
            {
                for (size_t j = best_steal + 1; j <= i; j++)
                    cdf[j]--;
            }
            else
            {
                assert(best_steal > i);
                for (size_t j = i + 1; j <= best_steal; j++)
                    cdf[j]++;
            }
        }
    }

#ifndef	NDEBUG
    // calculate updated freqs and make sure we didn't screw anything up
    assert(cdf[0] == 0 && cdf.back() == PROB_SCALE);
    for (size_t i = 0; i < cdf.size() - 1; i++)
    {
        assert(cdf[i + 1] > cdf[i]);
    }
#endif
    return cdf;
}

std::tuple<std::vector<std::vector<uint32_t>>, std::vector<int32_t>>
batched_pmf_to_quantized_cdf(
    std::vector<std::vector<double>> &pmfs,
    std::vector<int32_t> &offsets,
    bool overflow_coding)
{
    std::vector<std::vector<uint32_t>> cdfs;
    cdfs.reserve(pmfs.size());
    for (size_t i = 0; i < pmfs.size(); ++i)
    {
        auto &pmf = pmfs[i];
        auto &offset = offsets[i];
        cdfs.push_back(pmf_to_quantized_cdf(pmf, offset, overflow_coding));
    }
    return std::make_tuple(cdfs, offsets);
}
