#include <torch/extension.h>


std::vector<uint32_t> pmf_to_quantized_cdf(std::vector<double> &pmf, uint32_t precision)
{
    assert(precision <= 32);
    const uint32_t prob_scale = 1u << precision;
    std::vector<uint32_t> cdf(pmf.size() + 2);
    cdf[0] = 0u;
    cdf.back() = prob_scale;

#ifndef	NDEBUG
    for (auto &p : pmf)
    {
        assert(p >= 0 && std::isfinite(p));
    }
#endif
    double total = std::accumulate(pmf.begin(), pmf.end(), 0.);
    double overflow = std::max(1. - total, 0.);
    total += overflow;
    pmf.push_back(overflow);
    std::vector<double> float_cdf(pmf.size());

    std::partial_sum(pmf.begin(), pmf.end(), float_cdf.begin());
    std::transform(float_cdf.begin(), float_cdf.end(), cdf.begin() + 1,
                   [prob_scale, total](auto p)
                   { return static_cast<uint32_t>(std::round(prob_scale * (p / total))); });

    for (size_t i = 0; i < pmf.size(); i++)
    {
        if (cdf[i + 1] == cdf[i])  // symbol i was set to zero freq
        {
            // find best symbol to steal frequency from (try to steal from low-freq ones)
            uint32_t best_freq = ~0u;
            size_t best_steal = 0;
            bool found_steal = false;
            for (size_t j = 0; j < pmf.size(); j++)
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
    assert(cdf[0] == 0 && cdf.back() == prob_scale);
    for (size_t i = 0; i < pmf.size(); i++)
    {
        assert(cdf[i + 1] > cdf[i]);
    }
#endif
    return cdf;
}

std::vector<std::vector<uint32_t>>
batched_pmf_to_quantized_cdf(std::vector<std::vector<double>> &pmfs, uint32_t precision)
{
    std::vector<std::vector<uint32_t>> cdfs;
    cdfs.reserve(pmfs.size());
    for (auto &pmf : pmfs)
    {
        cdfs.push_back(pmf_to_quantized_cdf(pmf, precision));
    }
    return cdfs;
}
