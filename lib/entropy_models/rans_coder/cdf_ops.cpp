#include <torch/extension.h>


template <bool OVERFLOW_CODING>
std::vector<uint32_t>
pmf_to_quantized_cdf(double * const pmf, const size_t pmf_size, int32_t * const offset)
{
    std::vector<uint32_t> cdf;
    if (OVERFLOW_CODING)
        cdf.resize(pmf_size + 2);
    else
        cdf.resize(pmf_size + 1);
    cdf[0] = 0u;

    for (size_t i = 0; i < pmf_size; ++i)
    {
        assert(pmf[i] >= 0 && std::isfinite(pmf[i]));
    }
    double total = std::accumulate(pmf, pmf + pmf_size, 0.);
    double overflow = std::max(1. - total, 0.);
    if (OVERFLOW_CODING) total += overflow;

    std::partial_sum(pmf, pmf + pmf_size, pmf);
    std::transform(pmf, pmf + pmf_size, cdf.begin() + 1,
                   [total](auto p)
                   { return static_cast<uint32_t>(std::round(PROB_SCALE * (p / total))); });
    cdf.back() = PROB_SCALE;

    assert(cdf.size() >= 3);

    if (OVERFLOW_CODING)
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
        offset[0] += cdf_start;

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

    // calculate updated freqs and make sure we didn't screw anything up
    assert(cdf[0] == 0 && cdf.back() == PROB_SCALE);
    for (size_t i = 0; i < cdf.size() - 1; i++)
    {
        assert(cdf[i + 1] > cdf[i]);
    }
    return cdf;
}

template <bool OVERFLOW_CODING>
std::vector<std::vector<uint32_t>> _batched_pmf_to_quantized_cdf(
    PyArrayDouble &pmf_array, PyArrayInt32 &offset_array)
{
    const py::buffer_info pmf_arr_buf = pmf_array.request();
    const py::buffer_info offset_arr_buf = offset_array.request();
    int32_t * const offset_ptr = reinterpret_cast<int32_t *>(offset_arr_buf.ptr);
    assert(pmf_arr_buf.ndim == 2);
    assert(offset_arr_buf.ndim == 1);
    assert(pmf_arr_buf.shape[0] == offset_arr_buf.shape[0]);
    std::vector<std::vector<uint32_t>> cdfs;
    cdfs.resize(pmf_arr_buf.shape[0]);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for collapse(1)
    for (size_t i = 0; i < pmf_arr_buf.shape[0]; ++i)
    {
        double * const pmf_ptr = reinterpret_cast<double *>(
            static_cast<uint8_t *>(pmf_arr_buf.ptr) + i * pmf_arr_buf.strides[0]);
        cdfs[i] = pmf_to_quantized_cdf<OVERFLOW_CODING>(pmf_ptr, pmf_arr_buf.shape[1], offset_ptr + i);
    }
    Py_END_ALLOW_THREADS
    return cdfs;
}

std::vector<std::vector<uint32_t>> batched_pmf_to_quantized_cdf(
    PyArrayDouble &pmf_array, PyArrayInt32 &offset_array, bool overflow_coding)
{
    if (overflow_coding)
        return _batched_pmf_to_quantized_cdf<true>(pmf_array, offset_array);
    else
        return _batched_pmf_to_quantized_cdf<false>(pmf_array, offset_array);
}
