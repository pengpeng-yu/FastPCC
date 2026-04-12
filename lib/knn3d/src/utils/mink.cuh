/*
 * Adapted from PyTorch3D MinK helpers.
 */

#pragma once

template <typename key_t, typename value_t>
class MinK {
 public:
  __device__ MinK(key_t* keys, value_t* vals, int K)
      : keys(keys), vals(vals), K(K), _size(0) {}

  __device__ __forceinline__ void add(const key_t& key, const value_t& val) {
    if (_size < K) {
      keys[_size] = key;
      vals[_size] = val;
      if (_size == 0 || key > max_key) {
        max_key = key;
        max_idx = _size;
      }
      _size++;
    } else if (key < max_key) {
      keys[max_idx] = key;
      vals[max_idx] = val;
      max_key = key;
      for (int k = 0; k < K; ++k) {
        key_t cur_key = keys[k];
        if (cur_key > max_key) {
          max_key = cur_key;
          max_idx = k;
        }
      }
    }
  }

  __device__ __forceinline__ int size() {
    return _size;
  }

 private:
  key_t* keys;
  value_t* vals;
  int K;
  int _size;
  key_t max_key;
  int max_idx;
};
