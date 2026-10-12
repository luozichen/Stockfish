/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Input features and network structure used in NNUE evaluation function

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <array>
#include <cstdint>
#include <iosfwd>

#include "features/half_ka_v2_hm.h"
#include "features/full_threats.h"
#include "features/pp_3wide.h"
#include "layers/affine_transform.h"
#include "layers/affine_transform_sparse_input.h"
#include "layers/clipped_relu.h"
#include "layers/sqr_clipped_relu.h"
#include "nnue_common.h"
#include "nnz_helper.h"

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using ThreatFeatureSet = Features::FullThreats;
using PairFeatureSet   = Features::PP_3Wide;
using PSQFeatureSet    = Features::HalfKAv2_hm;

// Number of input feature dimensions after conversion
constexpr IndexType L1 = 1024;
constexpr int       L2 = 32;
constexpr int       L3 = 32;

constexpr IndexType LayerStacks = 8;

// The 128 values the output layer reads: both hidden layers, each in their
// squared-clipped and clipped flavors. These are stashed per thread during
// propagate(): architecture-independent network outputs, usable for cheap
// hashing right after a fresh evaluation.
constexpr int FC2InputSize = L2 * 2 + L3 * 2;

inline int8_t* fc2_input_stash() {
    static thread_local int8_t stash[FC2InputSize] = {0};
    return stash;
}

// Width of the SimHash key over the stashed activations: SketchBits balanced
// +-1 projections, one sign bit each, addressing the sketch correction source.
constexpr int SketchBits = 12;

// SketchBits balanced +-1 projection patterns over the stashed output-layer
// input, generated at compile time: each pattern holds exactly half +1 and
// half -1 entries, so raw projections are centered when activation means are
// uniform. The coefficients are permuted from canonical to stored layout:
// under USE_PAIR_ACTIVATIONS (and LASX) the pair propagation writes 4-value
// chunks interleaved per 128-bit lane (the layout the next layer's weight
// scrambling compensates for), keeping the sketch architecture independent.
constexpr auto SketchPatterns = []() {
    std::array<std::array<int8_t, FC2InputSize>, SketchBits> q{};
    std::array<int, FC2InputSize>                            slot{};
    u64                                                      s = 0x9E3779B97F4A7C15ULL;
    for (auto& row : q)
    {
        for (int i = 0; i < FC2InputSize; ++i)
        {
            slot[i] = i;
            row[i]  = 1;
        }
        // flip exactly half of the positions to -1 (partial Fisher-Yates)
        for (int i = 0; i < FC2InputSize / 2; ++i)
        {
            s           = s * 6364136223846793005ULL + 1442695040888963407ULL;
            const int j = i + int((s >> 32) % unsigned(FC2InputSize - i));
            const int t = slot[i], r = slot[j];
            slot[i]     = r;
            slot[j]     = t;
            row[r]      = -1;
        }
    }

    const auto chunk_lane = [](int c) {
#if defined(USE_PAIR_ACTIVATIONS) || defined(USE_LASX)
    #if defined(USE_AVX512)
        return (c % 4) * 2 + c / 4;
    #else
        return (c % 2) * 4 + c / 2;
    #endif
#else
        return c;
#endif
    };
    for (auto& row : q)
    {
        const std::array<int8_t, FC2InputSize> p = row;
        for (int d = 0; d < FC2InputSize; ++d)
        {
            const int phys = (d / 32) * 32 + chunk_lane((d % 32) / 4) * 4 + d % 4;
            row[phys]      = p[d];
        }
    }
    return q;
}();

// Raw SimHash projections of the stashed activations:
// proj[k] = sum_i SketchPatterns[k][i] * stash[i].
// The stash is non-negative 8-bit, which the unsigned*signed dot products
// below exploit. All intermediate sums stay far within their widths: a
// maddubs lane holds two products (|.| <= 254), the i16 accumulation at most
// four such lanes (<= 1016), the madd to 32-bit at most 8 products.
inline void sketch_projections(const int8_t* stash, i32* proj) {
#if defined(USE_AVX2)
    const __m256i ones = _mm256_set1_epi16(1);
    for (int k = 0; k < SketchBits; ++k)
    {
        const auto* pat = reinterpret_cast<const __m256i*>(SketchPatterns[k].data());
        const auto* in  = reinterpret_cast<const __m256i*>(stash);
        // loadu everywhere: the patterns carry no alignment guarantee
        const __m256i s0 = _mm256_maddubs_epi16(_mm256_loadu_si256(in + 0), _mm256_loadu_si256(pat + 0));
        const __m256i s1 = _mm256_maddubs_epi16(_mm256_loadu_si256(in + 1), _mm256_loadu_si256(pat + 1));
        const __m256i s2 = _mm256_maddubs_epi16(_mm256_loadu_si256(in + 2), _mm256_loadu_si256(pat + 2));
        const __m256i s3 = _mm256_maddubs_epi16(_mm256_loadu_si256(in + 3), _mm256_loadu_si256(pat + 3));
        const __m256i sum =
          _mm256_adds_epi16(_mm256_adds_epi16(s0, s1), _mm256_adds_epi16(s2, s3));
        const __m256i sum32 = _mm256_madd_epi16(sum, ones);
        int32_t       tmp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), sum32);
        proj[k] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }
#elif defined(USE_SSSE3)
    // Same unsigned*signed dot products as the AVX2 path, at 128 bits. Lane
    // sums hold at most 8 products (<= 2032), far inside i16.
    const __m128i zero = _mm_setzero_si128();
    const __m128i ones = _mm_set1_epi16(1);
    for (int k = 0; k < SketchBits; ++k)
    {
        const auto* pat = reinterpret_cast<const __m128i*>(SketchPatterns[k].data());
        const auto* in  = reinterpret_cast<const __m128i*>(stash);
        __m128i     acc = zero;
        for (int c = 0; c < FC2InputSize / 16; ++c)
            acc = _mm_adds_epi16(acc, _mm_maddubs_epi16(_mm_loadu_si128(in + c),
                                                        _mm_loadu_si128(pat + c)));
        acc = _mm_madd_epi16(acc, ones);
        int32_t tmp[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), acc);
        proj[k] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
#elif defined(USE_SSE2)
    const __m128i zero = _mm_setzero_si128();
    for (int k = 0; k < SketchBits; ++k)
    {
        const auto* pat = reinterpret_cast<const __m128i*>(SketchPatterns[k].data());
        const auto* in  = reinterpret_cast<const __m128i*>(stash);
        __m128i     acc = zero;
        for (int c = 0; c < FC2InputSize / 16; ++c)
        {
            const __m128i bytes = _mm_loadu_si128(in + c);
            const __m128i patt  = _mm_loadu_si128(pat + c);
            // unsigned widening of the non-negative stash, sign-preserving
            // widening of the +-1 patterns
            const __m128i sLo = _mm_unpacklo_epi8(bytes, zero);
            const __m128i sHi = _mm_unpackhi_epi8(bytes, zero);
            const __m128i sgn = _mm_cmpgt_epi8(zero, patt);
            const __m128i pLo = _mm_unpacklo_epi8(patt, sgn);
            const __m128i pHi = _mm_unpackhi_epi8(patt, sgn);
            acc = _mm_add_epi32(acc, _mm_madd_epi16(sLo, pLo));
            acc = _mm_add_epi32(acc, _mm_madd_epi16(sHi, pHi));
        }
        int32_t tmp[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), acc);
        proj[k] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
#else
    for (int k = 0; k < SketchBits; ++k)
    {
        i32 sum = 0;
        for (int i = 0; i < FC2InputSize; ++i)
            sum += i32(SketchPatterns[k][i]) * stash[i];
        proj[k] = sum;
    }
#endif
}

struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2;
    static constexpr int       FC_1_OUTPUTS                 = L3;

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS> fc_0;
    Layers::SqrClippedReLU<FC_0_OUTPUTS, WeightScaleBits + 1>                      ac_sqr_0;
    Layers::ClippedReLU<FC_0_OUTPUTS, WeightScaleBits + 1>                         ac_0;
    Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS>                        fc_1;
    Layers::SqrClippedReLU<FC_1_OUTPUTS, WeightScaleBits>                          ac_sqr_1;
    Layers::ClippedReLU<FC_1_OUTPUTS, WeightScaleBits>                             ac_1;
    Layers::AffineTransform<FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 1>                fc_2;

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() {
        // input slice hash
        u32 hashValue = 0xEC42E90Du;
        hashValue ^= TransformedFeatureDimensions * 2;

        hashValue = decltype(fc_0)::get_hash_value(hashValue);
        // TODO: consider including hash value of ac_sqr_0 in the overall hash value.
        // For now omitted on purpose because hash value is not written by trainer (yet)
        hashValue = decltype(ac_0)::get_hash_value(hashValue);
        hashValue = decltype(fc_1)::get_hash_value(hashValue);
        hashValue = decltype(ac_1)::get_hash_value(hashValue);
        hashValue = decltype(fc_2)::get_hash_value(hashValue);

        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        return fc_0.read_parameters(stream) && ac_0.read_parameters(stream)
            && fc_1.read_parameters(stream) && ac_1.read_parameters(stream)
            && fc_2.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        return fc_0.write_parameters(stream) && ac_0.write_parameters(stream)
            && fc_1.write_parameters(stream) && ac_1.write_parameters(stream)
            && fc_2.write_parameters(stream);
    }

    i32 propagate(const TransformedFeatureType* transformedFeatures,
                  const NNZInfo<L1>&            nnzInfo) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
            alignas(CacheLineSize) typename decltype(ac_sqr_0)::OutputType
              concat_buffer[ceil_to_multiple<IndexType>(FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 32)];
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;
        };

        Buffer buffer;

        fc_0.propagate(transformedFeatures, buffer.fc_0_out, nnzInfo);
#if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_0.propagate_pair(buffer.fc_0_out, buffer.concat_buffer,
                                buffer.concat_buffer + FC_0_OUTPUTS);
#else
        ac_sqr_0.propagate(buffer.fc_0_out, buffer.concat_buffer);
        ac_0.propagate(buffer.fc_0_out, buffer.concat_buffer + FC_0_OUTPUTS);
#endif

        fc_1.propagate(buffer.concat_buffer, buffer.fc_1_out);
#if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_1.propagate_pair(buffer.fc_1_out, buffer.concat_buffer + FC_0_OUTPUTS * 2,
                                buffer.concat_buffer + FC_0_OUTPUTS * 2 + FC_1_OUTPUTS);
#else
        ac_sqr_1.propagate(buffer.fc_1_out, buffer.concat_buffer + FC_0_OUTPUTS * 2);
        ac_1.propagate(buffer.fc_1_out, buffer.concat_buffer + FC_0_OUTPUTS * 2 + FC_1_OUTPUTS);
#endif

        {
            int8_t* stash = fc2_input_stash();
            for (int i = 0; i < FC2InputSize; ++i)
                stash[i] = buffer.concat_buffer[i];
        }

        fc_2.propagate(buffer.concat_buffer, buffer.fc_2_out);

        static_assert(FC_0_OUTPUTS >= 2);
        i32 fwdOut = buffer.fc_2_out[0];
        i32 skip_0 = buffer.fc_0_out[FC_0_OUTPUTS - 2] - buffer.fc_0_out[FC_0_OUTPUTS - 1];
        fwdOut += skip_0;

        // fwdOut is such that 1.0 is equal to HiddenOneVal*(1<<WeightScaleBits)*2 in
        // quantized form, but we want 1.0 to be equal to 600*OutputScale
        // to make overflow impossible we cast to int64_t
        constexpr i64 multiplier = 600 * OutputScale;
        constexpr i64 denominator =
          static_cast<i64>(HiddenOneVal) * static_cast<i64>(1U << WeightScaleBits) * 2;

        i32 outputValue = static_cast<i32>((static_cast<i64>(fwdOut) * multiplier) / denominator);
        return outputValue;
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, fc_0.get_content_hash());
        hash_combine(h, ac_sqr_0.get_content_hash());
        hash_combine(h, ac_0.get_content_hash());
        hash_combine(h, fc_1.get_content_hash());
        // hash_combine(h, ac_sqr_1.get_content_hash()); TODO
        hash_combine(h, ac_1.get_content_hash());
        hash_combine(h, fc_2.get_content_hash());
        hash_combine(h, get_hash_value());
        return h;
    }
};

}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::NetworkArchitecture> {
    Stockfish::usize
    operator()(const Stockfish::Eval::NNUE::NetworkArchitecture& arch) const noexcept {
        return arch.get_content_hash();
    }
};

#endif  // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
