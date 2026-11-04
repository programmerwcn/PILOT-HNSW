#pragma once
#include "dimension_policy.h"
#include <immintrin.h>
inline bool dimension_hard=false;
inline void fused_policy(const std::string& name){
    dimension_hard=name.rfind("hpca_",0)==0||name.rfind("hprune_",0)==0;
    dimension_policy(dimension_hard?name.substr(1):name);
}

// Fused original-coordinate distance: reuse all processed dimensions when the
// candidate survives. For uint8 all sums are exact integers (< 2^24 at D=128).
// Float summation differs from the reference SIMD kernel and is tested through
// returned recall/trajectory audits, not asserted to be bitwise identical.
template<int Block,class T>
inline float dimension_fused_impl(const T* a,const T* b,int d,float threshold,uint64_t& work,bool& rejected) {
    float partial=0;rejected=false;
    for(int offset=0;offset<d;offset+=Block) {
        float block=0;const int n=std::min(Block,d-offset);
        #pragma omp simd reduction(+:block)
        for(int j=0;j<n;++j){float z=float(a[offset+j])-float(b[offset+j]);block+=z*z;}
        partial+=block;work+=n;
        const float ratio=dimension_factor==0?1.f:std::min(1.f,dimension_factor*float(offset+n)/d);
        if(offset+n<d&&partial>threshold*ratio*1.00001f){rejected=true;return partial;}
    }
    return partial;
}
#if defined(__AVX2__)
template<int Block>
inline float dimension_fused_impl(const uint8_t* a,const uint8_t* b,int d,float threshold,uint64_t& work,bool& rejected) {
    if constexpr(Block<16)return dimension_fused_impl<Block,uint8_t>(a,b,d,threshold,work,rejected);
    unsigned partial=0;rejected=false;
    for(int offset=0;offset<d;offset+=Block){
        __m256i acc=_mm256_setzero_si256();const int n=std::min(Block,d-offset);
        int j=0;for(;j+16<=n;j+=16){
            __m256i x=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+offset+j)));
            __m256i y=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b+offset+j)));
            __m256i z=_mm256_sub_epi16(x,y);acc=_mm256_add_epi32(acc,_mm256_madd_epi16(z,z));
        }
        __m128i sum=_mm_add_epi32(_mm256_castsi256_si128(acc),_mm256_extracti128_si256(acc,1));
        sum=_mm_hadd_epi32(sum,sum);sum=_mm_hadd_epi32(sum,sum);partial+=_mm_cvtsi128_si32(sum);
        for(;j<n;++j){int z=int(a[offset+j])-int(b[offset+j]);partial+=z*z;}
        work+=n;const float ratio=dimension_factor==0?1.f:std::min(1.f,dimension_factor*float(offset+n)/d);
        if(offset+n<d&&float(partial)>threshold*ratio*1.00001f){rejected=true;return partial;}
    }
    return partial;
}
#endif
template<class T>
inline float dimension_fused(const T* a,const T* b,int d,float threshold,uint64_t& work,bool& rejected) {
    switch(dimension_block){
        case 8:return dimension_fused_impl<8>(a,b,d,threshold,work,rejected);
        case 16:return dimension_fused_impl<16>(a,b,d,threshold,work,rejected);
        case 32:return dimension_fused_impl<32>(a,b,d,threshold,work,rejected);
        case 64:return dimension_fused_impl<64>(a,b,d,threshold,work,rejected);
        case 128:return dimension_fused_impl<128>(a,b,d,threshold,work,rejected);
        default:throw std::runtime_error("unsupported fused block size");
    }
}
