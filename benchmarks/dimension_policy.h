#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <stdexcept>

// Names: pca_DIM_MARGIN, prune_BLOCK_FACTOR. Factors act on squared distances.
// prune factor 0 uses the nonnegative partial-sum bound (with FP slack).
// Positive factors are empirical approximation policies, NOT ADSampling bounds.
inline int dimension_kind=0, dimension_p=0, dimension_block=0;
inline float dimension_margin=1, dimension_factor=0;
inline void dimension_policy(const std::string& name) {
    dimension_kind=0;dimension_p=0;dimension_margin=1;
    if(name.rfind("pca_",0)==0 || name.rfind("prune_",0)==0) {
        std::string s=name;std::replace(s.begin(),s.end(),'_',' ');
        std::istringstream in(s);std::string kind;int n;float v;
        if(!(in>>kind>>n>>v)||n<=0||v<0)throw std::runtime_error("invalid dimension policy");
        if(kind=="pca"){dimension_kind=1;dimension_p=n;dimension_margin=v;}
        else {dimension_kind=2;dimension_block=n;dimension_factor=v;}
    }
}
template<class T>
inline bool dimension_reject(const T* a,const T* b,int d,float threshold,uint64_t& work) {
    float partial=0;
    for(int offset=0;offset+dimension_block<d;offset+=dimension_block) {
        float block=0;
        #pragma omp simd reduction(+:block)
        for(int j=0;j<dimension_block;++j){float z=float(a[offset+j])-float(b[offset+j]);block+=z*z;}
        partial+=block;work+=dimension_block;
        const float ratio=dimension_factor==0?1.f:std::min(1.f,dimension_factor*float(offset+dimension_block)/d);
        if(partial>threshold*ratio*1.00001f)return true;
    }
    // Survivors use the original exact kernel, preserving its rounding and order.
    // This deliberately charges duplicate work rather than understating NDC.
    return false;
}
