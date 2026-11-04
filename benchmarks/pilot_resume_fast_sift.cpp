#include "pilot_feature_math.h"
#include "pilot_probe_state.h"

#include <string>
#include <cmath>
static double target_stop=.9, target_alpha=1, target_scale=1;
static int target_ef=20000;
static unsigned long long target_hierarchy=0;
static std::string target_arm(const std::string& label){
 auto p=label.find(':');auto method=label.substr(0,p);
 double value=p==std::string::npos?1:std::stod(label.substr(p+1));
 target_scale=1;target_stop=.9;target_alpha=1;target_ef=20000;
 if(method=="plain")target_ef=int(value);
 else if(method=="darth")target_stop=value;
 else if(method=="laet")target_alpha=value;
 else target_scale=value;
 auto at=label.find('@');if(at!=std::string::npos)target_ef=std::stoi(label.substr(at+1));
 return method;
}
static int target_cap(int ef){return std::max(100,int(std::ceil(ef*target_scale)));}
#include "dimension_fused_policy.h"
#include "plateau_gate.h"
// Native same-graph SIFT100M evaluator for fixed/plain and difficult-only PCA-NAV.
// The serving predictor is deliberately supplied as held-out tier assignments;
// zero-anchor probe feature collection/training is layered on this native kernel.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <immintrin.h>
#include <omp.h>

#include <fcntl.h>
#include <jsoncpp/json/json.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"

extern "C" {
typedef void* BoosterHandle;
int LGBM_BoosterCreateFromModelfile(const char*, int*, BoosterHandle*);
int LGBM_BoosterPredictForMat(BoosterHandle, const void*, int, int32_t, int32_t,
                              int, int, int, int, const char*, int64_t*, double*);
int LGBM_BoosterFree(BoosterHandle);
const char* LGBM_GetLastError();
}
constexpr int LGBM_FLOAT32=0,LGBM_PREDICT_NORMAL=0;

using hnswlib::HierarchicalNSW;
using hnswlib::L2SpaceU8;
using hnswlib::tableint;

namespace {
using Clock = std::chrono::steady_clock;
using Pair = std::pair<float, uint32_t>;

bool residualOnlyPca() {
    static const bool enabled = std::getenv("PCA_MODE") &&
        std::string(std::getenv("PCA_MODE")) == "residual";
    return enabled;
}

bool colocatedPca() {
    static const bool enabled = std::getenv("PCA_MODE") &&
        std::string(std::getenv("PCA_MODE")) == "colocated";
    return enabled;
}

int pcaPrefetchDistance() {
    static const int distance = [] {
        const char* value = std::getenv("PCA_PREFETCH_DISTANCE");
        return value ? std::max(0, std::atoi(value)) : 0;
    }();
    return distance;
}

float pcaParentRatio() {
    static const float ratio = [] {
        const char* value=std::getenv("PCA_PARENT_RATIO");
        return value?std::max(0.0f,std::strtof(value,nullptr)):0.0f;
    }();
    return ratio;
}

uint64_t pcaMinExpanded() {
    static const uint64_t value=[] {const char*v=std::getenv("PCA_MIN_EXPANDED");return v?std::strtoull(v,nullptr,10):0ULL;}();
    return value;
}

class MappedFile {
public:
    explicit MappedFile(const std::string& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st {};
        if (::fstat(fd_, &st) || st.st_size <= 0) throw std::runtime_error("cannot stat " + path);
        size_ = static_cast<size_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
        data_ = static_cast<const uint8_t*>(p);
#ifdef MADV_RANDOM
        ::madvise(const_cast<uint8_t*>(data_), size_, MADV_RANDOM);
#endif
    }
    ~MappedFile() {
        if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
    }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    uint64_t prewarm() const {
#ifdef MADV_NORMAL
        ::madvise(const_cast<uint8_t*>(data_), size_, MADV_NORMAL);
#endif
        volatile uint64_t checksum=0; constexpr size_t page=4096;
        for(size_t i=0;i<size_;i+=page)checksum+=data_[i];
        checksum+=data_[size_-1];
#ifdef MADV_RANDOM
        ::madvise(const_cast<uint8_t*>(data_), size_, MADV_RANDOM);
#endif
        return checksum;
    }
private:
    std::string path_;
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

template<class T> T readPod(std::istream& in) {
    T v {};
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("short PCA model");
    return v;
}

struct PcaModel {
    uint64_t n = 0;
    uint32_t dim = 0, dpca = 0;
    std::vector<float> mu, projection, quant_scale;
    explicit PcaModel(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open " + path);
        char magic[8]; in.read(magic, 8);
        if (std::memcmp(magic, "SPCA100\0", 8)) throw std::runtime_error("bad PCA model magic");
        n = readPod<uint64_t>(in); dim = readPod<uint32_t>(in); dpca = readPod<uint32_t>(in);
        mu.resize(dim); projection.resize(size_t(dim) * dpca);
        in.read(reinterpret_cast<char*>(mu.data()), mu.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(projection.data()), projection.size() * sizeof(float));
        if (!in) throw std::runtime_error("short PCA arrays");
        quant_scale.resize(dpca);
        in.read(reinterpret_cast<char*>(quant_scale.data()), quant_scale.size()*sizeof(float));
        if(!in){in.clear();quant_scale.clear();}
    }
};

struct PcaTable {
    PcaTable(const std::string& scores_path, const std::string& residual_path, const PcaModel& model)
        : scores_file(scores_path), residual_file(residual_path), m(model) {
        quantized=!m.quant_scale.empty();
        stored_dpca=m.dpca;
        const size_t expected=m.n*m.dpca*(quantized?sizeof(int8_t):sizeof(float));
        if(quantized&&scores_file.size()%m.n==0){const size_t row_bytes=scores_file.size()/m.n;
            if(row_bytes>=5&&row_bytes-4<=m.dpca){interleaved=true;stored_dpca=uint32_t(row_bytes-4);}}
        if (scores_file.size() != expected && !interleaved) throw std::runtime_error("PCA score size mismatch");
        if (residual_file.size() != m.n * sizeof(float))
            throw std::runtime_error("PCA residual size mismatch");
        scores_f32 = quantized?nullptr:reinterpret_cast<const float*>(scores_file.data());
        scores_i8 = quantized?reinterpret_cast<const int8_t*>(scores_file.data()):nullptr;
        residual = reinterpret_cast<const float*>(residual_file.data());
    }
    const float* rowF32(uint32_t node) const { return scores_f32 + uint64_t(node) * m.dpca; }
    const int8_t* rowI8(uint32_t node) const { return interleaved?scores_i8+uint64_t(node)*(stored_dpca+4)+4:scores_i8+uint64_t(node)*stored_dpca; }
    const void* rowAddress(uint32_t node) const {
        return interleaved ? static_cast<const void*>(scores_i8 + uint64_t(node)*(stored_dpca+4))
                           : static_cast<const void*>(scores_i8 + uint64_t(node)*stored_dpca);
    }
    float residualAt(uint32_t node)const{
        // Residual-only serving deliberately uses the compact 400 MB array:
        // its much smaller working set has better cache/TLB locality than the
        // 3.6 GB interleaved code table.
        if(!interleaved || residualOnlyPca())return residual[node];
        float value;std::memcpy(&value,scores_i8+uint64_t(node)*(stored_dpca+4),4);return value;
    }
    uint64_t prewarm() const { return scores_file.prewarm() ^ residual_file.prewarm(); }
    MappedFile scores_file, residual_file;
    const PcaModel& m;
    bool quantized=false,interleaved=false;
    uint32_t stored_dpca=0;
    const float* scores_f32 = nullptr;
    const int8_t* scores_i8 = nullptr;
    const float* residual = nullptr;
};

struct ThreadState {
    explicit ThreadState(size_t n) : seen(n, 0) {}
    uint32_t next() {
        if (++epoch == 0) { std::fill(seen.begin(), seen.end(), 0); epoch = 1; }
        return epoch;
    }
    std::vector<uint32_t> seen;
    uint32_t epoch = 0;
};

struct TierModel {
    BoosterHandle handle=nullptr; int feature_count=0;
    TierModel(const std::string&path,int features):feature_count(features){int iterations=0;
        if(LGBM_BoosterCreateFromModelfile(path.c_str(),&iterations,&handle))throw std::runtime_error(LGBM_GetLastError());}
    ~TierModel(){if(handle)LGBM_BoosterFree(handle);}
    int predict(const std::vector<float>&features)const{
        if(int(features.size())!=feature_count)throw std::runtime_error("predictor feature count mismatch");
        int64_t nout=0;double scores[3]={};
        if(LGBM_BoosterPredictForMat(handle,features.data(),LGBM_FLOAT32,1,feature_count,1,
             LGBM_PREDICT_NORMAL,0,-1,"num_threads=1",&nout,scores))throw std::runtime_error(LGBM_GetLastError());
        if(nout!=3)throw std::runtime_error("predictor is not three-class");
        return int(std::max_element(scores,scores+3)-scores);
    }
};

struct ScalarModel {
    BoosterHandle handle=nullptr; int feature_count=0;
    ScalarModel(const std::string&path,int features):feature_count(features){int iterations=0;
        if(LGBM_BoosterCreateFromModelfile(path.c_str(),&iterations,&handle))throw std::runtime_error(LGBM_GetLastError());}
    ~ScalarModel(){if(handle)LGBM_BoosterFree(handle);}
    float predict(const std::vector<float>&features)const{
        if(int(features.size())!=feature_count)throw std::runtime_error("scalar predictor feature count mismatch");
        int64_t nout=0;double score=0;
        if(LGBM_BoosterPredictForMat(handle,features.data(),LGBM_FLOAT32,1,feature_count,1,
             LGBM_PREDICT_NORMAL,0,-1,"num_threads=1",&nout,&score))throw std::runtime_error(LGBM_GetLastError());
        if(nout!=1)throw std::runtime_error("scalar predictor returned multiple outputs");return float(score);
    }
};

struct Result {
    uint64_t trigger_exact=0,trigger_step=0;double recall = 0, ndc = 0, seconds = 0;
    uint64_t exact = 0, estimates = 0, verifies = 0, rejects = 0;
    uint64_t visited = 0, expanded = 0, pushes = 0, pops = 0, pca_dims = 0;
    uint64_t projection_ns = 0, pca_ns = 0, exact_ns = 0, heap_ns = 0, graph_ns = 0;
    uint64_t probe_ns = 0, feature_ns = 0, predictor_ns = 0, continuation_ns = 0;
    uint64_t predictor_calls = 0;
    int ef = 0, tier = -1;
};

uint64_t nsSince(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

bool detailedKernelTiming() {
    static const bool enabled = std::getenv("DETAILED_KERNEL_TIMING") &&
        std::string(std::getenv("DETAILED_KERNEL_TIMING")) == "1";
    return enabled;
}

// The SIFT100M serving model has exactly 32 quantized PCA coordinates.  Evaluate
// the high-variance half first: because every term is non-negative, exceeding
// the admission threshold after 16 dimensions is an exact early-out with
// respect to the existing quantized bound.
inline float pcaDistanceI8x32(const int8_t* code, const float* scale,
                             const float* query, float base, float threshold,
                             uint32_t& evaluated_dims) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m128i bytes0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code));
    const __m512 x0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes0));
    const __m512 d0 = _mm512_sub_ps(_mm512_mul_ps(x0, _mm512_loadu_ps(scale)),
                                    _mm512_loadu_ps(query));
    float sum = base + _mm512_reduce_add_ps(_mm512_mul_ps(d0, d0));
    evaluated_dims = 16;
    if (sum >= threshold) return sum;
    const __m128i bytes1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + 16));
    const __m512 x1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes1));
    const __m512 d1 = _mm512_sub_ps(_mm512_mul_ps(x1, _mm512_loadu_ps(scale + 16)),
                                    _mm512_loadu_ps(query + 16));
    evaluated_dims = 32;
    return sum + _mm512_reduce_add_ps(_mm512_mul_ps(d1, d1));
#else
    float sum = base;
    for (uint32_t c = 0; c < 16; ++c) {
        const float d = float(code[c]) * scale[c] - query[c]; sum += d*d;
    }
    evaluated_dims = 16;
    if (sum >= threshold) return sum;
    for (uint32_t c = 16; c < 32; ++c) {
        const float d = float(code[c]) * scale[c] - query[c]; sum += d*d;
    }
    evaluated_dims = 32;
    return sum;
#endif
}

inline float pcaDistanceI8x16(const int8_t* code, const float* scale,
                             const float* query, float base) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512 x = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(code))));
    const __m512 d = _mm512_sub_ps(_mm512_mul_ps(x,_mm512_loadu_ps(scale)),
                                   _mm512_loadu_ps(query));
    return base+_mm512_reduce_add_ps(_mm512_mul_ps(d,d));
#else
    for(uint32_t c=0;c<16;++c){const float d=float(code[c])*scale[c]-query[c];base+=d*d;}return base;
#endif
}

inline float pcaDistanceI8x8(const int8_t* code,const float*scale,const float*query,float base){
#if defined(__AVX2__)
    const __m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(code));
    const __m256 x=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
    const __m256 d=_mm256_fmsub_ps(x,_mm256_loadu_ps(scale),_mm256_loadu_ps(query));
    const __m256 sq=_mm256_mul_ps(d,d);__m128 lo=_mm256_castps256_ps128(sq),hi=_mm256_extractf128_ps(sq,1);
    __m128 s=_mm_add_ps(lo,hi);s=_mm_add_ps(s,_mm_movehl_ps(s,s));
    s=_mm_add_ss(s,_mm_shuffle_ps(s,s,1));return base+_mm_cvtss_f32(s);
#else
    for(uint32_t c=0;c<8;++c){const float d=float(code[c])*scale[c]-query[c];base+=d*d;}return base;
#endif
}

float exactDistance(const HierarchicalNSW<int>& index, const uint8_t* q, uint32_t node) {
    return static_cast<float>(index.fstdistfunc_(q, index.getDataByInternalId(node), index.dist_func_param_));
}

class CalibrationLogger {
public:
    CalibrationLogger(){const char*path=std::getenv("PCA_CALIBRATION_LOG");if(path){out.open(path);if(!out)throw std::runtime_error("cannot open PCA_CALIBRATION_LOG");
        out<<"estimate_ratio,exact_ratio,node_residual_ratio,parent_ratio,projected_fraction,lower,false_reject\n";}}
    bool enabled()const{return out.is_open();}
    void record(float estimate,float exact,float threshold,float node_residual,float query_residual,
                float parent,float lower){
        const float residual_sum=node_residual+query_residual;
        out<<estimate/std::max(threshold,1e-6f)<<','<<exact/std::max(lower,1e-6f)<<','
           <<node_residual/std::max(residual_sum,1e-6f)<<','<<parent/std::max(lower,1e-6f)<<','
           <<std::max(estimate-residual_sum,0.0f)/std::max(estimate,1e-6f)<<','<<lower<<','<<(exact<lower)<<'\n';
    }
private:std::ofstream out;
};

CalibrationLogger& calibrationLogger(){static CalibrationLogger logger;return logger;}

void projectQuery(const PcaTable&pca,const uint8_t*q,std::vector<float>&qp,float&query_residual){
    qp.assign(pca.m.dpca,0);float centered_norm=0;
    for(uint32_t j=0;j<pca.m.dim;++j){const float z=float(q[j])-pca.m.mu[j];centered_norm+=z*z;
        const float*pj=pca.m.projection.data()+size_t(j)*pca.m.dpca;
        for(uint32_t c=0;c<pca.m.dpca;++c)qp[c]+=z*pj[c];}
    float projected_norm=0;for(float v:qp)projected_norm+=v*v;
    query_residual=std::max(centered_norm-projected_norm,0.0f);
}

float lidEstimate(const std::vector<float>&d,size_t count){count=std::min(count,d.size());if(count<2)return 0;
    const float rk=std::max(d[count-1],1e-6f);double sum=0;for(size_t i=0;i<count;++i)sum+=std::log(std::max(d[i],1e-6f)/rk);
    return sum<-1e-9?float(-double(count)/sum):0;}

std::vector<float> zeroAnchorFeatures(const HierarchicalNSW<int>&index,const uint8_t*q,
        std::priority_queue<Pair>top,
        std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>>cand,
        uint64_t ndis,uint64_t ninserts,float first,size_t expanded,size_t last_best_step,
        uint64_t checks,uint64_t new_nodes,const std::vector<float>&qp){
    std::vector<Pair>nearest;while(!top.empty()){nearest.push_back(top.top());top.pop();}std::sort(nearest.begin(),nearest.end());
    std::vector<float>dists;for(auto[d,id]:nearest)dists.push_back(std::sqrt(std::max(d,0.0f)));
    while(dists.size()<10)dists.push_back(dists.empty()?0:dists.back());
    std::vector<float>frontier;while(!cand.empty()&&frontier.size()<100){frontier.push_back(std::sqrt(std::max(cand.top().first,0.0f)));cand.pop();}
    std::sort(frontier.begin(),frontier.end());if(frontier.empty())frontier=dists;
    double avg=std::accumulate(dists.begin(),dists.begin()+10,0.0)/10,var=0;for(int i=0;i<10;++i)var+=(dists[i]-avg)*(dists[i]-avg);var/=10;
    const float med=(dists[4]+dists[5])*.5f,contrast=(frontier[frontier.size()/2]-med)/std::max(med,1e-6f);
    float cone=pilotConeCached(q,128,nearest,10,[&](unsigned id){return reinterpret_cast<const uint8_t*>(index.getDataByInternalId(id));});
    std::array<uint32_t,10>ids10{};for(int i=0;i<10;++i)ids10[i]=nearest[i].second;size_t directed=0;
    for(int a=0;a<10;++a){auto*l=reinterpret_cast<unsigned*>(index.get_linklist0(ids10[a]));const uint32_t n=index.getListCount(l);auto*ids=reinterpret_cast<tableint*>(l+1);
        for(uint32_t z=0;z<n;++z)for(int b=0;b<10;++b)if(a!=b&&ids[z]==ids10[b])++directed;}
    const float mutual=float(directed)/90.0f,novelty=checks?float(new_nodes)/checks:0,detour=expanded?float(last_best_step)/expanded:0;
    const float lid_diff=lidEstimate(dists,std::min<size_t>(100,dists.size()))-lidEstimate(frontier,std::min<size_t>(100,frontier.size()));
    std::vector<float>f={contrast,cone,mutual,novelty,dists[0],dists[9],float(avg),float(var),(dists[4]+dists[5])*.5f,dists[2],dists[7],
        float(ndis),float(ninserts),std::sqrt(std::max(first,0.0f)),detour,lid_diff};
    // Skip the eight inert compatibility fields: serving uses clean51.
    double sum=0,sq=0;float mn=255;for(uint32_t j=0;j<128;++j){sum+=q[j];sq+=double(q[j])*q[j];mn=std::min(mn,float(q[j]));}
    const double mean=sum/128;f.push_back(float(std::sqrt(std::max(0.0,sq/128-mean*mean))));f.push_back(float(std::sqrt(sq)));f.push_back(mn);
    for(uint32_t c=0;c<std::min<uint32_t>(32,qp.size());++c)f.push_back(qp[c]);
    if(f.size()!=51)throw std::runtime_error("clean zero-anchor feature count mismatch");return f;
}


uint32_t measuredEntry(const HierarchicalNSW<int>& index,const uint8_t*q,uint32_t saved){
    target_hierarchy=1;tableint cur=index.enterpoint_node_;int distance=exactDistance(index,q,cur);
    for(int level=index.maxlevel_;level>0;--level){bool changed=true;while(changed){changed=false;
        auto*links=reinterpret_cast<unsigned*>(index.get_linklist(cur,level));const int n=index.getListCount(links);
        auto*ids=reinterpret_cast<tableint*>(links+1);
        for(int j=0;j<n;++j){++target_hierarchy;int d=exactDistance(index,q,ids[j]);if(d<distance){distance=d;cur=ids[j];changed=true;}}}}
    if(cur!=saved)throw std::runtime_error("hierarchical entry mismatch");return cur;
}

Result searchOne(const HierarchicalNSW<int>& index, const PcaTable& pca, const uint8_t* q,
                 const unsigned* gt, size_t k, uint32_t entry, int ef, bool compressed,
                 float admission_margin, ThreadState& state) {
    Result out; out.ef = ef;
    const auto wall0 = Clock::now(); entry=measuredEntry(index,q,entry);
    const bool profile_kernels=detailedKernelTiming();
    std::priority_queue<Pair> top;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> candidates;
    const uint32_t epoch = state.next();

    std::vector<float> qp;
    float query_residual = 0;
    if (compressed) {
        const auto t0 = Clock::now();
        qp.assign(pca.m.dpca, 0.0f);
        float centered_norm = 0;
        for (uint32_t j = 0; j < pca.m.dim; ++j) {
            const float z = float(q[j]) - pca.m.mu[j];
            centered_norm += z * z;
            const float* pj = pca.m.projection.data() + size_t(j) * pca.m.dpca;
            for (uint32_t c = 0; c < pca.m.dpca; ++c) qp[c] += z * pj[c];
        }
        float projected_norm = 0;
        for (float v : qp) projected_norm += v * v;
        query_residual = std::max(centered_norm - projected_norm, 0.0f);
        out.projection_ns = nsSince(t0);
    }

    auto t0 = Clock::now();
    float lower = exactDistance(index, q, entry);
    if(profile_kernels)out.exact_ns += nsSince(t0); ++out.exact;
    if(profile_kernels)t0 = Clock::now();
    top.emplace(lower, entry); candidates.emplace(lower, entry);
    if(profile_kernels)out.heap_ns += nsSince(t0); ++out.pushes;
    state.seen[entry] = epoch; out.visited = 1;

    while (!candidates.empty()) {
        if(profile_kernels)t0 = Clock::now();
        const auto [current_dist, current] = candidates.top();
        if (top.size() >= size_t(ef) && current_dist > lower) {
            if(profile_kernels)out.heap_ns += nsSince(t0); break;
        }
        candidates.pop(); if(profile_kernels)out.heap_ns += nsSince(t0); ++out.pops; ++out.expanded;

        if(profile_kernels)t0 = Clock::now();
        auto* links = reinterpret_cast<unsigned int*>(index.get_linklist0(current));
        const uint32_t count = index.getListCount(links);
        const tableint* ids = reinterpret_cast<tableint*>(links + 1);
        if(profile_kernels)out.graph_ns += nsSince(t0);
        for (uint32_t ni = 0; ni < count; ++ni) {
            const uint32_t node = ids[ni];
            if(profile_kernels)t0 = Clock::now();
            if (state.seen[node] == epoch) { if(profile_kernels)out.graph_ns += nsSince(t0); continue; }
            state.seen[node] = epoch; ++out.visited;
            if(profile_kernels)out.graph_ns += nsSince(t0);

            if (compressed && top.size() >= size_t(ef) && out.expanded >= pcaMinExpanded()) {
                if(profile_kernels)t0 = Clock::now();
                float estimate = pca.residualAt(node) + query_residual;
                const float threshold=lower*admission_margin*admission_margin;bool rejected=estimate>=threshold;
                if(!rejected&&pca.quantized){const int8_t*bx=pca.rowI8(node);
#pragma omp simd reduction(+:estimate)
                    for(uint32_t c=0;c<pca.m.dpca;++c){const float v=float(bx[c])*pca.m.quant_scale[c]-qp[c];estimate+=v*v;}out.pca_dims+=pca.m.dpca;rejected=estimate>=threshold;}
                else if(!rejected){const float*bx=pca.rowF32(node);
#pragma omp simd reduction(+:estimate)
                    for(uint32_t c=0;c<pca.m.dpca;++c){const float v=bx[c]-qp[c];estimate+=v*v;}out.pca_dims+=pca.m.dpca;rejected=estimate>=threshold;}
                if(profile_kernels)out.pca_ns += nsSince(t0); ++out.estimates;
                if (rejected) { ++out.rejects; continue; }
            }

            if(profile_kernels)t0 = Clock::now();
            const float distance = exactDistance(index, q, node);
            if(profile_kernels)out.exact_ns += nsSince(t0); ++out.exact;
            if (compressed) ++out.verifies;
            if (top.size() < size_t(ef) || distance < lower) {
                if(profile_kernels)t0 = Clock::now();
                candidates.emplace(distance, node); top.emplace(distance, node); ++out.pushes;
                if (top.size() > size_t(ef)) top.pop();
                lower = top.top().first;
                if(profile_kernels)out.heap_ns += nsSince(t0);
            }
        }
    }

    while (top.size() > k) top.pop();
    const auto serving_end=Clock::now();
    std::unordered_set<unsigned> truth(gt, gt + k);
    size_t hits = 0;
    while (!top.empty()) { hits += truth.count(top.top().second); top.pop(); }
    out.recall = double(hits) / k;
    out.ndc = target_hierarchy + double(out.exact) + double(out.pca_dims) / pca.m.dim;
    out.seconds = std::chrono::duration<double>(serving_end - wall0).count();
    return out;
}

Result searchAdaptive(const HierarchicalNSW<int>&index,const PcaTable&pca,const TierModel&model,
        const uint8_t*q,const unsigned*gt,size_t k,uint32_t entry,const std::vector<int>&caps,
        bool pca_hard,float admission_margin,ThreadState&state){
    PilotProbeState resume;Result out;const auto wall0=Clock::now();PlateauGate gate(k);entry=measuredEntry(index,q,entry);const uint32_t epoch=state.next();
    const bool use_colocated=colocatedPca(),residual_only=residualOnlyPca(),profile_kernels=detailedKernelTiming();
    const bool calibration_enabled=std::getenv("PCA_CALIBRATION_LOG")!=nullptr;
    const int prefetch_distance=pcaPrefetchDistance();const float parent_ratio=pcaParentRatio();
    const uint64_t min_expanded=pcaMinExpanded();const float margin_squared=admission_margin*admission_margin;
    std::priority_queue<Pair>top;std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>>cand,deferred;
    std::vector<float>qp;float query_residual=0;auto t0=Clock::now();projectQuery(pca,q,qp,query_residual);out.projection_ns=nsSince(t0);
    t0=Clock::now();const float first=exactDistance(index,q,entry);out.exact_ns+=nsSince(t0);++out.exact;
    gate.observe(first,true,first);top.emplace(first,entry);cand.emplace(first,entry);state.seen[entry]=epoch;out.visited=1;out.pushes=1;resume.record(first,entry);
    uint64_t checks=0,new_nodes=1,ninserts=1;size_t last_best_step=0;float best=first;
    const auto probe0=Clock::now();
    while(!cand.empty()&&out.expanded<100){const auto[current_dist,current]=cand.top();cand.pop();++out.pops;++out.expanded;resume.expand(current);
        auto*links=reinterpret_cast<unsigned*>(index.get_linklist0(current));const uint32_t count=index.getListCount(links);auto*ids=reinterpret_cast<tableint*>(links+1);
        for(uint32_t ni=0;ni<count;++ni){++checks;const uint32_t node=ids[ni];if(state.seen[node]==epoch)continue;state.seen[node]=epoch;++out.visited;++new_nodes;
            const float d=exactDistance(index,q,node);++out.exact;resume.record(d,node);gate.observe(d,top.size()<200||d<top.top().first,top.top().first);if(top.size()<200||d<top.top().first){cand.emplace(d,node);top.emplace(d,node);++out.pushes;++ninserts;
                if(top.size()>200){top.pop();}if(d<best){best=d;last_best_step=out.expanded;}}
            }}
    out.probe_ns=nsSince(probe0);
    t0=Clock::now();const auto features=zeroAnchorFeatures(index,q,top,cand,out.exact,ninserts,first,out.expanded,last_best_step,checks,new_nodes,qp);out.feature_ns=nsSince(t0);
    t0=Clock::now();const int tier=model.predict(features);out.predictor_ns=nsSince(t0);out.predictor_calls=1;
    out.tier=tier;out.ef=caps.at(tier);const size_t ef=out.ef;
    // The common routing probe keeps up to 200 candidates.  Learned ef may be
    // below 200 (e.g. 100/114 under min-ef tiering), so restore the requested
    // result-heap capacity before applying the continuation termination rule.
    resume.restore(ef,top,cand);
    const bool use_pca=(plateau_mode>0&&(!plateau_hard_only||tier==2))||(plateau_mode<0&&pca_hard&&tier==2);
    const bool fast_pca8=use_pca&&use_colocated&&pca.quantized&&!residual_only&&pca.stored_dpca==8;
    const ptrdiff_t pca_record_bytes=ptrdiff_t(pca.stored_dpca+sizeof(float));
    // All evaluated unexpanded candidates were restored without distance recomputation.
    float dimension_qr=query_residual;if(dimension_kind==1){if(dimension_p>int(pca.stored_dpca))throw std::runtime_error("SIFT prefix unavailable");for(unsigned c=dimension_p;c<qp.size();++c)dimension_qr+=qp[c]*qp[c];}float lower=top.top().first;const auto continuation0=Clock::now();
    while(!cand.empty()){const auto[current_dist,current]=cand.top();if(top.size()>=ef&&current_dist>lower)break;cand.pop();++out.pops;++out.expanded;
        auto*links=reinterpret_cast<unsigned*>(index.get_linklist0(current));const uint32_t count=index.getListCount(links);auto*ids=reinterpret_cast<tableint*>(links+1);
        for(uint32_t ni=0;ni<count;++ni){
            if(use_pca&&prefetch_distance>0&&ni+uint32_t(prefetch_distance)<count) {
                const uint32_t future=ids[ni+prefetch_distance];
                const char* address=use_colocated
                    ? reinterpret_cast<const char*>(index.getDataByInternalId(future))-ptrdiff_t(pca.stored_dpca+sizeof(float))
                    : reinterpret_cast<const char*>(pca.rowAddress(future));
                _mm_prefetch(address,_MM_HINT_T0);
            }
            const uint32_t node=ids[ni];if(state.seen[node]==epoch)continue;state.seen[node]=epoch;++out.visited;

            bool fused=dimension_kind==2&&(!dimension_hard||tier==2)&&top.size()>=ef;float fused_value=0;if(fused){bool rejected;fused_value=dimension_fused(q,reinterpret_cast<const uint8_t*>(index.getDataByInternalId(node)),128,lower,out.pca_dims,rejected);if(rejected){++out.rejects;continue;}}
            if(dimension_kind==1&&(!dimension_hard||tier==2)&&top.size()>=ef){
                const int8_t* code=pca.rowI8(node);float estimate=pca.residualAt(node)+dimension_qr;
                // Stored32 residual + decoded discarded energy. Quantization
                // makes this an empirical score; no lower-bound claim is made.
                for(unsigned c=0;c<pca.stored_dpca;++c){float z=float(code[c])*pca.m.quant_scale[c];
                    if(c<unsigned(dimension_p))z-=qp[c];estimate+=z*z;}
                ++out.estimates;out.pca_dims+=pca.stored_dpca;
                if(estimate>=lower*dimension_margin*dimension_margin){++out.rejects;continue;}
            }
            if(dimension_kind==0&&((plateau_mode<0&&use_pca&&top.size()>=ef&&out.expanded>=min_expanded&&current_dist>=lower*parent_ratio)||(plateau_mode>0&&use_pca&&gate.check(top.size()>=ef,lower,out.exact,out.expanded)))){
                const int8_t* colocated=use_colocated
                    ? reinterpret_cast<const int8_t*>(index.getDataByInternalId(node))-pca_record_bytes : nullptr;
                float node_residual=0;
                if(colocated)std::memcpy(&node_residual,colocated,sizeof(float));else node_residual=pca.residualAt(node);
                float estimate=node_residual+query_residual;
                const float threshold=lower*margin_squared;bool rejected=estimate>=threshold;
                if(profile_kernels)t0=Clock::now();
                if(!rejected&&fast_pca8){estimate=pcaDistanceI8x8(colocated+4,pca.m.quant_scale.data(),qp.data(),estimate);
                    out.pca_dims+=8;rejected=estimate>=threshold;}
                else if(!rejected&&residual_only) {
                    // Let the exact uint8 kernel handle residual survivors.
                }
                else if(!rejected&&pca.quantized){const int8_t*bi=colocated?colocated+4:pca.rowI8(node);uint32_t dims=0;
                    if(pca.stored_dpca==32)estimate=pcaDistanceI8x32(bi,pca.m.quant_scale.data(),qp.data(),estimate,threshold,dims);
                    else if(pca.stored_dpca==16){estimate=pcaDistanceI8x16(bi,pca.m.quant_scale.data(),qp.data(),estimate);dims=16;}
                    else if(pca.stored_dpca==8){estimate=pcaDistanceI8x8(bi,pca.m.quant_scale.data(),qp.data(),estimate);dims=8;}
                    else {
#pragma omp simd reduction(+:estimate)
                        for(uint32_t c=0;c<pca.stored_dpca;++c){const float v=float(bi[c])*pca.m.quant_scale[c]-qp[c];estimate+=v*v;}
                        dims=pca.stored_dpca;
                    }
                    out.pca_dims+=dims;rejected=estimate>=threshold;}
                else if(!rejected){const float*bf=pca.rowF32(node);
#pragma omp simd reduction(+:estimate)
                    for(uint32_t c=0;c<pca.m.dpca;++c){const float v=bf[c]-qp[c];estimate+=v*v;}out.pca_dims+=pca.m.dpca;rejected=estimate>=threshold;}
                if(profile_kernels)out.pca_ns+=nsSince(t0);++out.estimates;if(rejected){++out.rejects;
                    if(calibration_enabled){const float shadow=exactDistance(index,q,node);
                        calibrationLogger().record(estimate,shadow,threshold,node_residual,query_residual,current_dist,lower);}
                    continue;}}
            if(profile_kernels)t0=Clock::now();const float d=fused?fused_value:exactDistance(index,q,node);if(profile_kernels)out.exact_ns+=nsSince(t0);if(!fused)++out.exact;if(use_pca)++out.verifies;
            gate.observe(d,top.size()<ef||d<lower,lower);if(top.size()<ef||d<lower){cand.emplace(d,node);top.emplace(d,node);++out.pushes;if(top.size()>ef)top.pop();lower=top.top().first;}}}
    out.trigger_exact=gate.trigger_exact;out.trigger_step=gate.trigger_step;out.continuation_ns=nsSince(continuation0);while(top.size()>k)top.pop();const auto serving_end=Clock::now();std::unordered_set<unsigned>truth(gt,gt+k);size_t hits=0;
    while(!top.empty()){hits+=truth.count(top.top().second);top.pop();}out.recall=double(hits)/k;out.ndc=target_hierarchy+pca.m.dpca+double(out.exact)+double(out.pca_dims)/pca.m.dim;
    out.seconds=std::chrono::duration<double>(serving_end-wall0).count();return out;
}

enum class TerminationPolicy { Darth, Laet };

Result searchLearnedTermination(const HierarchicalNSW<int>&index,const ScalarModel&model,
        const uint8_t*q,const unsigned*gt,size_t k,uint32_t entry,int ef,
        TerminationPolicy policy,int darth_initial,int darth_minimum,int laet_warmup,
        float laet_alpha,uint64_t budget_cap,ThreadState&state){
    Result out;out.ef=ef;const auto wall0=Clock::now();entry=measuredEntry(index,q,entry);const uint32_t epoch=state.next();
    std::priority_queue<Pair> efheap,topk;
    std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>>cand;
    const float first=exactDistance(index,q,entry);++out.exact;out.visited=1;out.pushes=1;
    efheap.emplace(first,entry);topk.emplace(first,entry);cand.emplace(first,entry);state.seen[entry]=epoch;
    float lower=first;uint64_t inserts=1,next_prediction=darth_initial,budget=0;bool stopped=false;
    const auto prefix0=Clock::now();bool prefix_done=false;
    while(!cand.empty()&&!stopped){const auto[current_dist,current]=cand.top();
        if(efheap.size()>=size_t(ef)&&current_dist>lower)break;
        cand.pop();++out.pops;const uint64_t step=out.expanded++;
        auto*links=reinterpret_cast<unsigned*>(index.get_linklist0(current));const uint32_t count=index.getListCount(links);
        auto*ids=reinterpret_cast<tableint*>(links+1);
        for(uint32_t ni=0;ni<count;++ni){const uint32_t node=ids[ni];if(state.seen[node]==epoch)continue;
            state.seen[node]=epoch;++out.visited;const float distance=exactDistance(index,q,node);++out.exact;
            if(topk.size()<k||distance<topk.top().first){topk.emplace(distance,node);if(topk.size()>k)topk.pop();++inserts;}
            if(efheap.size()<size_t(ef)||distance<lower){cand.emplace(distance,node);efheap.emplace(distance,node);++out.pushes;
                if(efheap.size()>size_t(ef))efheap.pop();lower=efheap.top().first;}
            const bool checkpoint=(policy==TerminationPolicy::Darth&&out.exact>=next_prediction) ||
                                  (policy==TerminationPolicy::Laet&&!budget&&out.exact>=uint64_t(laet_warmup));
            if(checkpoint&&topk.size()>=k){if(!prefix_done){out.probe_ns=nsSince(prefix0);prefix_done=true;}
                const auto feature0=Clock::now();auto heap=topk;std::vector<float>ds;ds.reserve(k);
                while(!heap.empty()){ds.push_back(std::sqrt(std::max(heap.top().first,0.0f)));heap.pop();}std::sort(ds.begin(),ds.end());
                const float first_root=std::sqrt(std::max(first,0.0f));
                std::vector<float>features;if(policy==TerminationPolicy::Darth){double mean=0,var=0;
                    for(float x:ds)mean+=x;mean/=ds.size();for(float x:ds)var+=(x-mean)*(x-mean);var/=ds.size();
                    features={float(step),float(out.exact),float(inserts),first_root,ds.front(),float(mean),ds.back(),float(var),
                              ds[size_t(.50*(k-1))],ds[size_t(.25*(k-1))],ds[size_t(.75*(k-1))]};
                }else{const float d1=ds[0],d10=ds[std::min<size_t>(9,k-1)];
                    features={first_root,d1,d10,first_root>0?d1/first_root:-1.0f,first_root>0?d10/first_root:-1.0f};
                    for(uint32_t j=0;j<128;++j)features.push_back(float(q[j]));}
                out.feature_ns+=nsSince(feature0);const auto pred0=Clock::now();const float prediction=model.predict(features);
                out.predictor_ns+=nsSince(pred0);++out.predictor_calls;
                if(policy==TerminationPolicy::Darth){const float recall=std::clamp(prediction,0.0f,1.0f);
                    if(recall>=target_stop){stopped=true;break;}
                    const int delta=darth_minimum+int((darth_initial-darth_minimum)*(target_stop-recall));
                    next_prediction=out.exact+std::max(1,delta);
                }else budget=std::min<uint64_t>(budget_cap,std::max<uint64_t>(laet_warmup,std::llround(laet_alpha*prediction)));
            }
            if(policy==TerminationPolicy::Laet&&budget&&out.exact>=budget){stopped=true;break;}
        }
    }
    if(!prefix_done)out.probe_ns=nsSince(prefix0);
    const auto serving_end=Clock::now();auto heap=topk;std::unordered_set<unsigned>truth(gt,gt+k);size_t hits=0;while(!heap.empty()){hits+=truth.count(heap.top().second);heap.pop();}
    out.recall=double(hits)/k;out.ndc=target_hierarchy+out.exact;out.seconds=std::chrono::duration<double>(serving_end-wall0).count();
    out.continuation_ns=uint64_t(out.seconds*1e9)-std::min<uint64_t>(uint64_t(out.seconds*1e9),out.probe_ns+out.feature_ns+out.predictor_ns);
    return out;
}

std::unordered_map<uint32_t, uint32_t> readEntries(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("cannot open " + path);
    std::string line; std::getline(in, line);
    std::unordered_map<uint32_t, uint32_t> entries;
    while (std::getline(in, line)) {
        const size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        const size_t comma2 = line.find(',', comma + 1);
        entries[std::stoul(line.substr(0, comma))] =
            std::stoul(line.substr(comma + 1, comma2 - comma - 1));
    }
    return entries;
}

Json::Value readJson(const std::string& path) {
    Json::Value v; Json::CharReaderBuilder rb; std::string errors; std::ifstream in(path);
    if (!Json::parseFromStream(rb, in, &v, &errors)) throw std::runtime_error("bad JSON: " + errors);
    return v;
}

struct Arm { std::string name; std::vector<Result> rows; double wall_seconds = 0; };

Json::Value summarize(const Arm& arm) {
    Json::Value out; out["name"] = arm.name; out["n"] = Json::UInt64(arm.rows.size());
    out["wall_seconds"] = arm.wall_seconds;
    out["throughput_ms_per_query"] = arm.rows.empty()?0.0:arm.wall_seconds*1000.0/arm.rows.size();
    out["qps"] = arm.wall_seconds>0?arm.rows.size()/arm.wall_seconds:0.0;
    long double recall=0, ndc=0, sec=0, exact=0, estimates=0, verifies=0, rejects=0;
    long double visited=0, expanded=0, pushes=0, pops=0, pca_dims=0;
    long double projection_ns=0, pca_ns=0, exact_ns=0, heap_ns=0, graph_ns=0;
    long double probe_ns=0,feature_ns=0,predictor_ns=0,continuation_ns=0,predictor_calls=0;
    Json::Value per(Json::arrayValue);
    for (const auto& r : arm.rows) {
        recall+=r.recall; ndc+=r.ndc; sec+=r.seconds; exact+=r.exact; estimates+=r.estimates;
        verifies+=r.verifies; rejects+=r.rejects; visited+=r.visited; expanded+=r.expanded;
        pushes+=r.pushes; pops+=r.pops; pca_dims+=r.pca_dims; projection_ns+=r.projection_ns;
        pca_ns+=r.pca_ns; exact_ns+=r.exact_ns; heap_ns+=r.heap_ns; graph_ns+=r.graph_ns;
        probe_ns+=r.probe_ns;feature_ns+=r.feature_ns;predictor_ns+=r.predictor_ns;continuation_ns+=r.continuation_ns;
        predictor_calls+=r.predictor_calls;
        Json::Value x; x["recall"]=r.recall; x["ndc"]=r.ndc; x["seconds"]=r.seconds;
        x["ef"]=r.ef; x["tier"]=r.tier; x["exact"]=Json::UInt64(r.exact);
        x["estimates"]=Json::UInt64(r.estimates); x["verifies"]=Json::UInt64(r.verifies);
        x["rejects"]=Json::UInt64(r.rejects); x["visited"]=Json::UInt64(r.visited);
        x["expanded"]=Json::UInt64(r.expanded); x["pushes"]=Json::UInt64(r.pushes);
        x["pops"]=Json::UInt64(r.pops); x["trigger_exact"]=Json::UInt64(r.trigger_exact);x["trigger_step"]=Json::UInt64(r.trigger_step);per.append(x);
    }
    const long double n=arm.rows.size();
    out["mean_recall"]=double(recall/n); out["mean_ndc"]=double(ndc/n);
    out["mean_query_ms"]=double(sec/n*1000); out["mean_exact"]=double(exact/n);
    out["mean_estimates"]=double(estimates/n); out["mean_verifies"]=double(verifies/n);
    out["mean_rejects"]=double(rejects/n); out["mean_visited"]=double(visited/n);
    out["mean_expanded"]=double(expanded/n); out["mean_pushes"]=double(pushes/n);
    out["mean_pops"]=double(pops/n); out["mean_pca_dims"]=double(pca_dims/n);
    Json::Value timing; timing["projection_ms_per_query"]=double(projection_ns/n/1e6L);
    timing["pca_kernel_ms_per_query"]=double(pca_ns/n/1e6L);
    timing["exact_kernel_ms_per_query"]=double(exact_ns/n/1e6L);
    timing["heap_ms_per_query"]=double(heap_ns/n/1e6L);
    timing["graph_visited_ms_per_query"]=double(graph_ns/n/1e6L);
    timing["probe_ms_per_query"]=double(probe_ns/n/1e6L);
    timing["feature_ms_per_query"]=double(feature_ns/n/1e6L);
    timing["predictor_ms_per_query"]=double(predictor_ns/n/1e6L);
    timing["continuation_ms_per_query"]=double(continuation_ns/n/1e6L);
    timing["predictor_calls_per_query"]=double(predictor_calls/n);
    out["instrumented_timing"]=timing; out["per_query"]=per;
    return out;
}

template<class TierFn>
Arm runArm(const std::string& name, const std::vector<uint32_t>& qids,
           const Matrix<uint8_t>& queries, const Matrix<unsigned>& gt,
           const std::unordered_map<uint32_t,uint32_t>& entries,
           const HierarchicalNSW<int>& index, const PcaTable& pca,
           const std::vector<int>& caps, size_t k, bool pca_hard, TierFn tier_fn,
           ThreadState& state, float admission_margin=1.0f) {
    Arm arm; arm.name=name; arm.rows.reserve(qids.size()); const auto wall0=Clock::now();
    for (size_t pos=0; pos<qids.size(); ++pos) {
        const uint32_t qi=qids[pos]; const int tier=tier_fn(pos); const int ef=caps.at(tier);
        const bool compressed=pca_hard && tier==2;
        Result r=searchOne(index,pca,queries.data+uint64_t(qi)*queries.d,
                           gt.data+uint64_t(qi)*gt.d,k,entries.at(qi),ef,compressed,admission_margin,state);
        r.tier=tier; arm.rows.push_back(r);
        if((pos+1)%50==0) std::cerr<<'['<<name<<"] "<<pos+1<<'/'<<qids.size()<<'\n';
    }
    arm.wall_seconds=std::chrono::duration<double>(Clock::now()-wall0).count();
    return arm;
}

Arm runAdaptiveArm(const std::string&name,const std::vector<uint32_t>&qids,
        const Matrix<uint8_t>&queries,const Matrix<unsigned>&gt,
        const std::unordered_map<uint32_t,uint32_t>&entries,const HierarchicalNSW<int>&index,
        const PcaTable&pca,const TierModel&model,const std::vector<int>&caps,size_t k,
        bool pca_hard,float margin,ThreadState&state){
    const char*threads_env=std::getenv("QUERY_THREADS");const int threads=threads_env?std::max(1,std::atoi(threads_env)):1;
    Arm arm;arm.name=name;arm.rows.resize(qids.size());
    std::vector<std::unique_ptr<ThreadState>>parallel_states;
    if(threads>1)for(int t=0;t<threads;++t)parallel_states.emplace_back(new ThreadState(index.cur_element_count));
    const auto wall0=Clock::now();
    if(threads==1){for(size_t pos=0;pos<qids.size();++pos){const uint32_t qi=qids[pos];arm.rows[pos]=searchAdaptive(index,pca,model,
        queries.data+uint64_t(qi)*queries.d,gt.data+uint64_t(qi)*gt.d,k,entries.at(qi),caps,pca_hard,margin,state);
        if((pos+1)%50==0)std::cerr<<'['<<name<<"] "<<pos+1<<'/'<<qids.size()<<'\n';}}
    else {
#pragma omp parallel for num_threads(threads) schedule(static)
        for(long long pos=0;pos<(long long)qids.size();++pos){const uint32_t qi=qids[pos];ThreadState&local=*parallel_states[omp_get_thread_num()];
            arm.rows[pos]=searchAdaptive(index,pca,model,queries.data+uint64_t(qi)*queries.d,
                gt.data+uint64_t(qi)*gt.d,k,entries.at(qi),caps,pca_hard,margin,local);}
    }
    arm.wall_seconds=std::chrono::duration<double>(Clock::now()-wall0).count();return arm;
}

Arm runTerminationArm(const std::string&name,const std::vector<uint32_t>&qids,
        const Matrix<uint8_t>&queries,const Matrix<unsigned>&gt,
        const std::unordered_map<uint32_t,uint32_t>&entries,const HierarchicalNSW<int>&index,
        const ScalarModel&model,TerminationPolicy policy,int ef,int initial,int minimum,
        int warmup,float alpha,uint64_t budget_cap,ThreadState&state){
    Arm arm;arm.name=name;arm.rows.reserve(qids.size());const auto wall0=Clock::now();
    for(size_t pos=0;pos<qids.size();++pos){const uint32_t qi=qids[pos];arm.rows.push_back(searchLearnedTermination(
        index,model,queries.data+uint64_t(qi)*queries.d,gt.data+uint64_t(qi)*gt.d,100,entries.at(qi),ef,
        policy,initial,minimum,warmup,alpha,budget_cap,state));
        if((pos+1)%50==0)std::cerr<<'['<<name<<"] "<<pos+1<<'/'<<qids.size()<<'\n';}
    arm.wall_seconds=std::chrono::duration<double>(Clock::now()-wall0).count();return arm;
}

} // namespace


int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: reported_sift CONFIG.json");
 auto cfg=readJson(argv[1]);
 std::string qp=cfg["query"].asString(),gp=cfg["gt"].asString();Matrix<uint8_t>queries(qp.data());Matrix<unsigned>gt(gp.data());
 auto entries=readEntries(cfg["entries"].asString());auto plan=readJson(cfg["planner"].asString());
 std::vector<uint32_t>qids;std::vector<int>caps;
 for(auto&q:plan["query_ids"])qids.push_back(q.asUInt());for(auto&e:plan["tier_ef_initial"])caps.push_back(e.asInt());
 if(cfg.isMember("limit")&&cfg["limit"].asUInt()<qids.size())qids.resize(cfg["limit"].asUInt());
 PcaModel pm(cfg["pca_model"].asString());PcaTable pca(cfg["pca_scores"].asString(),cfg["pca_residual"].asString(),pm);
 std::cerr<<"[load] index\n";L2SpaceU8 space(queries.d);HierarchicalNSW<int>index(&space,cfg["index"].asString(),false);
 if(index.M_!=16||index.ef_construction_!=100||index.cur_element_count!=100000000)throw std::runtime_error("index configuration mismatch");
 std::cerr<<"[warm] PCA\n";pca.prewarm();ThreadState state(index.cur_element_count);
 TierModel tier(cfg["tier_model"].asString(),51);ScalarModel darth(cfg["darth_model"].asString(),11),laet(cfg["laet_model"].asString(),133);
 Json::Value output;output["config"]=cfg;output["timing_scope"]="Single-thread hierarchy + query feature/projection/model + base search; GT scoring excluded; index and PCA resident";
 std::vector<std::string>methods={"current","never","heap_full","stall128","combined"};if(cfg.isMember("methods")){methods.clear();for(auto&m:cfg["methods"])methods.push_back(m.asString());}
 for(int rep=-1;rep<cfg.get("repeats",5).asInt();++rep){
  auto ids=qids;if(rep<0&&ids.size()>20)ids.resize(20);
  for(size_t j=0;j<methods.size();++j){const auto&label=methods[(j+std::max(0,rep))%methods.size()];auto m=target_arm(label);auto scaled_caps=caps;for(auto&c:scaled_caps)c=target_cap(c);if(m=="never"&&label.find('@')!=std::string::npos)scaled_caps.at(2)=target_ef;fused_policy(m);set_plateau_policy(dimension_kind==1?"heap_full":dimension_kind==2?"never":m);Arm arm;
   if(m=="plain")arm=runArm(m,ids,queries,gt,entries,index,pca,{target_ef,target_ef,target_ef},100,false,[](size_t){return 0;},state);
   else if(m!="darth"&&m!="laet")arm=runAdaptiveArm(m,ids,queries,gt,entries,index,pca,tier,scaled_caps,100,true,1.05f,state);
   else if(m=="darth")arm=runTerminationArm(m,ids,queries,gt,entries,index,darth,TerminationPolicy::Darth,target_ef,1979,396,241,1.0f,120000,state);
   else arm=runTerminationArm(m,ids,queries,gt,entries,index,laet,TerminationPolicy::Laet,target_ef,1979,396,241,target_alpha,120000,state);
   arm.name=label;auto summary=summarize(arm);summary["repeat"]=rep;
   for(Json::ArrayIndex i=0;i<summary["per_query"].size();++i)summary["per_query"][i]["query_id"]=ids[i];
   std::cerr<<"[done] "<<m<<" repeat="<<rep<<" recall="<<summary["mean_recall"]<<" ndc="<<summary["mean_ndc"]<<" ms="<<summary["mean_query_ms"]<<'\n';
   if(rep>=0)output["runs"].append(summary);
   std::ofstream out(cfg["output"].asString());out<<output<<'\n';
  }
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
