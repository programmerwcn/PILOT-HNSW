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
#include <map>
#include "plateau_gate.h"
#include <json/json.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

extern "C" {
typedef void* BoosterHandle;
int LGBM_BoosterCreateFromModelfile(const char*, int*, BoosterHandle*);
int LGBM_BoosterPredictForMat(BoosterHandle, const void*, int, int32_t, int32_t,
                              int, int, int, int, const char*, int64_t*, double*);
int LGBM_BoosterFree(BoosterHandle);
const char* LGBM_GetLastError();
}
constexpr int LGBM_FLOAT64 = 1, LGBM_PREDICT_NORMAL = 0;
using Clock = std::chrono::steady_clock;
using Pair = std::pair<float, uint32_t>;

struct Matrix {
    uint32_t d = 0;
    std::vector<float> x;
    size_t n() const { return d ? x.size() / d : 0; }
    const float* row(size_t i) const { return x.data() + i * d; }
};

Matrix read_fvecs(const std::string& p) {
    std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("open " + p);
    int32_t d; f.read((char*)&d,4); f.seekg(0,std::ios::end); size_t bytes=f.tellg();
    size_t n=bytes/(4+4*d); f.seekg(0);
    Matrix m; m.d=d; m.x.resize(n*d);
    std::vector<float> row(d);
    for(size_t i=0;i<n;++i){ int32_t dd; f.read((char*)&dd,4); f.read((char*)m.x.data()+i*d*4,d*4); }
    return m;
}
std::vector<std::vector<uint32_t>> read_ivecs(const std::string& p) {
    std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("open "+p);
    int32_t d; f.read((char*)&d,4); f.seekg(0,std::ios::end); size_t n=size_t(f.tellg())/(4+4*d); f.seekg(0);
    std::vector<std::vector<uint32_t>> g(n,std::vector<uint32_t>(d));
    for(size_t i=0;i<n;++i){int32_t dd;f.read((char*)&dd,4);f.read((char*)g[i].data(),d*4);} return g;
}
struct Graph { std::vector<uint64_t> ptr; std::vector<uint32_t> adj; };
Graph read_graph(const std::string& p, size_t n) {
    std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("open "+p);
    uint32_t magic,ver; uint64_t num; f.read((char*)&magic,4);f.read((char*)&ver,4);f.read((char*)&num,8);
    if(magic!=0x424c4e45||ver!=1||num!=n) throw std::runtime_error("bad graph header");
    std::vector<uint32_t> deg(n);
    for(size_t i=0;i<n;++i){uint64_t id;uint32_t k;f.read((char*)&id,8);f.read((char*)&k,4);deg[id]=k;f.seekg(8ull*k,std::ios::cur);}
    Graph g;g.ptr.resize(n+1);for(size_t i=0;i<n;++i)g.ptr[i+1]=g.ptr[i]+deg[i];g.adj.resize(g.ptr[n]);
    f.clear();f.seekg(16);
    std::vector<uint64_t> tmp;
    for(size_t i=0;i<n;++i){uint64_t id;uint32_t k;f.read((char*)&id,8);f.read((char*)&k,4);tmp.resize(k);f.read((char*)tmp.data(),8ull*k);for(uint32_t j=0;j<k;++j)g.adj[g.ptr[id]+j]=tmp[j];}
    return g;
}
std::vector<int64_t> read_entries(const std::string& p,size_t n){
    std::ifstream f(p);if(!f)throw std::runtime_error("open "+p);std::vector<int64_t> e(n,-1);std::string s;std::getline(f,s);
    while(std::getline(f,s)){auto z=s.find(',');if(z!=std::string::npos)e[std::stoull(s.substr(0,z))]=std::stoll(s.substr(z+1));}return e;
}
Json::Value read_json(const std::string& p){std::ifstream f(p);Json::Value v;Json::CharReaderBuilder b;std::string err;if(!Json::parseFromStream(b,f,&v,&err))throw std::runtime_error(err);return v;}

struct PCA {
    uint64_t n=0;uint32_t d=0,p=0;std::vector<float> mu,proj,b,r2;
    static PCA load(const std::string& path){
        std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open "+path);PCA x;char magic[8];f.read(magic,8);f.read((char*)&x.n,8);f.read((char*)&x.d,4);f.read((char*)&x.p,4);
        if(std::memcmp(magic,"PCA64V1",7))throw std::runtime_error("bad PCA");x.mu.resize(x.d);x.proj.resize(size_t(x.d)*x.p);x.b.resize(size_t(x.n)*x.p);x.r2.resize(x.n);
        f.read((char*)x.mu.data(),x.mu.size()*4);f.read((char*)x.proj.data(),x.proj.size()*4);f.read((char*)x.b.data(),x.b.size()*4);f.read((char*)x.r2.data(),x.r2.size()*4);return x;
    }

    PCA prefix(unsigned k)const{
      if(!k||k>p)throw std::runtime_error("PCA prefix unavailable");
      PCA x;x.n=n;x.d=d;x.p=k;x.mu=mu;x.proj.resize(size_t(d)*k);x.b.resize(n*k);x.r2=r2;
      for(unsigned j=0;j<d;++j)for(unsigned c=0;c<k;++c)x.proj[size_t(j)*k+c]=proj[size_t(j)*p+c];
      for(size_t i=0;i<n;++i){double residual=r2[i];for(unsigned c=0;c<p;++c){float v=b[i*p+c];if(c<k)x.b[i*k+c]=v;else residual+=double(v)*v;}x.r2[i]=residual;}
      return x;
    }
    void query(const float* q,std::vector<float>& q64,float& rq2)const{
        q64.assign(p,0);double total=0;
        for(uint32_t j=0;j<d;++j){float z=q[j]-mu[j];total+=z*z;for(uint32_t k=0;k<p;++k)q64[k]+=z*proj[j*p+k];}
        double kept=0;for(float z:q64)kept+=z*z;rq2=std::max(0.0,double(total-kept));
    }
};
inline float l2sq(const float* a,const float* b,int d){float s=0;
#pragma omp simd reduction(+:s)
    for(int i=0;i<d;++i){float z=a[i]-b[i];s+=z*z;}return s;
}
struct Model{
    BoosterHandle h=nullptr;int nf=0;
    Model()=default;Model(const std::string&p,int n):nf(n){int it;if(LGBM_BoosterCreateFromModelfile(p.c_str(),&it,&h))throw std::runtime_error(LGBM_GetLastError());}
    Model(const Model&)=delete;Model&operator=(const Model&)=delete;
    ~Model(){if(h)LGBM_BoosterFree(h);}
    double predict(const std::vector<double>& x)const{int64_t out=0;double y=0;if(LGBM_BoosterPredictForMat(h,x.data(),LGBM_FLOAT64,1,nf,1,LGBM_PREDICT_NORMAL,0,-1,"num_threads=1",&out,&y))throw std::runtime_error(LGBM_GetLastError());return y;}
};
struct Result{
    uint64_t trigger_exact=0,trigger_step=0;double validation_ms=0,returned_recall=0,touched_recall=0,ndc=0,search_ndc=0,projection_ndc=0,ms=0;uint64_t full=0,pca=0,pca_dims=0,expanded=0,pushes=0,pops=0,predicts=0;double predicted=0,budget=0;bool adaptive_stop=false;
};
struct Runner{
    const Matrix& base;const Matrix& query;const std::vector<std::vector<uint32_t>>& gt;const Graph& graph;const std::vector<int64_t>& entry;const PCA* pca;
    std::vector<uint32_t> seen;uint32_t epoch=0;static constexpr int K=100;
    Runner(const Matrix&b,const Matrix&q,const std::vector<std::vector<uint32_t>>&g,const Graph&a,const std::vector<int64_t>&e,const PCA*p):base(b),query(q),gt(g),graph(a),entry(e),pca(p),seen(b.n()){}
    bool isgt(size_t qi,uint32_t x)const{return false; /* No touched-GT diagnostics inside search timer. */}
    void finish(size_t qi,std::priority_queue<Pair>& topk,std::unordered_set<uint32_t>& touched,Result&r){
        auto validation_start=Clock::now();int hit=0;auto h=topk;while(!h.empty()){if(std::find(gt[qi].begin(),gt[qi].begin()+K,h.top().second)!=gt[qi].begin()+K)++hit;h.pop();}r.returned_recall=hit/double(K);r.touched_recall=touched.size()/double(K);r.validation_ms+=std::chrono::duration<double,std::milli>(Clock::now()-validation_start).count();
    }
    Result adaptive_plan(size_t qi,int ef,bool compressed){
        PilotProbeState resume;Result r;auto t0=Clock::now();++epoch;if(!epoch){std::fill(seen.begin(),seen.end(),0);++epoch;}
        const float*q=query.row(qi);const uint32_t ep=entry[qi];std::vector<float>q64;float rq2=0;pca->query(q,q64,rq2);
        auto exact=[&](uint32_t x){++r.full;return l2sq(base.row(x),q,base.d);};
        std::priority_queue<Pair>top,topk,deferred;std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>>cand;std::unordered_set<uint32_t>touched;
        const float first=exact(ep);top.push({first,ep});topk.push({first,ep});cand.push({first,ep});seen[ep]=epoch;if(isgt(qi,ep))touched.insert(ep);r.pushes=1;resume.record(first,ep);
        // Faithful zero-anchor common routing probe: 100 expansions, heap cap 200.
        while(!cand.empty()&&r.expanded<100){auto[current,id]=cand.top();cand.pop();++r.pops;++r.expanded;resume.expand(id);
            for(uint64_t z=graph.ptr[id];z<graph.ptr[id+1];++z){uint32_t x=graph.adj[z];if(seen[x]==epoch)continue;seen[x]=epoch;float dx=exact(x);resume.record(dx,x);if(isgt(qi,x))touched.insert(x);
                if(top.size()<200||dx<top.top().first){cand.push({dx,x});top.push({dx,x});topk.push({dx,x});++r.pushes;if(top.size()>200){top.pop();}if(topk.size()>K)topk.pop();}
                }}
        resume.restore(ef,top,cand);
        float lower=top.top().first;
        while(!cand.empty()){auto[current,id]=cand.top();if(top.size()>=size_t(ef)&&current>lower)break;cand.pop();++r.pops;++r.expanded;
            for(uint64_t z=graph.ptr[id];z<graph.ptr[id+1];++z){uint32_t x=graph.adj[z];if(seen[x]==epoch)continue;seen[x]=epoch;
                if(compressed&&top.size()>=size_t(ef)){++r.pca;const float*bx=pca->b.data()+size_t(x)*pca->p;float estimate=pca->r2[x]+rq2+l2sq(bx,q64.data(),pca->p);r.pca_dims+=pca->p;if(estimate>=lower)continue;}
                float dx=exact(x);if(isgt(qi,x))touched.insert(x);if(top.size()<size_t(ef)||dx<lower){cand.push({dx,x});top.push({dx,x});topk.push({dx,x});++r.pushes;if(top.size()>size_t(ef))top.pop();if(topk.size()>K)topk.pop();lower=top.top().first;}}}
        finish(qi,topk,touched,r);r.search_ndc=r.full+r.pca_dims/double(base.d);r.projection_ndc=compressed?pca->p:32;r.ndc=r.search_ndc+r.projection_ndc;r.ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count()-r.validation_ms;return r;
    }
    Result hnsw(size_t qi,int ef,bool compressed,const Model* darth=nullptr,
                bool progressive=false,double stop_tau=-1,int stop_min=2000,int stop_window=512){
        Result r;auto t0=Clock::now();PlateauGate gate(K);bool projected=false;++epoch;if(!epoch){std::fill(seen.begin(),seen.end(),0);++epoch;}
        const float*q=query.row(qi);uint32_t ep=entry[qi];auto exact=[&](uint32_t x){++r.full;return std::sqrt(l2sq(base.row(x),q,base.d));};
        std::vector<float>q64;float rq2=0;if(compressed){pca->query(q,q64,rq2);projected=true;}
        std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>> cand;std::priority_queue<Pair> top,topk;std::unordered_set<uint32_t> touched;if(isgt(qi,ep))touched.insert(ep);
        float d0=exact(ep);seen[ep]=epoch;cand.push({d0,ep});top.push({d0,ep});topk.push({d0,ep});float lower=d0;gate.observe(d0*d0,true,d0*d0);uint64_t inserts=1;double nextpred=2671;double pred=0;
        std::deque<std::pair<uint64_t,float>> radius_history;
        while(!cand.empty()){auto [cd,id]=cand.top();if(cd>lower&&(int)top.size()==ef)break;cand.pop();++r.pops;++r.expanded;
            for(uint64_t z=graph.ptr[id];z<graph.ptr[id+1];++z){uint32_t x=graph.adj[z];if(seen[x]==epoch)continue;seen[x]=epoch;float dx;bool fused=dimension_kind==2&&(int)top.size()>=ef;if(fused){bool rejected;dx=std::sqrt(dimension_fused(base.row(x),q,base.d,lower*lower,r.pca_dims,rejected));if(rejected)continue;}
                if(compressed||gate.check(top.size()>=size_t(ef),lower*lower,r.full,r.expanded)){if(!projected){pca->query(q,q64,rq2);projected=true;}++r.pca;const float* bx=pca->b.data()+size_t(x)*pca->p;float ds=pca->r2[x]+rq2;
                    if(progressive){
                        ds+=l2sq(bx,q64.data(),16);r.pca_dims+=16;
                        if((int)top.size()>=ef&&ds>=lower*lower)continue;
                        ds+=l2sq(bx+16,q64.data()+16,16);r.pca_dims+=16;
                        if((int)top.size()>=ef&&ds>=lower*lower)continue;
                        ds+=l2sq(bx+32,q64.data()+32,32);r.pca_dims+=32;
                    }else{ds+=l2sq(bx,q64.data(),pca->p);r.pca_dims+=pca->p;}
                    float dh=std::sqrt(std::max(0.f,ds));if((int)top.size()>=ef&&dh>=lower*dimension_margin)continue;dx=exact(x);}
                else if(!fused)dx=exact(x);
                gate.observe(dx*dx,(int)top.size()<ef||dx<lower,lower*lower);if(isgt(qi,x))touched.insert(x);
                if((int)top.size()<ef||dx<lower){cand.push({dx,x});top.push({dx,x});++r.pushes;++inserts;if((int)top.size()>ef)top.pop();lower=top.top().first;topk.push({dx,x});if((int)topk.size()>K)topk.pop();}
                if(darth&&r.full>=nextpred&&(int)topk.size()>=K){std::vector<float>ds;auto hh=topk;while(!hh.empty()){ds.push_back(hh.top().first);hh.pop();}std::sort(ds.begin(),ds.end());double mean=std::accumulate(ds.begin(),ds.end(),0.0)/K,var=0;for(float v:ds)var+=(v-mean)*(v-mean);var/=K;
                    std::vector<double>f={double(r.expanded),double(r.full),double(inserts),d0,ds[0],mean,ds.back(),var,ds[49],ds[24],ds[74]};pred=std::clamp(darth->predict(f),0.0,1.0);++r.predicts;
                    if(pred>=target_stop){r.predicted=pred;finish(qi,topk,touched,r);r.search_ndc=r.ndc=r.full; r.ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count()-r.validation_ms;return r;}
                    nextpred=r.full+std::max(1,int(534+(2671-534)*(target_stop-pred)));
                }
            }
            if(stop_tau>=0&&(int)topk.size()>=K&&(r.expanded%64==0)){
                float rk=topk.top().first;radius_history.push_back({r.expanded,rk});
                while(radius_history.size()>1&&radius_history[1].first+uint64_t(stop_window)<=r.expanded)radius_history.pop_front();
                if(r.expanded>=uint64_t(stop_min)&&!radius_history.empty()&&radius_history.front().first+uint64_t(stop_window)<=r.expanded){
                    double improvement=(radius_history.front().second-rk)/std::max(1e-12f,radius_history.front().second);
                    if(improvement<stop_tau){r.adaptive_stop=true;break;}
                }
            }
        }
        r.predicted=pred;finish(qi,topk,touched,r);r.search_ndc=r.full+r.pca_dims/double(base.d);r.trigger_exact=gate.trigger_exact;r.trigger_step=gate.trigger_step;r.projection_ndc=projected?pca->p:0;r.ndc=r.search_ndc+r.projection_ndc;r.ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count()-r.validation_ms;return r;
    }
    Result laet(size_t qi,const Model&model,double alpha=0.85,int F=1260,int cap=120000){
        Result r;auto t0=Clock::now();++epoch;if(!epoch){std::fill(seen.begin(),seen.end(),0);++epoch;}const float*q=query.row(qi);uint32_t ep=entry[qi];
        auto dist=[&](uint32_t x){++r.full;return l2sq(base.row(x),q,base.d);};float d0=dist(ep);seen[ep]=epoch;
        std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>>cand;std::priority_queue<Pair>topk;cand.push({d0,ep});topk.push({d0,ep});std::unordered_set<uint32_t>touched;if(isgt(qi,ep))touched.insert(ep);uint64_t budget=0;
        while(!cand.empty()){auto [cd,id]=cand.top();cand.pop();++r.pops;++r.expanded;for(uint64_t z=graph.ptr[id];z<graph.ptr[id+1];++z){uint32_t x=graph.adj[z];if(seen[x]==epoch)continue;seen[x]=epoch;float dx=dist(x);cand.push({dx,x});++r.pushes;topk.push({dx,x});if((int)topk.size()>K)topk.pop();if(isgt(qi,x))touched.insert(x);
                if(!budget&&r.full>=uint64_t(F)&&(int)topk.size()>=K){std::vector<float>ds;auto h=topk;while(!h.empty()){ds.push_back(h.top().first);h.pop();}std::sort(ds.begin(),ds.end());std::vector<double>f(base.d+5);for(uint32_t j=0;j<base.d;++j)f[j]=q[j];double d1=ds[0],d10=ds[9];f[base.d]=d0;f[base.d+1]=d1;f[base.d+2]=d10;f[base.d+3]=d0>0?d1/d0:-1;f[base.d+4]=d0>0?d10/d0:-1;r.predicted=std::pow(2.,model.predict(f));++r.predicts;budget=std::min<uint64_t>(cap,std::max<uint64_t>(F,std::llround(alpha*r.predicted)));r.budget=budget;}
                if(budget&&r.full>=budget){finish(qi,topk,touched,r);r.search_ndc=r.ndc=r.full;r.ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count()-r.validation_ms;return r;}
            }}
        finish(qi,topk,touched,r);r.search_ndc=r.ndc=r.full;r.ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count()-r.validation_ms;return r;
    }
};
struct Agg{double rr=0,tr=0,ndc=0,search_ndc=0,projection_ndc=0,ms=0;uint64_t n=0,full=0,pca=0,pca_dims=0,expanded=0,predicts=0,adaptive=0,attain=0;void add(const Result&r){++n;rr+=r.returned_recall;tr+=r.touched_recall;ndc+=r.ndc;search_ndc+=r.search_ndc;projection_ndc+=r.projection_ndc;ms+=r.ms;full+=r.full;pca+=r.pca;pca_dims+=r.pca_dims;expanded+=r.expanded;predicts+=r.predicts;adaptive+=r.adaptive_stop;attain+=r.returned_recall>=.89999;}};
Json::Value jout(const Agg&a){Json::Value v;v["n"]=Json::UInt64(a.n);v["returned_recall"]=a.rr/a.n;v["target_attainment"]=double(a.attain)/a.n;v["touched_recall"]=a.tr/a.n;v["mean_ndc"]=a.ndc/a.n;v["mean_search_ndc"]=a.search_ndc/a.n;v["mean_projection_ndc"]=a.projection_ndc/a.n;v["mean_ms"]=a.ms/a.n;v["qps"]=1000*a.n/a.ms;v["mean_full"]=double(a.full)/a.n;v["mean_pca"]=double(a.pca)/a.n;v["mean_pca_dims"]=double(a.pca_dims)/a.n;v["mean_expanded"]=double(a.expanded)/a.n;v["mean_predictions"]=double(a.predicts)/a.n;v["adaptive_stop_rate"]=double(a.adaptive)/a.n;return v;}

int main(int argc,char**argv){
 try {
    if(argc!=3)throw std::runtime_error("usage: reported_float CONFIG.json OUT.json");
    auto cfg=read_json(argv[1]);
    std::cerr<<"[load] base/query/graph\n";
    Matrix base=read_fvecs(cfg["base"].asString()),q=read_fvecs(cfg["query"].asString());
    auto gt=read_ivecs(cfg["gt"].asString());
    auto graph=read_graph(cfg["graph"].asString(),base.n());
    auto entry=read_entries(cfg["entry"].asString(),q.n());
    auto pca=PCA::load(cfg["pca"].asString());
    if(pca.n!=base.n()||pca.d!=base.d)throw std::runtime_error("PCA/base shape mismatch");
    auto replay=read_json(cfg["replay"].asString());
    Model darth(cfg["darth_model"].asString(),11),laet(cfg["laet_model"].asString(),base.d+5);
    Runner run(base,q,gt,graph,entry,&pca);
    std::vector<std::string>methods={"current","never","heap_full","stall128","combined"};if(cfg.isMember("methods")){methods.clear();for(auto&m:cfg["methods"])methods.push_back(m.asString());}
    std::map<int,PCA>prefixes;Json::Value output;output["config"]=cfg;
    output["timing_scope"]="Base-layer serving from saved entry; PCA projection and termination models included; replayed tier assignment; GT scoring excluded";
    const int repeats=cfg.get("repeats",5).asInt();
    const size_t limit=std::min<size_t>(replay.size(),cfg.get("limit",replay.size()).asUInt());
    for(int rep=-1;rep<repeats;++rep){
      for(size_t mi=0;mi<methods.size();++mi){
        const auto&label=methods[(mi+std::max(0,rep))%methods.size()];auto method=target_arm(label);
        fused_policy(method);run.pca=&pca;if(dimension_kind==1){if(!prefixes.count(dimension_p))prefixes.emplace(dimension_p,pca.prefix(dimension_p));run.pca=&prefixes.at(dimension_p);}Agg all;Json::Value rows(Json::arrayValue);
        const size_t count=rep<0?std::min<size_t>(20,limit):limit;
        for(size_t i=0;i<count;++i){
          const auto&plan=replay[Json::ArrayIndex(i)];size_t qi=plan["query_id"].asUInt();
          if(qi>=q.n()||qi>=gt.size()||entry.at(qi)<0||gt[qi].size()<100)throw std::runtime_error("query/GT/entry mismatch");
          fused_policy(method);set_plateau_policy(dimension_kind==1?"heap_full":dimension_kind==2?"never":method);if(dimension_hard&&!plan.get("compressed",true).asBool()){dimension_kind=0;set_plateau_policy("never");}if(plateau_hard_only&&!plan.get("compressed",true).asBool())set_plateau_policy("never");if(method=="plain"||method=="darth"||method=="laet")set_plateau_policy("never");Result r;
          if(method=="plain")r=run.hnsw(qi,target_ef,false);
          else if(method!="darth"&&method!="laet")r=run.hnsw(qi,target_cap(plan["ef"].asInt()),method=="current"&&plan.get("compressed",true).asBool());
          else if(method=="darth")r=run.hnsw(qi,target_ef,false,&darth);
          else r=run.laet(qi,laet,target_alpha,cfg.get("laet_warmup",1260).asInt());
          all.add(r);Json::Value row;row["query_id"]=Json::UInt64(qi);row["recall"]=r.returned_recall;row["ndc"]=r.ndc;row["search_ndc"]=r.search_ndc;row["projection_ndc"]=r.projection_ndc;row["trigger_exact"]=Json::UInt64(r.trigger_exact);row["trigger_step"]=Json::UInt64(r.trigger_step);row["estimates"]=Json::UInt64(r.pca);row["exact"]=Json::UInt64(r.full);row["ef"]=(method=="plain"||method=="darth")?target_ef:target_cap(plan["ef"].asInt());if(method=="laet")row["ef"]=Json::nullValue;row["ms"]=r.ms;row["validation_ms_excluded"]=r.validation_ms;rows.append(row);
        }
        if(rep>=0){Json::Value arm;arm["method"]=label;arm["repeat"]=rep;arm["overall"]=jout(all);arm["per_query"]=rows;output["runs"].append(arm);}
        std::cerr<<"[arm] repeat="<<rep<<" method="<<method<<" n="<<count<<" recall="<<all.rr/all.n<<" ndc="<<all.ndc/all.n<<" ms="<<all.ms/all.n<<'\n';
        std::ofstream out(argv[2]);Json::StreamWriterBuilder b;b["indentation"]=" ";out<<Json::writeString(b,output)<<'\n';
      }
    }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
