// Portable online entry point: shared probe, clean51 classifier, state reuse,
// and the retained distance/activation kernels. Historical runners stay intact.
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <json/json.h>
#include "probe.h"
#include "../benchmarks/dimension_fused_policy.h"
#include "../benchmarks/plateau_gate.h"
extern "C" {
typedef void* BoosterHandle;
int LGBM_BoosterCreateFromModelfile(const char*,int*,BoosterHandle*);
int LGBM_BoosterGetNumFeature(BoosterHandle,int*);
int LGBM_BoosterPredictForMat(BoosterHandle,const void*,int,int32_t,int32_t,int,int,int,int,const char*,int64_t*,double*);
int LGBM_BoosterFree(BoosterHandle);
const char* LGBM_GetLastError();
}
using Clock=std::chrono::steady_clock;
using Pair=std::pair<float,unsigned>;
using Top=std::priority_queue<Pair>;
using Frontier=std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>>;
struct Bin {
 int fd=-1;size_t bytes=0;void*ptr=MAP_FAILED;uint32_t n=0,d=0;
 Bin(const std::string& path,size_t item){
  fd=open(path.c_str(),O_RDONLY);struct stat st{};
  if(fd<0||fstat(fd,&st)||st.st_size<8)throw std::runtime_error("cannot read binary: "+path);
  bytes=st.st_size;ptr=mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd,0);
  if(ptr==MAP_FAILED)throw std::runtime_error("mmap failed: "+path);
  auto h=static_cast<const uint32_t*>(ptr);n=h[0];d=h[1];
  if(!n||!d||bytes!=8+uint64_t(n)*d*item)throw std::runtime_error("bad binary header/size: "+path);
 }
 ~Bin(){if(ptr!=MAP_FAILED)munmap(ptr,bytes);if(fd>=0)close(fd);}
 template<class T>const T*row(size_t i)const{if(i>=n)throw std::runtime_error("row outside binary");return reinterpret_cast<const T*>(static_cast<const char*>(ptr)+8)+i*d;}
};
struct Model {
 BoosterHandle handle=nullptr;
 Model(const std::string& path){int it,n;
  if(LGBM_BoosterCreateFromModelfile(path.c_str(),&it,&handle))throw std::runtime_error(LGBM_GetLastError());
  if(LGBM_BoosterGetNumFeature(handle,&n)||n!=51)throw std::runtime_error("expected a clean51 classifier");
 }
 ~Model(){if(handle)LGBM_BoosterFree(handle);}
 int predict(const std::vector<float>& x){std::vector<double>v(x.begin(),x.end());double y[3];int64_t count=0;
  if(LGBM_BoosterPredictForMat(handle,v.data(),1,1,51,1,0,0,-1,"num_threads=1",&count,y)||count!=3)throw std::runtime_error("invalid multiclass model");
  return std::max_element(y,y+3)-y;
 }
};
struct PCA {
 Bin mean,components;
 PCA(const Json::Value&cfg):mean(cfg["pca_mean"].asString(),4),components(cfg["pca_components"].asString(),4){
  if(mean.n!=1||mean.d!=components.d||components.n<32)throw std::runtime_error("PCA must include at least 32 components");
 }
 template<class Scalar>std::vector<float>project(const Scalar*q,unsigned count)const{
  if(count>components.n)throw std::runtime_error("PCA prefix outside model");
  std::vector<float>out(count,0);for(unsigned c=0;c<count;++c)for(unsigned j=0;j<mean.d;++j)out[c]+=(float(q[j])-mean.row<float>(0)[j])*components.row<float>(c)[j];return out;
 }
};
struct SearchResult{Top top;double ndc=0;uint64_t partial_dims=0,estimates=0;int tier=-1,ef=0;size_t cached=0;};

template<class Dist,class Scalar>
SearchResult online(const hnswlib::HierarchicalNSW<Dist>& index,const Scalar*q,unsigned d,unsigned k,
 pilot::ProbeState&state,const PCA&pca,Model&model,const Json::Value&cfg,const Bin*scores,const Bin*norms){
 SearchResult r;PilotProbeState resume;uint64_t hierarchy=0;
 auto f=pilot::routing(index,q,d,state,resume,hierarchy,cfg["probe_steps"].asUInt());
 r.ndc=f[11]+hierarchy;r.cached=resume.evaluated.size();f.resize(16);
 auto g=pilot::geometry(q,d);f.insert(f.end(),g.begin(),g.end());
 auto qp=pca.project(q,32);f.insert(f.end(),qp.begin(),qp.end());r.tier=model.predict(f);
 r.ef=cfg["caps"][r.tier].asInt();if(r.ef<int(k)||r.ef>int(index.cur_element_count))throw std::runtime_error("invalid tier cap");
 const std::string executor=cfg.get("executor","exact").asString();
 bool eligible=!cfg.get("hard_only",false).asBool()||r.tier==2;
 fused_policy(executor=="partial"?"prune_"+std::to_string(cfg.get("block",64).asInt())+"_0":"never");
 set_plateau_policy(cfg.get("activation","heap_full").asString());PlateauGate gate(k);
 Frontier cand;resume.restore(r.ef,r.top,cand);float lower=r.top.top().first;
 unsigned prefix=cfg.get("prefix",32).asUInt();float qr=0;bool projected=false;
 auto exact=[&](unsigned id){++r.ndc;return float(index.fstdistfunc_(q,index.getDataByInternalId(id),index.dist_func_param_));};
 uint64_t step=0;
 while(!cand.empty()){
  auto[cd,id]=cand.top();if(r.top.size()>=size_t(r.ef)&&cd>lower)break;cand.pop();++step;
  auto*l=reinterpret_cast<unsigned*>(index.get_linklist0(id));auto*ids=reinterpret_cast<hnswlib::tableint*>(l+1);
  for(unsigned z=0;z<index.getListCount(l);++z){unsigned node=ids[z];if(state.seen[node]==state.epoch)continue;state.seen[node]=state.epoch;
   bool active=eligible&&gate.check(r.top.size()>=size_t(r.ef),lower,uint64_t(r.ndc),step);float distance;
   if(executor=="pca"&&active){
    if(!scores||!norms||prefix>scores->d)throw std::runtime_error("missing PCA base assets");
    if(!projected){qp=pca.project(q,prefix);for(unsigned j=0;j<d;++j){float v=float(q[j])-pca.mean.row<float>(0)[j];qr+=v*v;}for(float v:qp)qr-=v*v;qr=std::max(0.f,qr);projected=true;}
    auto label=index.getExternalLabel(node);const float*b=scores->row<float>(label);float residual=norms->row<float>(label)[0],estimate=qr;
    for(unsigned c=0;c<prefix;++c){residual-=b[c]*b[c];float v=qp[c]-b[c];estimate+=v*v;}
    estimate+=std::max(0.f,residual);r.partial_dims+=prefix;++r.estimates;
    float margin=cfg.get("margin",1.).asFloat();if(estimate>lower*margin*margin)continue;
   }
   if(executor=="partial"&&active){bool rejected;distance=dimension_fused(q,reinterpret_cast<const Scalar*>(index.getDataByInternalId(node)),d,lower,r.partial_dims,rejected);if(rejected)continue;}
   else distance=exact(node);
   bool admitted=r.top.size()<size_t(r.ef)||distance<lower;gate.observe(distance,admitted,lower);
   if(admitted){cand.emplace(distance,node);r.top.emplace(distance,node);if(r.top.size()>size_t(r.ef))r.top.pop();lower=r.top.top().first;}
  }
 }
 while(r.top.size()>k)r.top.pop();Top external;while(!r.top.empty()){auto x=r.top.top();r.top.pop();external.emplace(x.first,index.getExternalLabel(x.second));}r.top=std::move(external);
 r.ndc+=double(r.partial_dims)/d;return r;
}

template<class Dist,class Scalar,class Space>
Json::Value execute(const Json::Value&cfg){
 Bin queries(cfg["query"].asString(),sizeof(Scalar));Space space(queries.d);
 hnswlib::HierarchicalNSW<Dist>index(&space,cfg["index"].asString(),false);
 const unsigned k=cfg.get("k",100).asUInt();if(!k||index.cur_element_count<std::max(10u,k))throw std::runtime_error("index too small for k/probe");
 if(index.data_size_!=queries.d*sizeof(Scalar))throw std::runtime_error("query/index dimensionality mismatch");
 pilot::ProbeState state(index.cur_element_count);const std::string mode=cfg["mode"].asString();
 std::unique_ptr<PCA>pca;std::unique_ptr<Model>model;std::unique_ptr<Bin>scores,norms,gt;
 if(mode=="features"||mode=="pilot"){pca=std::make_unique<PCA>(cfg);if(pca->mean.d!=queries.d)throw std::runtime_error("PCA/query dimensions");}
 if(mode=="pilot"){
  const std::string executor=cfg.get("executor","exact").asString();
  if(executor!="exact"&&executor!="partial"&&executor!="pca")throw std::runtime_error("unknown executor");
  std::string activation=cfg.get("activation","heap_full").asString();
  if(activation!="heap_full"&&activation!="combined128"&&activation!="combined256"&&activation!="stall128"&&activation!="never")throw std::runtime_error("unknown activation");
  if(executor=="partial"){int b=cfg.get("block",64).asInt();if(b!=8&&b!=16&&b!=32&&b!=64&&b!=128)throw std::runtime_error("unsupported partial block");}
  if(executor=="pca"&&(cfg.get("prefix",32).asInt()<1||cfg.get("prefix",32).asUInt()>pca->components.n||cfg.get("margin",1.).asDouble()<=0))throw std::runtime_error("invalid PCA policy");
  if(!cfg.isMember("model_metadata"))throw std::runtime_error("pilot mode requires model_metadata");
  std::ifstream metadata_file(cfg["model_metadata"].asString());Json::Value metadata;metadata_file>>metadata;
  if(!metadata_file||metadata["probe_steps"]!=cfg["probe_steps"]||metadata["dimensions"].asUInt()!=queries.d||metadata["feature_schema"].asString()!="clean51")throw std::runtime_error("model/probe feature configuration mismatch");
  model=std::make_unique<Model>(cfg["model"].asString());
  if(cfg.get("executor","exact").asString()=="pca"){
   scores=std::make_unique<Bin>(cfg["pca_scores"].asString(),4);norms=std::make_unique<Bin>(cfg["pca_norms"].asString(),4);
   if(scores->n!=index.cur_element_count||norms->n!=scores->n||norms->d!=1)throw std::runtime_error("PCA/index shape mismatch");
  }
 }
 if(cfg.isMember("gt")){gt=std::make_unique<Bin>(cfg["gt"].asString(),4);if(gt->n!=queries.n||gt->d<k)throw std::runtime_error("ground-truth shape mismatch");}
 Json::Value output;output["config"]=cfg;output["rows"]=Json::Value(Json::arrayValue);
 std::vector<unsigned>qids;if(cfg.isMember("query_ids")){for(auto&id:cfg["query_ids"])qids.push_back(id.asUInt());}else for(unsigned i=0;i<queries.n;++i)qids.push_back(i);
 if(qids.empty())throw std::runtime_error("empty query selection");
 auto steps=cfg.get("probe_steps",100).asInt();if(steps<=0)throw std::runtime_error("probe_steps must be positive");
 if(mode=="pilot"&&cfg["caps"].size()!=3)throw std::runtime_error("caps must have three elements");
 std::vector<int>efs;if(mode=="grid"){for(auto&e:cfg["ef_grid"])efs.push_back(e.asInt());}else efs.push_back(cfg.get("ef",200).asInt());
 if(efs.empty())throw std::runtime_error("empty ef grid");
 int repeats=cfg.get("repeats",1).asInt();if(repeats<1)throw std::runtime_error("repeats must be positive");
 for(int rep=(mode=="pilot"||mode=="plain")?-1:0;rep<repeats;++rep)for(unsigned qi:qids)for(int ef:efs){
  if(qi>=queries.n)throw std::runtime_error("query ID out of range");const Scalar*q=queries.row<Scalar>(qi);
  auto start=Clock::now();SearchResult r;Json::Value row;row["query_id"]=qi;row["repeat"]=rep;
  if(mode=="features"){
   PilotProbeState resume;uint64_t hierarchy=0;auto f=pilot::routing(index,q,queries.d,state,resume,hierarchy,steps);r.ndc=f[11]+hierarchy;f.resize(16);
   auto g=pilot::geometry(q,queries.d);f.insert(f.end(),g.begin(),g.end());auto qp=pca->project(q,32);f.insert(f.end(),qp.begin(),qp.end());
   row["features"]=Json::Value(Json::arrayValue);for(float v:f){if(!std::isfinite(v))throw std::runtime_error("non-finite feature");row["features"].append(v);}
  }else if(mode=="pilot")r=online(index,q,queries.d,k,state,*pca,*model,cfg,scores.get(),norms.get());
  else if(mode=="grid"||mode=="plain"){
   if(ef<int(k)||ef>int(index.cur_element_count))throw std::runtime_error("ef outside [k,N]");
   Metric metric;metric.clear();auto found=index.searchKnnNDCExplicitEf(const_cast<Scalar*>(q),k,ef,0,&metric);
   while(!found.empty()){r.top.emplace(float(found.top().first),found.top().second);found.pop();}r.ndc=metric.ndc+metric.hierarchical_ndc;r.ef=ef;
  }else throw std::runtime_error("mode must be features, grid, plain or pilot");
  row["ms"]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();row["ndc"]=r.ndc;row["ef"]=r.ef;row["tier"]=r.tier;row["probe_cached_distances"]=Json::UInt64(r.cached);
  row["estimates"]=Json::UInt64(r.estimates);row["partial_dims"]=Json::UInt64(r.partial_dims);
  if(mode!="features"){
   std::set<unsigned>truth;if(gt)for(unsigned j=0;j<k;++j)truth.insert(gt->row<uint32_t>(qi)[j]);unsigned hit=0;
   std::vector<unsigned>found;while(!r.top.empty()){unsigned id=r.top.top().second;r.top.pop();found.push_back(id);hit+=truth.count(id);}
   std::reverse(found.begin(),found.end());for(auto id:found)row["neighbors"].append(id);if(gt)row["recall"]=double(hit)/k;
  }
  if(rep>=0)output["rows"].append(row);
 }
 return output;
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--help"){std::cout<<"Usage: pilot_search CONFIG.json\nModes: features, grid, plain, pilot. Paths are relative to CONFIG.json.\n";return 0;}
 if(argc!=2)throw std::runtime_error("Usage: pilot_search CONFIG.json");
 auto path=std::filesystem::absolute(argv[1]);std::ifstream in(path);Json::Value cfg;in>>cfg;if(!in)throw std::runtime_error("invalid JSON config");
 for(auto key:{"index","query","gt","model","pca_mean","pca_components","pca_scores","pca_norms","model_metadata","output","log"})if(cfg.isMember(key)){
  auto p=std::filesystem::path(cfg[key].asString());if(p.is_relative())p=path.parent_path()/p;cfg[key]=p.lexically_normal().string();
 }
 for(auto key:{"index","query","mode","output","log"})if(!cfg.isMember(key)||cfg[key].asString().empty())throw std::runtime_error(std::string("missing config field: ")+key);
 auto outpath=std::filesystem::path(cfg["output"].asString());auto logpath=std::filesystem::path(cfg["log"].asString());
 if(outpath==logpath)throw std::runtime_error("output and log paths must differ");
 for(auto key:{"index","query","gt","model","pca_mean","pca_components","pca_scores","pca_norms","model_metadata"})if(cfg.isMember(key)){
  if(cfg[key].asString()==outpath.string()||cfg[key].asString()==logpath.string())throw std::runtime_error("output/log must not overwrite inputs");
 }
 std::filesystem::create_directories(outpath.parent_path());std::filesystem::create_directories(logpath.parent_path());
 std::ofstream log(logpath);if(!log)throw std::runtime_error("cannot open log");log<<cfg<<'\n';
 cfg["probe_steps"]=cfg.get("probe_steps",100);hnswlib::logger.setOutputEnabled(false);
 std::string dtype=cfg.get("dtype","float32").asString();Json::Value result;
 if(dtype=="float32")result=execute<float,float,hnswlib::L2Space>(cfg);
 else if(dtype=="uint8")result=execute<int,uint8_t,hnswlib::L2SpaceU8>(cfg);
 else throw std::runtime_error("dtype must be float32 or uint8");
 std::ofstream out(outpath);out<<result<<'\n';if(!out)throw std::runtime_error("cannot write output");log<<"completed rows="<<result["rows"].size()<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
