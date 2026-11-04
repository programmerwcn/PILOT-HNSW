
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
// Reuse the exact published probe feature computation, without its driver.
#define main historical_deep_features_main
#include "pilot_resume_fast_deep_probe.cpp"
#undef main
#include <chrono>
#include <unordered_set>

extern "C" {
typedef void* BoosterHandle;
int LGBM_BoosterCreateFromModelfile(const char*,int*,BoosterHandle*);
int LGBM_BoosterPredictForMat(BoosterHandle,const void*,int,int32_t,int32_t,int,int,int,int,const char*,int64_t*,double*);
int LGBM_BoosterFree(BoosterHandle);
const char* LGBM_GetLastError();
}
using Timer=std::chrono::steady_clock;
using Item=std::pair<float,uint32_t>;
struct Learned {
 BoosterHandle h=nullptr;int outputs;
 Learned(const std::string&p,int n=1):outputs(n){int it;if(LGBM_BoosterCreateFromModelfile(p.c_str(),&it,&h))throw std::runtime_error(LGBM_GetLastError());}
 ~Learned(){if(h)LGBM_BoosterFree(h);}
 std::vector<double> predict(const std::vector<double>&x)const{
  std::vector<double>r(outputs);int64_t n;
  if(LGBM_BoosterPredictForMat(h,x.data(),1,1,x.size(),1,0,0,-1,"num_threads=1",&n,r.data())||n!=outputs)throw std::runtime_error("LightGBM prediction shape/error");return r;
 }
};
Json::Value readConfig(const std::string&p){std::ifstream f(p);Json::Value v;f>>v;if(!f)throw std::runtime_error("read "+p);return v;}
double elapsed(Timer::time_point t){return std::chrono::duration<double,std::milli>(Timer::now()-t).count();}

struct TimedSearch { std::priority_queue<Item> top;double ndc=0,projection_ndc=0;uint64_t exact=0,pca_dims=0,estimates=0,rejects=0,trigger_exact=0,trigger_step=0;double ms=0; };
// Included after the original Deep monitor types and TimedSearch definition.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
struct DeepMapped {
 int fd;size_t bytes;const float*data;
 explicit DeepMapped(const std::string&path){fd=open(path.c_str(),O_RDONLY);struct stat st{};
  if(fd<0||fstat(fd,&st))throw std::runtime_error("mmap open "+path);bytes=st.st_size;
  auto p=mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd,0);if(p==MAP_FAILED)throw std::runtime_error("mmap failed");data=(const float*)p;}
 ~DeepMapped(){munmap((void*)data,bytes);close(fd);}
 void prewarm()const{volatile float sum=0;for(size_t i=0;i<bytes/4;i+=1024)sum+=data[i];}
};
struct DeepPcaTable {
 DeepMapped scores,norms;const PcaModel&model;
 DeepPcaTable(const std::string&s,const std::string&n,const PcaModel&m):scores(s),norms(n),model(m){
  if(scores.bytes!=m.n*m.dpca*4||norms.bytes!=m.n*4||m.dim!=96||m.dpca<32)throw std::runtime_error("Deep PCA shapes");}
 void prewarm()const{scores.prewarm();norms.prewarm();}
};
TimedSearch plateauDeep(const HierarchicalNSW<float>&index,const float*q,int ef,ProbeState&state,
                       const DeepPcaTable&pca,const std::vector<float>&qp,float margin,const PilotProbeState* resume=nullptr){
 TimedSearch r;auto start=Timer::now();PlateauGate gate(100);
 if(!resume&&++state.epoch==0){std::fill(state.seen.begin(),state.seen.end(),0);++state.epoch;}
 auto distance=[&](uint32_t id){++r.exact;return index.fstdistfunc_(q,index.getDataByInternalId(id),index.dist_func_param_);};
 struct DistanceOnly{bool operator()(const Item&a,const Item&b)const{return a.first>b.first;}};
 std::priority_queue<Item,std::vector<Item>,DistanceOnly>cand;
 float lower;
 if(resume){resume->restore(ef,r.top,cand);lower=r.top.top().first;}
 else{
 uint32_t entry=index.enterpoint_node_;float best=distance(entry);
 for(int level=index.maxlevel_;level>0;--level){bool changed=true;while(changed){changed=false;
 auto*l=reinterpret_cast<unsigned*>(index.get_linklist(entry,level));auto*ids=reinterpret_cast<tableint*>(l+1);
 for(unsigned j=0;j<index.getListCount(l);++j){float d=distance(ids[j]);if(d<best){best=d;entry=ids[j];changed=true;}}}}
 lower=distance(entry);cand.emplace(lower,entry);r.top.emplace(lower,entry);state.seen[entry]=state.epoch;
 }
 gate.observe(lower,true,lower);uint64_t step=0;float qr=0;bool projected=false;const unsigned dp=dimension_kind==1?dimension_p:32;if(dp>32)throw std::runtime_error("Deep executor prefix exceeds shared query projection");
 while(!cand.empty()){auto[cd,id]=cand.top();if(r.top.size()>=size_t(ef)&&cd>lower)break;cand.pop();++step;
  auto*l=reinterpret_cast<unsigned*>(index.get_linklist0(id));auto*ids=reinterpret_cast<tableint*>(l+1);
  for(unsigned j=0;j<index.getListCount(l);++j){uint32_t node=ids[j];if(state.seen[node]==state.epoch)continue;state.seen[node]=state.epoch;
   if(gate.check(r.top.size()>=size_t(ef),lower,r.exact,step)){
    if(!projected){if(qp.size()!=32)throw std::runtime_error("missing shared projection");
     for(unsigned d=0;d<96;++d){float z=q[d]-pca.model.mu[d];qr+=z*z;}for(unsigned c=0;c<dp;++c)qr-=qp[c]*qp[c];qr=std::max(0.f,qr);projected=true;}
    const float*bx=pca.scores.data+uint64_t(node)*pca.model.dpca;
    float kept=0;for(int c=0;c<int(dp);++c)kept+=bx[c]*bx[c];
    float estimate=std::max(0.f,pca.norms.data[node]-kept)+qr;
    for(int c=0;c<int(dp);++c){float z=bx[c]-qp[c];estimate+=z*z;}++r.estimates;r.pca_dims+=dp;
    if(estimate>=lower*(dimension_kind==1?dimension_margin*dimension_margin:margin*margin)){++r.rejects;continue;}
   }
   float d;if(dimension_kind==2&&r.top.size()>=size_t(ef)){bool rejected;d=dimension_fused(q,reinterpret_cast<const float*>(index.getDataByInternalId(node)),96,lower,r.pca_dims,rejected);if(rejected){++r.rejects;continue;}}else d=distance(node);bool admitted=r.top.size()<size_t(ef)||d<lower;gate.observe(d,admitted,lower);
   if(admitted){cand.emplace(d,node);r.top.emplace(d,node);if(r.top.size()>size_t(ef))r.top.pop();lower=r.top.top().first;}
  }
 }
 while(r.top.size()>100)r.top.pop();
 // The saved graphs have identity labels; explicitly return external labels.
 decltype(r.top) external;while(!r.top.empty()){auto p=r.top.top();r.top.pop();external.emplace(p.first,index.getExternalLabel(p.second));}r.top=std::move(external);
 r.trigger_exact=gate.trigger_exact;r.trigger_step=gate.trigger_step;r.ndc=r.exact+r.pca_dims/96.;r.ms=elapsed(start);return r;
}

TimedSearch terminate(const HierarchicalNSW<float>&index,const float*q,const Learned&model,bool laet,ProbeState&state){
 TimedSearch r;const auto start=Timer::now();
 if(++state.epoch==0){std::fill(state.seen.begin(),state.seen.end(),0);++state.epoch;}
 auto distance=[&](uint32_t id){++r.ndc;return index.fstdistfunc_(q,index.getDataByInternalId(id),index.dist_func_param_);};
 uint32_t entry=index.enterpoint_node_;float best=distance(entry);
 for(int level=index.maxlevel_;level>0;--level){bool changed=true;while(changed){changed=false;auto*l=reinterpret_cast<unsigned*>(index.get_linklist(entry,level));auto*ids=reinterpret_cast<tableint*>(l+1);for(unsigned j=0;j<index.getListCount(l);++j){float d=distance(ids[j]);if(d<best){best=d;entry=ids[j];changed=true;}}}}
 const uint64_t hierarchy=r.ndc;const float first=distance(entry);
 struct DistanceOnly {bool operator()(const Item&a,const Item&b)const{return a.first>b.first;}};
 std::priority_queue<Item,std::vector<Item>,DistanceOnly>cand;std::priority_queue<Item>efheap;
 cand.emplace(first,entry);efheap.emplace(first,entry);r.top.emplace(first,entry);state.seen[entry]=state.epoch;
 float lower=first;uint64_t inserts=1,step=0,next=1979,budget=0;bool stop=false;
 while(!cand.empty()&&!stop){auto[cd,cur]=cand.top();cand.pop();if(efheap.size()>=size_t(target_ef)&&cd>lower)break;
  auto*l=reinterpret_cast<unsigned*>(index.get_linklist0(cur));auto*ids=reinterpret_cast<tableint*>(l+1);
  for(unsigned j=0;j<index.getListCount(l);++j){uint32_t id=ids[j];if(state.seen[id]==state.epoch)continue;state.seen[id]=state.epoch;float d=distance(id);
   if(efheap.size()<size_t(target_ef)||d<lower){cand.emplace(d,id);efheap.emplace(d,id);if(efheap.size()>size_t(target_ef))efheap.pop();lower=efheap.top().first;
    if(r.top.size()<100||d<r.top.top().first){r.top.emplace(d,id);if(r.top.size()>100)r.top.pop();++inserts;}}
   const uint64_t base_ndc=r.ndc-hierarchy;
   if(r.top.size()>=100&&((!laet&&base_ndc>=next)||(laet&&!budget&&base_ndc>=241))){
    auto h=r.top;std::vector<double>ds;while(!h.empty()){ds.push_back(h.top().first);h.pop();}std::sort(ds.begin(),ds.end());std::vector<double>x;
    if(laet){x={first,ds[0],ds[9],first>0?ds[0]/first:-1,first>0?ds[9]/first:-1};for(int k=0;k<96;++k)x.push_back(q[k]);}
    else{double mean=std::accumulate(ds.begin(),ds.end(),0.0)/100,var=0;for(double v:ds)var+=(v-mean)*(v-mean);x={double(step),double(base_ndc),double(inserts),first,ds[0],mean,ds[99],var/100,ds[49],ds[24],ds[74]};}
    double p=model.predict(x)[0];if(laet)budget=std::min<uint64_t>(120000,std::max<uint64_t>(241,uint64_t(std::nearbyint(target_alpha*std::pow(2.,p)))));
    else{p=std::clamp(p,0.,1.);if(p>=target_stop){stop=true;break;}next=base_ndc+std::max(1,int(396+(1979-396)*(target_stop-p)));}
   }
   if(laet&&budget&&base_ndc>=budget){stop=true;break;}
  }++step;
 }r.ms=elapsed(start);return r;
}

int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: reported_deep CONFIG.json");auto cfg=readConfig(argv[1]);auto plan=readConfig(cfg["plan"].asString());
 std::string qp=cfg["query"].asString(),gp=cfg["gt"].asString();Matrix<float>queries(qp.data());Matrix<unsigned>gt(gp.data());
 L2Space space(96);std::cerr<<"[load] Deep100M index\n";HierarchicalNSW<float>index(&space,cfg["index"].asString(),false);ProbeState state(index.cur_element_count);
 PcaModel pca(cfg["pca_model"].asString());DeepPcaTable table(cfg["pca_scores"].asString(),cfg["pca_norms"].asString(),pca);table.prewarm();Learned tier(cfg["tier_model"].asString(),3);
 std::vector<std::unique_ptr<Learned>>darth,laet;
 for(int f=0;f<5;++f){darth.emplace_back(new Learned(cfg["darth_dir"].asString()+"/darth_fold"+std::to_string(f)+".txt"));laet.emplace_back(new Learned(cfg["laet_dir"].asString()+"/laet_fold"+std::to_string(f)+".txt"));}
 std::vector<std::string>methods={"current","never","heap_full","stall128","combined"};if(cfg.isMember("methods")){methods.clear();for(auto&m:cfg["methods"])methods.push_back(m.asString());}Json::Value output;output["config"]=cfg;
 const size_t limit=std::min<size_t>(plan.size(),cfg.get("limit",1000).asUInt());
 set_plateau_policy("never");
 for(size_t i=0;i<limit;++i){const auto item=plan[Json::ArrayIndex(i)];uint32_t qi=item["query_id"].asUInt();const float*q=queries.data+size_t(qi)*96;
  Metric metric;metric.clear();auto ref=index.searchKnnNDCExplicitEf(const_cast<float*>(q),100,item["ef"].asInt(),0,&metric);
  auto got=plateauDeep(index,q,item["ef"].asInt(),state,table,{},1.1f);
  if(got.ndc!=metric.ndc+metric.hierarchical_ndc||got.top.size()!=ref.size())throw std::runtime_error("Deep plain NDC fidelity failed");
  while(!ref.empty()){if(ref.top().second!=got.top.top().second||ref.top().first!=got.top.top().first)throw std::runtime_error("Deep plain result fidelity failed");ref.pop();got.top.pop();}
 }output["plain_fidelity_queries"]=Json::UInt64(limit);std::cerr<<"[fidelity] "<<limit<<" queries exact IDs/distances/NDC match\n";

 for(int rep=-1;rep<cfg.get("repeats",5).asInt();++rep){for(size_t mi=0;mi<methods.size();++mi){const auto&label=methods[(mi+std::max(0,rep))%methods.size()];auto m=target_arm(label);fused_policy(m);set_plateau_policy(dimension_kind==1?"heap_full":dimension_kind==2?"never":m);Json::Value rows(Json::arrayValue);double ms=0,recall=0,ndc=0;const size_t count=rep<0?std::min<size_t>(20,limit):limit;
  for(size_t i=0;i<count;++i){auto item=plan[Json::ArrayIndex(i)];fused_policy(m);set_plateau_policy(dimension_kind==1?"heap_full":dimension_kind==2?"never":m);if(plateau_hard_only&&item["tier"].asInt()!=2)set_plateau_policy("never");if(dimension_hard&&item["tier"].asInt()!=2){dimension_kind=0;set_plateau_policy("never");}const uint32_t qi=item["query_id"].asUInt();const float*q=queries.data+size_t(qi)*96;TimedSearch r;double monitor_ms=0,monitor_ndc=0;int predicted=-1;std::vector<float>qp32;PilotProbeState resume;
   if(m=="darth"||m=="laet")r=terminate(index,q,*(m=="darth"?darth:laet)[item["fold"].asUInt()],m=="laet",state);
   else{
    if(m!="plain"&&m!="system_replay"){
     auto t=Timer::now();auto a=routing(index,q,96,state,&resume);monitor_ndc=a[11]+target_hierarchy+32;a.resize(16);auto g=geometry(q,96);a.insert(a.end(),g.begin(),g.end());auto p=pca.project32(q);qp32=p;a.insert(a.end(),p.begin(),p.end());std::vector<double>x(a.begin(),a.end());auto y=tier.predict(x);predicted=std::max_element(y.begin(),y.end())-y.begin();monitor_ms=elapsed(t);
     if(predicted!=item["tier"].asInt())throw std::runtime_error("online classifier mismatch at query "+std::to_string(qi));
    }
    if(m=="plain"){
    auto t=Timer::now();Metric metric;metric.clear();auto top=index.searchKnnNDCExplicitEf(const_cast<float*>(q),100,m=="plain"?target_ef:target_cap(item["ef"].asInt()),0,&metric);
    r.ms=elapsed(t);r.ndc=metric.ndc+metric.hierarchical_ndc;while(!top.empty()){r.top.emplace(top.top().first,top.top().second);top.pop();}
    }else{r=plateauDeep(index,q,target_cap(item["ef"].asInt()),state,table,qp32,cfg.get("margin",1.1).asFloat(),resume.evaluated.empty()?nullptr:&resume);}

   }
   std::unordered_set<unsigned>truth(gt.data+size_t(qi)*gt.d,gt.data+size_t(qi)*gt.d+100);int hit=0;while(!r.top.empty()){hit+=truth.count(r.top.top().second);r.top.pop();}
   Json::Value row;row["query_id"]=qi;row["recall"]=hit/100.;row["ndc"]=r.ndc+monitor_ndc;row["search_ndc"]=r.ndc;row["monitor_base_ndc"]=monitor_ndc;row["ef"]=(m=="plain"||m=="darth"||m=="laet")?target_ef:target_cap(item["ef"].asInt());row["trigger_exact"]=Json::UInt64(r.trigger_exact);row["trigger_step"]=Json::UInt64(r.trigger_step);row["estimates"]=Json::UInt64(r.estimates);row["rejects"]=Json::UInt64(r.rejects);row["exact"]=Json::UInt64(r.exact);row["pca_dims"]=Json::UInt64(r.pca_dims);row["search_ms"]=r.ms;row["monitor_ms"]=monitor_ms;row["ms"]=r.ms+monitor_ms;row["predicted_tier"]=predicted;row["probe_cached_distances"]=Json::UInt64(resume.evaluated.size());row["probe_reused"]=!resume.evaluated.empty();rows.append(row);ms+=r.ms+monitor_ms;recall+=hit/100.;ndc+=r.ndc;
  }
  std::cerr<<"[done] "<<m<<" repeat="<<rep<<" recall="<<recall/count<<" ndc="<<ndc/count<<" ms="<<ms/count<<'\n';
  if(rep>=0){Json::Value arm;arm["method"]=label;arm["repeat"]=rep;arm["per_query"]=rows;output["runs"].append(arm);}std::ofstream out(cfg["output"].asString());out<<output<<'\n';
 }}
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
