// Collect the 59-feature, zero-anchor-cost SIFT100M routing recipe.
// The graph probe is hnswlib's bounded routing probe (ef=200, m=100, k=10).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include <jsoncpp/json/json.h>

#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"

using hnswlib::HierarchicalNSW;
using hnswlib::L2SpaceU8;
using hnswlib::tableint;

namespace {
template<class T> T readPod(std::istream& in) {
    T v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if(!in) throw std::runtime_error("short PCA model"); return v;
}

struct PcaModel {
    uint64_t n=0; uint32_t dim=0,dpca=0; std::vector<float> mu,projection;
    explicit PcaModel(const std::string& path) {
        std::ifstream in(path,std::ios::binary); char magic[8]; in.read(magic,8);
        if(!in||std::memcmp(magic,"SPCA100\0",8)) throw std::runtime_error("bad PCA model");
        n=readPod<uint64_t>(in); dim=readPod<uint32_t>(in); dpca=readPod<uint32_t>(in);
        mu.resize(dim); projection.resize(size_t(dim)*dpca);
        in.read(reinterpret_cast<char*>(mu.data()),mu.size()*4);
        in.read(reinterpret_cast<char*>(projection.data()),projection.size()*4);
        if(!in) throw std::runtime_error("short PCA arrays");
    }
    std::vector<float> project32(const uint8_t* q) const {
        const uint32_t keep=std::min<uint32_t>(32,dpca); std::vector<float> out(keep,0);
        for(uint32_t j=0;j<dim;++j){const float z=float(q[j])-mu[j];
            for(uint32_t c=0;c<keep;++c)out[c]+=z*projection[size_t(j)*dpca+c];}
        return out;
    }
};

const std::vector<std::string>& names(){
    static const std::vector<std::string> v={
        "routing_contrast","routing_cone_concentration","routing_mutual_nn_rate","routing_novelty_rate",
        "routing_closest_nn_distance","routing_furthest_nn_distance","routing_avg","routing_var",
        "routing_med","routing_perc25","routing_perc75","routing_ndis","routing_ninserts",
        "routing_first_nn_distance","routing_entry_detour_ratio","routing_lid_topm_vs_frontier",
        "routing_effective_branching_factor","routing_local_clustering_coeff","routing_replacement_rate",
        "routing_gap12_normalized","routing_lid_estimate","routing_current_best_distance",
        "routing_distance_decay_rate","routing_basin_size_lid_method",
        "geom_std","geom_l2_norm","geom_min"};
    return v;
}

float lid(const std::vector<float>& distances,size_t count){
    count=std::min(count,distances.size());if(count<2)return 0;const float rk=std::max(distances[count-1],1e-6f);
    double sum=0;for(size_t i=0;i<count;++i)sum+=std::log(std::max(distances[i],1e-6f)/rk);
    return sum<-1e-9?float(-double(count)/sum):0;
}

tableint descend(const HierarchicalNSW<int>& index,const uint8_t*q){
    tableint cur=index.enterpoint_node_;int current=index.fstdistfunc_(q,index.getDataByInternalId(cur),index.dist_func_param_);
    for(int level=index.maxlevel_;level>0;--level){bool changed=true;while(changed){changed=false;
        auto*links=reinterpret_cast<unsigned*>(index.get_linklist(cur,level));const int n=index.getListCount(links);
        auto*ids=reinterpret_cast<tableint*>(links+1);for(int i=0;i<n;++i){const int d=index.fstdistfunc_(q,index.getDataByInternalId(ids[i]),index.dist_func_param_);
            if(d<current){current=d;cur=ids[i];changed=true;}}}}return cur;
}

struct ProbeState{explicit ProbeState(size_t n):seen(n,0){}std::vector<uint32_t>seen;uint32_t epoch=0;};

std::vector<float> routing(const HierarchicalNSW<int>&index,const uint8_t*q,ProbeState&state){
    using Pair=std::pair<int,uint32_t>;std::priority_queue<Pair>top;
    std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>>cand;
    if(++state.epoch==0){std::fill(state.seen.begin(),state.seen.end(),0);state.epoch=1;}
    const tableint entry=descend(index,q);const int first=index.fstdistfunc_(q,index.getDataByInternalId(entry),index.dist_func_param_);
    top.emplace(first,entry);cand.emplace(first,entry);state.seen[entry]=state.epoch;
    uint64_t ndis=1,ninserts=1,checks=0,new_nodes=1,replacements=0;size_t expanded=0,last_best_step=0;
    int best=first;
    while(!cand.empty()&&expanded<100){auto[current_dist,current]=cand.top();cand.pop();++expanded;
        auto*links=reinterpret_cast<unsigned*>(index.get_linklist0(current));const uint32_t n=index.getListCount(links);
        auto*ids=reinterpret_cast<tableint*>(links+1);for(uint32_t i=0;i<n;++i){++checks;const uint32_t node=ids[i];
            if(state.seen[node]==state.epoch)continue;state.seen[node]=state.epoch;++new_nodes;
            const int d=index.fstdistfunc_(q,index.getDataByInternalId(node),index.dist_func_param_);++ndis;
            if(top.size()<200||d<top.top().first){cand.emplace(d,node);top.emplace(d,node);++ninserts;
                if(top.size()>200){top.pop();++replacements;}if(d<best){best=d;last_best_step=expanded;}}}}
    auto topcopy=top;std::vector<Pair>nearest;nearest.reserve(topcopy.size());while(!topcopy.empty()){nearest.push_back(topcopy.top());topcopy.pop();}
    std::sort(nearest.begin(),nearest.end());std::vector<float>dists;dists.reserve(nearest.size());
    for(auto[d,id]:nearest)dists.push_back(std::sqrt(float(d)));while(dists.size()<10)dists.push_back(dists.empty()?0:dists.back());
    auto ccopy=cand;std::vector<float>frontier;frontier.reserve(std::min<size_t>(100,ccopy.size()));
    while(!ccopy.empty()&&frontier.size()<100){frontier.push_back(std::sqrt(float(ccopy.top().first)));ccopy.pop();}
    std::sort(frontier.begin(),frontier.end());if(frontier.empty())frontier=dists;
    const size_t topk=std::min<size_t>(10,dists.size());double avg=std::accumulate(dists.begin(),dists.begin()+topk,0.0)/topk,var=0;
    for(size_t i=0;i<topk;++i)var+=(dists[i]-avg)*(dists[i]-avg);var/=topk;
    const float med=(dists[4]+dists[5])*0.5f,p25=dists[2],p75=dists[7];
    const float frontier_med=frontier[frontier.size()/2];const float contrast=(frontier_med-med)/std::max(med,1e-6f);
    float cone=0;size_t pairs=0;for(size_t a=0;a<topk;++a)for(size_t b=a+1;b<topk;++b){
        const uint8_t*xa=reinterpret_cast<const uint8_t*>(index.getDataByInternalId(nearest[a].second));
        const uint8_t*xb=reinterpret_cast<const uint8_t*>(index.getDataByInternalId(nearest[b].second));
        double dot=0,na=0,nb=0;for(size_t j=0;j<128;++j){const float va=float(xa[j])-q[j],vb=float(xb[j])-q[j];dot+=va*vb;na+=va*va;nb+=vb*vb;}
        if(na>0&&nb>0){cone+=float(dot/std::sqrt(na*nb));++pairs;}}if(pairs)cone/=pairs;
    size_t directed=0;std::array<uint32_t,10>topids{};for(size_t i=0;i<topk;++i)topids[i]=nearest[i].second;
    for(size_t a=0;a<topk;++a){auto*l=reinterpret_cast<unsigned*>(index.get_linklist0(topids[a]));const uint32_t n=index.getListCount(l);auto*ids=reinterpret_cast<tableint*>(l+1);
        for(uint32_t z=0;z<n;++z)for(size_t b=0;b<topk;++b)if(a!=b&&ids[z]==topids[b])++directed;}
    const float mutual=topk>1?float(directed)/float(topk*(topk-1)):0;
    const float novelty=checks?float(new_nodes)/checks:0;const float detour=expanded?float(last_best_step)/expanded:0;
    const float lid_diff=lid(dists,std::min<size_t>(100,dists.size()))-lid(frontier,std::min<size_t>(100,frontier.size()));
    // The final eight compatibility slots deliberately remain zero: the GIST
    // position-indexing bug made them empirically inert, and clean51 removes them.
    return {contrast,cone,mutual,novelty,dists[0],dists[9],float(avg),float(var),med,p25,p75,
            float(ndis),float(ninserts),std::sqrt(float(first)),detour,lid_diff,0,0,0,0,0,0,0,0};
}

std::vector<float> geometry(const uint8_t* q,uint32_t d){
    double sum=0,sq=0;float mn=255;
    for(uint32_t j=0;j<d;++j){const float x=q[j];sum+=x;sq+=x*x;mn=std::min(mn,x);}
    const double mean=sum/d; const float sd=std::sqrt(std::max(0.0,sq/d-mean*mean));
    return {sd,float(std::sqrt(sq)),mn};
}
} // namespace

int main(int argc,char**argv){
    if(argc!=6){std::cerr<<"usage: "<<argv[0]<<" INDEX QUERY.u8bin PCA_MODEL.bin OUT.json LIMIT\n";return 2;}
    try{
        Matrix<uint8_t> queries(argv[2]); PcaModel pca(argv[3]);
        const size_t limit=std::min<size_t>(queries.n,static_cast<size_t>(std::stoull(argv[5])));
        if(queries.d!=128||pca.dim!=queries.d||pca.dpca<32)throw std::runtime_error("shape mismatch");
        L2SpaceU8 space(queries.d); std::cerr<<"[load] HNSW\n"; HierarchicalNSW<int> index(&space,argv[1],false);
        Json::Value root,feature_names(Json::arrayValue),rows(Json::arrayValue);
        for(const auto& n:names())feature_names.append(n); for(int c=0;c<32;++c)feature_names.append("pca_"+std::to_string(c));
        ProbeState probe(index.cur_element_count);
        for(size_t qi=0;qi<limit;++qi){uint8_t*q=queries.data+qi*queries.d;
            auto f=routing(index,q,probe);auto g=geometry(q,queries.d);f.insert(f.end(),g.begin(),g.end());
            auto p=pca.project32(q);f.insert(f.end(),p.begin(),p.end());if(f.size()!=59)throw std::runtime_error("feature count");
            Json::Value row,values(Json::arrayValue);row["query_index"]=Json::UInt64(qi);
            for(float x:f)values.append(std::isfinite(x)?x:0.0f);row["features"]=values;rows.append(row);
            if((qi+1)%100==0)std::cerr<<"[features] "<<qi+1<<'/'<<limit<<'\n';}
        Json::Value meta;meta["feature_count"]=59;meta["routing_feature_count"]=24;meta["geometry_feature_count"]=3;
        meta["pca_feature_count"]=32;meta["anchor_feature_count"]=0;meta["probe_ef"]=200;meta["probe_m"]=100;meta["probe_k"]=10;
        root["metadata"]=meta;root["feature_names"]=feature_names;root["queries"]=rows;
        std::ofstream out(argv[4]);Json::StreamWriterBuilder wb;wb["indentation"]=" ";std::unique_ptr<Json::StreamWriter>w(wb.newStreamWriter());w->write(root,&out);out<<'\n';
        std::cerr<<"[save] "<<argv[4]<<'\n';
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}return 0;
}
