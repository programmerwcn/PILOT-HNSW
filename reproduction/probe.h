#pragma once
// Portable template of the retained Deep/SIFT zero-anchor probe recipe.
// Parameterized step count is shared by offline features and online inference.
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <limits>
#include "../src/utils.h"
#include "../src/hnswlib/hnswlib.h"
#include "../benchmarks/pilot_probe_state.h"
namespace pilot {
using hnswlib::HierarchicalNSW;
using hnswlib::tableint;
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

template<class Dist,class Scalar>
tableint descend(const HierarchicalNSW<Dist>& index,const Scalar*q,uint64_t& calls){
    tableint cur=index.enterpoint_node_;float current=(++calls,index.fstdistfunc_(q,index.getDataByInternalId(cur),index.dist_func_param_));
    for(int level=index.maxlevel_;level>0;--level){bool changed=true;while(changed){changed=false;
        auto*links=reinterpret_cast<unsigned*>(index.get_linklist(cur,level));const int n=index.getListCount(links);
        auto*ids=reinterpret_cast<tableint*>(links+1);for(int i=0;i<n;++i){const float d=(++calls,index.fstdistfunc_(q,index.getDataByInternalId(ids[i]),index.dist_func_param_));
            if(d<current){current=d;cur=ids[i];changed=true;}}}}return cur;
}

struct ProbeState{explicit ProbeState(size_t n):seen(n,0){}std::vector<uint32_t>seen;uint32_t epoch=0;};

template<class Dist,class Scalar>
std::vector<float> routing(const HierarchicalNSW<Dist>&index,const Scalar*q,size_t dim,ProbeState&state,PilotProbeState& resume,uint64_t& hierarchy,size_t steps){
    using Pair=std::pair<float,uint32_t>;std::priority_queue<Pair>top;
    std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>>cand;
    if(++state.epoch==0){std::fill(state.seen.begin(),state.seen.end(),0);state.epoch=1;}
    const tableint entry=descend(index,q,hierarchy);const float first=index.fstdistfunc_(q,index.getDataByInternalId(entry),index.dist_func_param_);
    top.emplace(first,entry);cand.emplace(first,entry);state.seen[entry]=state.epoch;resume.record(first,entry);
    uint64_t ndis=1,ninserts=1,checks=0,new_nodes=1,replacements=0;size_t expanded=0,last_best_step=0;
    float best=first;
    while(!cand.empty()&&expanded<steps){auto[current_dist,current]=cand.top();cand.pop();++expanded;resume.expand(current);
        auto*links=reinterpret_cast<unsigned*>(index.get_linklist0(current));const uint32_t n=index.getListCount(links);
        auto*ids=reinterpret_cast<tableint*>(links+1);for(uint32_t i=0;i<n;++i){++checks;const uint32_t node=ids[i];
            if(state.seen[node]==state.epoch)continue;state.seen[node]=state.epoch;++new_nodes;
            const float d=index.fstdistfunc_(q,index.getDataByInternalId(node),index.dist_func_param_);++ndis;resume.record(d,node);
            if(top.size()<200||d<top.top().first){cand.emplace(d,node);top.emplace(d,node);++ninserts;
                if(top.size()>200){top.pop();++replacements;}if(d<best){best=d;last_best_step=expanded;}}}}
    auto topcopy=top;std::vector<Pair>nearest;nearest.reserve(topcopy.size());while(!topcopy.empty()){nearest.push_back(topcopy.top());topcopy.pop();}
    std::sort(nearest.begin(),nearest.end());std::vector<float>dists;dists.reserve(nearest.size());
    for(auto[d,id]:nearest)dists.push_back(std::sqrt(d));while(dists.size()<10)dists.push_back(dists.empty()?0:dists.back());
    auto ccopy=cand;std::vector<float>frontier;frontier.reserve(std::min<size_t>(100,ccopy.size()));
    while(!ccopy.empty()&&frontier.size()<100){frontier.push_back(std::sqrt(ccopy.top().first));ccopy.pop();}
    std::sort(frontier.begin(),frontier.end());if(frontier.empty())frontier=dists;
    const size_t topk=std::min<size_t>(10,dists.size());double avg=std::accumulate(dists.begin(),dists.begin()+topk,0.0)/topk,var=0;
    for(size_t i=0;i<topk;++i)var+=(dists[i]-avg)*(dists[i]-avg);var/=topk;
    const float med=(dists[4]+dists[5])*0.5f,p25=dists[2],p75=dists[7];
    const float frontier_med=frontier[frontier.size()/2];const float contrast=(frontier_med-med)/std::max(med,1e-6f);
    float cone=0;size_t pairs=0;for(size_t a=0;a<topk;++a)for(size_t b=a+1;b<topk;++b){
        const Scalar*xa=reinterpret_cast<const Scalar*>(index.getDataByInternalId(nearest[a].second));
        const Scalar*xb=reinterpret_cast<const Scalar*>(index.getDataByInternalId(nearest[b].second));
        double dot=0,na=0,nb=0;for(size_t j=0;j<dim;++j){const float va=float(xa[j])-float(q[j]),vb=float(xb[j])-float(q[j]);dot+=va*vb;na+=va*va;nb+=vb*vb;}
        if(na>0&&nb>0){cone+=float(dot/std::sqrt(na*nb));++pairs;}}if(pairs)cone/=pairs;
    size_t directed=0,reciprocal_nodes=0;std::array<uint32_t,10>topids{};for(size_t i=0;i<topk;++i)topids[i]=nearest[i].second;
    for(size_t a=0;a<topk;++a){auto*l=reinterpret_cast<unsigned*>(index.get_linklist0(topids[a]));const uint32_t n=index.getListCount(l);auto*ids=reinterpret_cast<tableint*>(l+1);
        bool links_back=false;
        for(uint32_t z=0;z<n;++z)for(size_t b=0;b<topk;++b)if(a!=b&&ids[z]==topids[b]){++directed;links_back=true;}
        if(links_back)++reciprocal_nodes;}
    const float mutual=topk>1?float(directed)/float(topk*(topk-1)):0;
    const float novelty=checks?float(new_nodes)/checks:0;const float detour=expanded?float(last_best_step)/expanded:0;
    const float lid_top=lid(dists,std::min<size_t>(100,dists.size()));
    const float lid_front=lid(frontier,std::min<size_t>(100,frontier.size()));
    const float lid_diff=lid_top-lid_front;
    // The eight slots the SIFT recipe leaves at zero, computed from probe state
    // that is already tracked -- no extra distance computations.
    const float branching=expanded?float(checks)/float(expanded):0;
    const float clustering=topk?float(reciprocal_nodes)/float(topk):0;
    const float replacement_rate=ninserts?float(replacements)/float(ninserts):0;
    const float gap12=(dists[1]-dists[0])/std::max(dists[0],1e-6f);
    const float first_d=std::sqrt(first),best_d=std::sqrt(best);
    const float decay=(first_d-best_d)/std::max(first_d,1e-6f);
    return {contrast,cone,mutual,novelty,dists[0],dists[9],float(avg),float(var),med,p25,p75,
            float(ndis),float(ninserts),first_d,detour,lid_diff,
            branching,clustering,replacement_rate,gap12,lid_top,best_d,decay,lid_front};
}

template<class Scalar>
std::vector<float> geometry(const Scalar* q,uint32_t d){
    double sum=0,sq=0;float mn=std::numeric_limits<float>::max();
    for(uint32_t j=0;j<d;++j){const float x=q[j];sum+=x;sq+=x*x;mn=std::min(mn,x);}
    const double mean=sum/d; const float sd=std::sqrt(std::max(0.0,sq/d-mean*mean));
    return {sd,float(std::sqrt(sq)),mn};
}

} // namespace pilot
