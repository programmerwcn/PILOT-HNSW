#pragma once
#include <vector>
#include <queue>
#include <unordered_set>
#include <stdexcept>
// One query's already-computed base-layer distances and expansion history.
// visited marks remain in the runner: restoration performs NO distance calls.
struct PilotProbeState {
 using Pair=std::pair<float,unsigned>;
 std::vector<Pair> evaluated;
 std::unordered_set<unsigned> expanded;
 void record(float d,unsigned id){evaluated.emplace_back(d,id);}
 void expand(unsigned id){if(!expanded.insert(id).second)throw std::runtime_error("duplicate probe expansion");}
 template<class Top,class Frontier> void restore(unsigned ef,Top&top,Frontier&frontier)const{
  if(!ef)throw std::runtime_error("zero ef");
  top=Top{};frontier=Frontier{};
  for(const auto&p:evaluated){top.push(p);if(top.size()>ef)top.pop();}
  for(const auto&p:evaluated)if(!expanded.count(p.second)&&(top.size()<ef||p.first<=top.top().first))frontier.push(p);
 }
};
