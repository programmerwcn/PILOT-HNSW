#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
// Preserve the original float products, double accumulation, pair order and
// float output accumulation. Cache repeated offsets/norms, not feature values.
template<class T,class Lookup,class Neighbors>
float pilotConeCached(const T*q,size_t dim,const Neighbors&nearest,size_t count,Lookup row){
 std::vector<float> offsets(count*dim);std::vector<double> norms(count,0);
 for(size_t a=0;a<count;++a){const T*x=row(nearest[a].second);for(size_t j=0;j<dim;++j){const float v=float(x[j])-float(q[j]);offsets[a*dim+j]=v;norms[a]+=v*v;}}
 float cone=0;size_t pairs=0;
 for(size_t a=0;a<count;++a)for(size_t b=a+1;b<count;++b){double dot=0;for(size_t j=0;j<dim;++j)dot+=offsets[a*dim+j]*offsets[b*dim+j];
 if(norms[a]>0&&norms[b]>0){cone+=float(dot/std::sqrt(norms[a]*norms[b]));++pairs;}}
 return pairs?cone/pairs:0;
}
