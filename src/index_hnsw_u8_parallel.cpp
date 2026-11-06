// Checkpointable parallel HNSW builder specialized for BIGANN uint8 u8bin.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <filesystem>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"

using hnswlib::HierarchicalNSW;
using hnswlib::L2SpaceU8;

struct Args {
    std::string data, index, checkpoint, log;
    int M = 16, efc = 100, threads = 16, checkpoint_every = 10'000'000;
    size_t seed = 2025;
    bool resume = false;
};

static Args parse(int argc, char** argv) {
    Args a; enum { CKPT = 1000, CKPT_EVERY, RESUME };
    option opts[]={{"log-file",required_argument,nullptr,'l'},{"help",no_argument,nullptr,'h'},{"data",required_argument,nullptr,'d'},{"index",required_argument,nullptr,'i'},
      {"M",required_argument,nullptr,'m'},{"ef-construction",required_argument,nullptr,'e'},
      {"threads",required_argument,nullptr,'j'},{"seed",required_argument,nullptr,'r'},
      {"checkpoint",required_argument,nullptr,CKPT},{"checkpoint-every",required_argument,nullptr,CKPT_EVERY},
      {"resume",no_argument,nullptr,RESUME},{nullptr,0,nullptr,0}};
    int c; while((c=getopt_long(argc,argv,"d:i:m:e:j:r:l:h",opts,nullptr))!=-1) switch(c){
      case 'h':std::cout<<"Usage: index_uint8 --data FILE --index FILE --log-file FILE [--threads N --M 16 --ef-construction 100 --seed 2025]\n";std::exit(0);
      case 'l':a.log=optarg;break;
      case 'd':a.data=optarg;break; case 'i':a.index=optarg;break; case 'm':a.M=atoi(optarg);break;
      case 'e':a.efc=atoi(optarg);break; case 'j':a.threads=std::max(1,atoi(optarg));break;
      case 'r':a.seed=strtoull(optarg,nullptr,10);break; case CKPT:a.checkpoint=optarg;break;
      case CKPT_EVERY:a.checkpoint_every=std::max(0,atoi(optarg));break; case RESUME:a.resume=true;break;
      default: std::exit(2);
    }
    if(a.data.empty()||a.index.empty()||a.log.empty()||a.M<=0||a.efc<=0) { std::cerr<<"required: --data FILE --index FILE --log-file FILE (positive M/ef-construction)\n";std::exit(2); }
    if(a.checkpoint.empty()) a.checkpoint=a.index+".checkpoint";
    return a;
}

static void header(const std::string& path,size_t& n,size_t& d){
    std::ifstream f(path,std::ios::binary);uint32_t nn=0,dd=0;f.read((char*)&nn,4);f.read((char*)&dd,4);
    if(!f||dd!=128)throw std::runtime_error("invalid SIFT u8bin header");n=nn;d=dd;
    f.seekg(0,std::ios::end);if((size_t)f.tellg()!=8+n*d)throw std::runtime_error("invalid u8bin size");
}

int main(int argc,char** argv){
    Args a=parse(argc,argv);
    if(!std::filesystem::is_regular_file(a.data))throw std::runtime_error("input file does not exist");
    for(const auto&p:{a.index,a.log,a.checkpoint}){
        auto parent=std::filesystem::path(p).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
    }
    auto absolute=[](const std::string&p){return std::filesystem::absolute(p).lexically_normal();};
    if(absolute(a.data)==absolute(a.index)||absolute(a.data)==absolute(a.log)||absolute(a.index)==absolute(a.log))
        throw std::runtime_error("data, index and log must use distinct paths");
    if(!std::freopen(a.log.c_str(),"a",stderr))throw std::runtime_error("cannot open log file");
    size_t n=0,d=0;header(a.data,n,d);L2SpaceU8 space(d);
    std::unique_ptr<HierarchicalNSW<int>> index;int resume_from=1;
    auto started=std::chrono::steady_clock::now();
    if(a.resume){
        std::ifstream st(a.checkpoint+".state");size_t sn,sm,se,ss;st>>resume_from>>sn>>sm>>se>>ss;
        if(!st||sn!=n||sm!=(size_t)a.M||se!=(size_t)a.efc||ss!=a.seed)
            throw std::runtime_error("checkpoint state missing or incompatible");
        std::cerr<<"[resume] loading "<<a.checkpoint<<" at prefix "<<resume_from<<"\n";
        index.reset(new HierarchicalNSW<int>(&space,a.checkpoint,false,n));
        if(index->cur_element_count!=n)throw std::runtime_error("checkpoint lacks prepared points");
    }else{
        std::cerr<<"[load] "<<n<<" x "<<d<<" uint8\n";
        std::unique_ptr<Matrix<uint8_t>> x(new Matrix<uint8_t>(a.data.data(),true,(int)n));
        index.reset(new HierarchicalNSW<int>(&space,n,a.M,a.efc,a.seed));
        for(size_t i=0;i<n;++i){index->preparePointWithId(x->data+i*d,i,i);
            if((i+1)%1'000'000==0)std::cerr<<"[prepare] "<<i+1<<'/'<<n<<'\n';}
        index->initializePreparedEntryPoint(0);x.reset();
    }
    auto save=[&](int prefix){
        std::string it=a.checkpoint+".tmp",st=a.checkpoint+".state.tmp";
        std::cerr<<"[checkpoint] "<<prefix<<'/'<<n<<'\n';index->saveIndex(it);
        if(std::rename(it.c_str(),a.checkpoint.c_str()))throw std::runtime_error("checkpoint rename failed");
        {std::ofstream f(st);f<<prefix<<' '<<n<<' '<<a.M<<' '<<a.efc<<' '<<a.seed<<'\n';f.flush();if(!f)throw std::runtime_error("state write failed");}
        if(std::rename(st.c_str(),(a.checkpoint+".state").c_str()))throw std::runtime_error("state rename failed");
    };
    std::atomic<int> done{resume_from};std::atomic<bool> failed{false};std::exception_ptr error;std::mutex em;
    int chunk=a.checkpoint_every>0?a.checkpoint_every:(int)n;
    for(int begin=resume_from;begin<(int)n;begin+=chunk){int end=std::min<int>(n,begin+chunk);
#pragma omp parallel for schedule(dynamic,64) num_threads(a.threads)
        for(int i=begin;i<end;++i){if(failed.load(std::memory_order_relaxed))continue;try{
            index->connectPreparedPoint((hnswlib::tableint)i);int cur=done.fetch_add(1)+1;
            if(cur%100000==0){
#pragma omp critical(progress)
                {double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                 double rate=(cur-resume_from)/std::max(sec,1e-9);std::cerr<<"[connect] "<<cur<<'/'<<n
                 <<" rate="<<std::fixed<<std::setprecision(0)<<rate<<"/s eta="<<(n-cur)/std::max(rate,1e-9)/3600<<"h\n";}
            }}catch(...){failed=true;std::lock_guard<std::mutex>g(em);if(!error)error=std::current_exception();}}
        if(error)std::rethrow_exception(error);if(a.checkpoint_every>0)save(end);
    }
    std::cerr<<"[save] "<<a.index<<'\n';index->saveIndex(a.index);return 0;
}
