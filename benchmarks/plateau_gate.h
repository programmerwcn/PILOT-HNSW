#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>

// -1=current historical policy; 0=never; 1=heap-full; 2=stall; 3=combined.
inline int plateau_mode=-1;
inline bool plateau_hard_only=false;
inline unsigned plateau_window=128;
inline void set_plateau_policy(const std::string& name) {
    plateau_hard_only=name.rfind("hard_combined",0)==0;
    plateau_mode=name=="current"?-1:name=="never"?0:name=="heap_full"?1:
                 name=="stall128"?2:3;
    plateau_window=128;
    const auto combined=name.find("combined");
    if(combined!=std::string::npos) {
        const auto suffix=name.substr(combined+8);
        if(!suffix.empty())plateau_window=unsigned(std::stoul(suffix));
    }
}
struct PlateauGate {
    int mode=plateau_mode;
    unsigned window=plateau_window, k;
    uint64_t count=0,last_update=0,trigger_exact=0,trigger_step=0,updates=0;
    bool active=false;
    std::priority_queue<float> topk;
    std::array<float,2048> radius{};
    std::array<unsigned char,2048> inserts{};
    unsigned pos=0,filled=0,insert_sum=0;
    explicit PlateauGate(unsigned kk):k(kk){if(window>radius.size())throw std::invalid_argument("plateau window > 2048");}
    void observe(float distance,bool admitted,float old_radius) {
        if(mode<2||active)return;
        ++count;
        if(topk.size()<k||distance<topk.top()) {
            topk.push(distance);if(topk.size()>k)topk.pop();
            last_update=count;++updates;
        }
        if(filled==window)insert_sum-=inserts[pos];else ++filled;
        inserts[pos]=admitted;insert_sum+=admitted;radius[pos]=old_radius;
        pos=(pos+1)%window;
    }
    bool check(bool full,float r,uint64_t exact,uint64_t step) {
        if(mode<1)return false;
        if(active)return true;
        if(!full)return false;
        bool fire=mode==1;
        if(mode>=2)fire=topk.size()==k&&count-last_update>=window;
        if(fire&&mode==3)fire=filled==window&&double(insert_sum)/window<=.2&&
            (radius[pos]-r)/std::max(radius[pos],1e-20f)<=.001f;
        if(fire){active=true;trigger_exact=exact;trigger_step=step;}
        return active;
    }
};
