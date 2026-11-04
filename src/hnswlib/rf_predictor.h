#pragma once

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <jsoncpp/json/json.h>

namespace hnswlib {

// Lightweight, header-only Random Forest predictor that loads a JSON dump
// produced by bridge/export_rf_to_json.py. Each tree stores parallel flat
// arrays (feature/threshold/left/right/value); leaves have left == -1.
// predict_proba averages the per-tree class-1 leaf probability.
class RandomForestPredictor {
public:
    RandomForestPredictor() = default;

    // Returns true on success; false if the file is missing or malformed.
    bool load_from_json(const std::string& path);

    // Walks every tree and averages the class-1 leaf probability.
    float predict_proba(const float* x) const;

    size_t n_features() const { return n_features_; }
    size_t n_trees() const { return trees_.size(); }
    const std::vector<float>& medians() const { return medians_; }
    const std::vector<std::string>& feature_names() const { return feature_names_; }

    // In-place NaN imputation using stored training medians.
    void apply_medians(float* x) const {
        for (size_t i = 0; i < n_features_ && i < medians_.size(); ++i) {
            if (std::isnan(x[i])) x[i] = medians_[i];
        }
    }

private:
    struct Tree {
        std::vector<int> feature;
        std::vector<float> threshold;
        std::vector<int> left;
        std::vector<int> right;
        std::vector<float> value;
    };

    size_t n_features_ = 0;
    std::vector<std::string> feature_names_;
    std::vector<float> medians_;
    std::vector<Tree> trees_;

    static float walk(const Tree& tree, const float* x) {
        int node = 0;
        while (tree.left[node] != -1 || tree.right[node] != -1) {
            int feat = tree.feature[node];
            node = (x[feat] <= tree.threshold[node]) ? tree.left[node] : tree.right[node];
        }
        return tree.value[node];
    }
};

inline bool RandomForestPredictor::load_from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    Json::CharReaderBuilder builder;
    Json::Value root;
    std::string err;
    if (!Json::parseFromStream(builder, f, &root, &err)) return false;

    n_features_ = root.get("n_features", 0).asUInt();
    if (n_features_ == 0) return false;

    feature_names_.clear();
    for (const auto& v : root["feature_names"]) feature_names_.push_back(v.asString());

    medians_.clear();
    for (const auto& v : root["medians"]) medians_.push_back(v.asFloat());

    trees_.clear();
    trees_.reserve(root["trees"].size());
    for (const auto& tnode : root["trees"]) {
        Tree t;
        for (const auto& v : tnode["feature"])   t.feature.push_back(v.asInt());
        for (const auto& v : tnode["threshold"]) t.threshold.push_back(v.asFloat());
        for (const auto& v : tnode["left"])      t.left.push_back(v.asInt());
        for (const auto& v : tnode["right"])     t.right.push_back(v.asInt());
        for (const auto& v : tnode["value"])     t.value.push_back(v.asFloat());

        const size_t n = t.feature.size();
        if (n == 0 ||
            t.threshold.size() != n || t.left.size() != n ||
            t.right.size() != n || t.value.size() != n) {
            return false;
        }
        trees_.push_back(std::move(t));
    }
    return !trees_.empty();
}

inline float RandomForestPredictor::predict_proba(const float* x) const {
    if (trees_.empty()) return 0.0f;
    float sum = 0.0f;
    for (const auto& t : trees_) sum += walk(t, x);
    return sum / static_cast<float>(trees_.size());
}

}  // namespace hnswlib
