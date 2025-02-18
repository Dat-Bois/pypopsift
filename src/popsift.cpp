
#include "popsift.h"
#include <algorithm>
#include <cmath>
#include <mutex>

namespace pps{

PopSiftContext *ctx = nullptr;
std::mutex g_mutex;

PopSiftContext::PopSiftContext() : ps(nullptr){
    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set(0, false);
}

PopSiftContext::~PopSiftContext(){
    ps->uninit();
    delete ps;
    ps = nullptr;
}

void PopSiftContext::setup(float peak_threshold, float edge_threshold, bool use_root, float downsampling){
    bool changed = false;
    if (this->peak_threshold != peak_threshold) { this->peak_threshold = peak_threshold; changed = true; }
    if (this->edge_threshold != edge_threshold) { this->edge_threshold = edge_threshold; changed = true; }
    if (this->use_root != use_root) { this->use_root = use_root; changed = true; }
    if (this->downsampling != downsampling) { this->downsampling = downsampling; changed = true; }

    if (changed){
        config.setThreshold(peak_threshold);
        config.setEdgeLimit(edge_threshold);
        config.setNormMode(use_root ? popsift::Config::RootSift : popsift::Config::Classic );
        config.setFilterSorting(popsift::Config::LargestScaleFirst);
        config.setMode(popsift::Config::OpenCV);
        config.setDownsampling(downsampling);
        // config.setOctaves(4);
        // config.setLevels(3);

        if (!ps){
            ps = new PopSift(config,
                        popsift::Config::ProcessingMode::ExtractingMode,
                        PopSift::ByteImages );
        }else{
            ps->configure(config, false);
        }
    }
}

PopSift *PopSiftContext::get(){
    return ps;
}

py::object popsift(pyarray_uint8 image,
                 float peak_threshold,
                 float edge_threshold,
                 int target_num_features,
                 bool use_root,
                 float downsampling) {
    py::gil_scoped_release release;

    if (!image.size()) return py::none();

    std::once_flag ctx_init_flag;
    if (!ctx) {
        std::call_once(ctx_init_flag, []() {ctx = new PopSiftContext();});
    }

    int width = image.shape(1);
    int height = image.shape(0);
    int numFeatures = 0;
    
    while(true){
        g_mutex.lock();
        ctx->setup(peak_threshold, edge_threshold, use_root, downsampling);
        std::unique_ptr<SiftJob> job(ctx->get()->enqueue( width, height, image.data() ));
        std::unique_ptr<popsift::Features> result(job->get());
        g_mutex.unlock();

        numFeatures = result->getFeatureCount();

        if (numFeatures >= target_num_features || peak_threshold < 0.0001){
            popsift::Feature* feature_list = result->getFeatures();
            std::vector<float> points(4 * numFeatures);
            std::vector<float> desc(128 * numFeatures);

            for (size_t i = 0; i < numFeatures; i++){
                popsift::Feature pFeat = feature_list[i];

                for(int oriIdx = 0; oriIdx < pFeat.num_ori; oriIdx++){
                    const popsift::Descriptor* pDesc = pFeat.desc[oriIdx];

                    for (int k = 0; k < 128; k++){
                        desc[128 * i + k] = pDesc->features[k];
                    }

                    points[4 * i + 0] = std::min<float>(std::round(pFeat.xpos), width - 1);
                    points[4 * i + 1] = std::min<float>(std::round(pFeat.ypos), height - 1);
                    points[4 * i + 2] = pFeat.sigma;
                    points[4 * i + 3] = pFeat.orientation[oriIdx];
                }
            }

            py::gil_scoped_acquire acquire;
            py::list retn;
            retn.append(py_array_from_data(&points[0], numFeatures, 4));
            retn.append(py_array_from_data(&desc[0], numFeatures, 128));
            return retn;
        }else{
           // Lower peak threshold if we don't meet the target
           peak_threshold = (peak_threshold * 2.0) / 3.0;
        }
    }

    // We should never get here
    return py::none();
}

}