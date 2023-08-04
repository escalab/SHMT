#ifndef __GPTPU_UTILS_H__
#define __GPTPU_UTILS_H__

namespace gptpu_utils{
class EdgeTpuHandler{
public:
    EdgeTpuHandler();
    ~EdgeTpuHandler();
    unsigned int list_devices(bool verbose);
    void open_device(unsigned int tpuid, bool verbose);
    unsigned int build_model(const std::string& model_path);
    void build_interpreter(unsigned int tpuid, unsigned int model_id);
    void populate_input(uint8_t* data, int size, unsigned int model_id);
    void model_invoke(unsigned int model_id, int iter);
    void get_output(int* data, unsigned int model_id);
    void get_raw_output(uint8_t* data, int size, unsigned int model_id, uint8_t& zero_point, float& scale);
};

}
#endif
