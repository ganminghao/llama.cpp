// this is a Sparkinfer function wrapper, create an API to call llama-sparkinfer function to avoid dependency issue

#pragma once 
#include "ggml.h" 

struct reload_plan_result {
    std::vector<int> groups_to_reload;
    std::vector<int> slots_for_evict;
};
typedef struct reload_plan_result reload_plan_result;

#ifdef __cplusplus  
extern "C" {  
#endif  

typedef struct ggml_spif_context ggml_spif_context; 
  
bool ggml_spif_reload_plan(ggml_spif_context * ctx, ggml_tensor * tensor);  
bool ggml_spif_reload_exec(ggml_spif_context * ctx, ggml_tensor * tensor);  
  
#ifdef __cplusplus  
}  
#endif
