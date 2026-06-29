// SwiGLU with GELU activation (parallel form, like llama.cpp's
// LLM_FFN_GELU / LLM_FFN_PAR — used by Gemma 3 / Gemma 4):
//
//   gate[i] = gelu_tanh(gate[i]) * up[i]
//
// where gelu_tanh is the standard tanh approximation:
//
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Constants: sqrt(2/pi) ≈ 0.7978845608, the cubic coefficient is the
// canonical Hendrycks-Gimpel value.
//
// Launch: 1D global = n in groups of GELU_MUL_LOCAL.

#ifndef GELU_MUL_LOCAL
#define GELU_MUL_LOCAL 256
#endif

__attribute__((reqd_work_group_size(GELU_MUL_LOCAL, 1, 1)))
__kernel void gelu_mul(
    __global       float* gate,    // input + output
    __global const float* up,
    const int             n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }
    const float g  = gate[gid];
    const float g3 = g * g * g;
    const float t  = tanh(0.7978845608f * (g + 0.044715f * g3));
    const float ge = 0.5f * g * (1.0f + t);
    gate[gid] = ge * up[gid];
}