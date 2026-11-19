/* CPU port of the forward compositing equations in diff-plane-rasterization.
 * Copyright (C) 2023, Inria / GRAPHDECO; modifications for CPU execution.
 * Same research/evaluation license as ../submodules/diff-plane-rasterization/LICENSE.md.
 * No approximate sort, alpha threshold, splat pruning, or mesh substitution.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <numeric>
#include <vector>

extern "C" int composite_cpu(
    int n, int width, int height, int threads,
    const float* xy, const float* depth, const int32_t* radii,
    const float* conic_opacity, const float* rgb, const float* maps,
    const float* background, float fx, float fy, float cx, float cy,
    float* image, float* plane_depth) noexcept {
    try {
        constexpr int tile = 16;
        const int nx = (width + tile - 1) / tile;
        const int ny = (height + tile - 1) / tile;
        std::vector<std::vector<int>> bins(nx * ny);
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        // CUB's radix sort is stable for equal depths; preserve input order too.
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return depth[a] < depth[b]; });
        for (int i : order) {
            if (radii[i] <= 0) continue;
            auto bound = [](float x, int maximum) {
                return std::min(maximum, std::max(0, int(x / tile)));
            };
            const int x0 = bound(xy[2*i] - radii[i], nx);
            const int y0 = bound(xy[2*i+1] - radii[i], ny);
            const int x1 = bound(xy[2*i] + radii[i] + tile - 1, nx);
            const int y1 = bound(xy[2*i+1] + radii[i] + tile - 1, ny);
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x) bins[y*nx+x].push_back(i);
        }
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (int bin = 0; bin < nx * ny; ++bin) {
            const int bx = (bin % nx) * tile, by = (bin / nx) * tile;
            for (int y = by; y < std::min(by+tile, height); ++y) {
                for (int x = bx; x < std::min(bx+tile, width); ++x) {
                    float transmittance = 1.f, color[3] = {}, map[5] = {};
                    for (int i : bins[bin]) {
                        const float dx = xy[2*i] - x, dy = xy[2*i+1] - y;
                        const float* co = conic_opacity + 4*i;
                        const float power = -.5f*(co[0]*dx*dx + co[2]*dy*dy) - co[1]*dx*dy;
                        if (power > 0.f) continue;
                        const float alpha = std::min(.99f, co[3]*std::exp(power));
                        if (alpha < 1.f/255.f) continue;
                        const float next = transmittance*(1.f-alpha);
                        // The crossing Gaussian is excluded by the CUDA kernel.
                        if (next < .0001f) break;
                        for (int c = 0; c < 3; ++c)
                            color[c] += rgb[3*i+c]*alpha*transmittance;
                        if (maps)
                            for (int c = 0; c < 5; ++c)
                                map[c] += maps[5*i+c]*alpha*transmittance;
                        transmittance = next;
                    }
                    const int pixel = y*width+x;
                    for (int c = 0; c < 3; ++c)
                        image[3*pixel+c] = color[c] + transmittance*background[c];
                    if (maps)
                        plane_depth[pixel] = map[4] / -(map[0]*((x-cx)/fx)
                                                     + map[1]*((y-cy)/fy) + map[2] + 1.e-8f);
                }
            }
        }
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}
