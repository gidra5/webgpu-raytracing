We can measure performance of the raytracer in two ways:
* Convergence speed. How fast over time is the algorithm converges to "ground truth" image. For monte-carlo based approaches it usually means amount of samples per pixel.
* Single sample image render time. How much time needed to render a single sample from the scene.
* Scalability. How does it scale wrt scene and material complexity.

For practical real-time applications we need all of these to be exceptional.
## Filtering

We can filter out bad samples with exponential average, or based on some "quality" heuristic, like distance from the sample to target point.

we can render multiple faces of the clip space cube, to fix low sample rate at grazing angles. Expensive memory-wise.
We could also fix edge sampling error, if we use the face coverage for a square we are sampling inside.
By computing the face coverage for a particular pixel square, we could improve our texture sampling accuracy, when pixel covers multiple faces, like on edges.

# Caching/accumulation

Non-Euclidean rendering. Since in such geometries the light does not follow straight lines, usual techniques can't be applied. But since all medium are scattering to some degree, we can "keep lines straight" by continuously scattering in the direction of straight lines. That basically just scales light contribution along the path stronger than predicted by Beer's law, but other than that it allows us to reuse all knowledge from a standard Euclidean raytracing.

https://www.reddit.com/r/GraphicsProgramming/s/8M9RchBD2d
https://www.cemyuksel.com/research/papers/fuzzy_boolean-SIGGRAPH24.pdf
cone tracing
https://ssteinberg.xyz/2025/08/28/elliptical_cone_tracing_ads/

ReGIR - An advanced implementation for many-lights offline rendering | Tom Clabault
https://tomclabault.github.io/blog/2025/regir/

spectral
https://inria.hal.science/hal-03331619/file/Efficient%20Spectral%20Rendering%20on%20the%20GPU.pdf

sampling
https://blog.demofox.org/2022/03/02/sampling-importance-resampling/
https://eheitzresearch.wordpress.com/792-2/
https://eheitzresearch.wordpress.com/749-2/
https://graphics.cs.utah.edu/research/projects/gris/
https://graphics.cs.utah.edu/research/projects/virtual-blue-noise-lighting/
https://graphics.cs.utah.edu/research/projects/virtual-blue-noise-lighting/
https://graphics.cs.utah.edu/research/projects/ray-tracing-hw-adaptive-lod/
https://graphics.cs.utah.edu/research/projects/area-restir/
https://www.cemyuksel.com/research/

depth function:
https://youtu.be/h1ocYFrtsM4?t=868

fft and dft
https://paulbourke.net/miscellaneous/dft/

https://briansharpe.wordpress.com/
https://www.decarpentier.nl/graphics

https://web.archive.org/web/20180822041342/http://ericpolman.com/2016/03/17/reflective-shadow-maps/
https://web.archive.org/web/20240717124558/https://ericpolman.com/2016/06/28/light-propagation-volumes/

https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-voxel-grids.html

ROBUST MONTE CARLO METHODS FOR LIGHT TRANSPORT SIMULATION
https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf

Using matrices to compute further bounces:
https://bartwronski.com/2022/02/15/light-transport-matrices-svd-spectral-analysis-and-matrix-completion/

Approximating ray traced reflections using screenspace data
https://publications.lib.chalmers.se/records/fulltext/193772/193772.pdf

Rasterization-based Progressive Photon Mapping
https://abasilak.github.io/papers/journals/vc2020/paper.pdf

Efficient GPU Screen-Space Ray Tracing
https://jcgt.org/published/0003/04/04/paper.pdf

A Survey of Multifragment Rendering
https://abasilak.github.io/papers/journals/eg2020star/paper.pdf

A Multiview and Multilayer Approach for Interactive Ray Tracing
http://graphics.cs.aueb.gr/graphics/docs/papers/IRT-i3D2016-av.pdf

Hero Wavelength Spectral Sampling
https://cgg.mff.cuni.cz/~wilkie/Website/EGSR_14_files/WNDWH14HWSS.pdf

pbr book
https://www.pbr-book.org/4ed/contents
https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Metropolis_Sampling
https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Bias
https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency#eq:splitting-candidate-integral

https://getcode.substack.com/p/massively-parallel-fun-with-gpus
https://www.semanticscholar.org/paper/Practical-approach-to-the-fast-Monte-Carlo-Gruzdev-Frolov/255f93658617156c20f54cb9f0dc6b4b8c84dcb6

virtualized geometry
https://discourse.threejs.org/t/virtually-geometric/28420

virtualized textures
https://discourse.threejs.org/t/virtual-textures/53353

https://gamedevnotesblog.wordpress.com/category/algorithms/optimisation/

spatiotemporal upsampling
https://www.researchgate.net/publication/220792188_Spatio-temporal_upsampling_on_the_GPU
https://github.com/lukedan/ReSTIR-Vulkan?tab=readme-ov-file
https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf
https://github.com/jacquespillet/SVGF?tab=readme-ov-file
https://www.cg.tuwien.ac.at/sites/default/files/course/4411/attachments/08_next%20event%20estimation.pdf
https://d1qx31qr3h6wln.cloudfront.net/publications/siga21_volumeReSTIR.pdf

acceleration structures
https://my.eng.utah.edu/~cs6958/papers/thesis_ize.pdf

Importance Sampling: https://ameye.dev/notes/sampling-the-hemisphere/
https://www.reddit.com/r/GraphicsProgramming/s/lw9OdJUSkF
https://arxiv.org/pdf/1707.08358
https://advances.realtimerendering.com/s2018/s2018_real_time_correct_soft_shadows.pdf

accelerating convergence:
https://en.wikipedia.org/wiki/Monte_Carlo_method
https://en.wikipedia.org/wiki/Series_acceleration
https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process
http://numbers.computation.free.fr/Constants/Miscellaneous/seriesacceleration.html

importance sampling advances
https://arxiv.org/pdf/2102.05407
https://math.arizona.edu/~tgk/mc/book_chap6.pdf
https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
https://jcgt.org/published/0014/01/08/paper.pdf
https://dl.acm.org/doi/abs/10.1145/1015706.1015750
https://arxiv.org/html/2504.05562v1
https://www.researchgate.net/publication/252576633_Fast_Filtering_and_Tone_Mapping_using_Importance_sampling
https://github.com/electronicarts/importance-sampled-FAST-noise

conservative rendering:
https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-42-conservative-rasterization
https://github.com/andrewlowndes/perfect-antialiasing/tree/main

denoising/filtering
https://jo.dreggn.org/home/2010_atrous.pdf
https://alain.xyz/blog/ray-tracing-denoising
https://www.reddit.com/r/GraphicsProgramming/s/Vm4LmWi3Dc
https://www.reddit.com/r/GraphicsProgramming/s/enCyTAwmm0
https://web.ece.ucsb.edu/~psen/Papers/Sen11_RandomParameterFiltering_LoRes.pdf
https://cseweb.ucsd.edu/~viscomp/classes/cse274/wi18/papers/a18-sen.pdf
https://perso.telecom-paristech.fr/boubek/papers/BCD/BCD_lowres.pdf
https://web.ece.ucsb.edu/~psen/Papers/Sen15_DenoisingMCRenders.pdf
https://people.engr.tamu.edu/nimak/Data/EG13_RemovingMCNoiseWithGeneralDenoising.pdf
https://eheitzresearch.wordpress.com/772-2/

area light source
https://eheitzresearch.wordpress.com/415-2/

gradient domain
https://mediatech.aalto.fi/publications/graphics/GPT/
https://mediatech.aalto.fi/publications/graphics/GMLT/
https://diglib.eg.org/server/api/core/bitstreams/a365c243-3d09-4939-a185-3af4b4b531f1/content
https://files.is.tue.mpg.de/black/papers/OpenDR.pdf
https://www.semanticscholar.org/paper/Lossless-Basis-Expansion-for-Gradient%E2%80%90Domain-Fang-Hachisuka/3a6618b2b68f8b7f3a7ae3bfc70126ee761ed297
https://cs.uwaterloo.ca/~thachisu/lbegdr.pdf
https://www.cs.umd.edu/~zwicker/publications/GradientDomainRenderingSTAR-CGF2019.pdf

metropolis
https://graphics.stanford.edu/papers/metro/
https://users.cg.tuwien.ac.at/zsolnai/gfx/adaptive_metropolis/
https://web.archive.org/web/20240920170720/https://www.uni-kl.de/AG-Heinrich/MediaMLT.pdf

triangle intersect:
https://stackoverflow.com/questions/13163129/ray-triangle-intersection#:~:text=I%20have%20done%20a%20lot%20of%20benchmarks%2C,as%20fast%20as%20M%C3%B6ller%20and%20Trumbore's%20algorithm
https://www.researchgate.net/publication/41910471_Yet_Faster_Ray-Triangle_Intersection_Using_SSE4
https://www.researchgate.net/publication/352128555_Robust_Visibility_Surface_Determination_in_Object_Space_via_Plucker_Coordinates

https://blog.demofox.org/2020/06/04/a-link-between-russian-roulette-and-rejection-sampling-importance-sampling/
https://blog.demofox.org/2020/11/25/multiple-importance-sampling-in-1d/
https://blog.demofox.org/2018/06/12/monte-carlo-integration-explanation-in-1d/
https://blog.demofox.org/2019/05/25/generating-random-numbers-from-a-specific-distribution-with-the-metropolis-algorithm-mcmc/
https://blog.demofox.org/2020/07/11/interpolating-data-over-arbitrary-shapes-with-laplaces-equation-and-walk-on-spheres/
https://blog.demofox.org/2018/04/16/prefix-sums-and-summed-area-tables/
https://blog.demofox.org/2016/07/28/fourier-transform-and-inverse-of-images/

https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
https://www.johndcook.com/blog/standard_deviation/
https://blog.demofox.org/2020/03/10/how-do-i-calculate-variance-in-1-pass/

path guiding, zero variance theory
https://drive.google.com/file/d/1xIU8YB-R6iS2JHanA9v9P-3WbmqALxfe/view (ch. 74.4.2, p. 363)

https://www.semanticscholar.org/paper/Accelerating-Path-Tracing-by-Re-Using-Paths-Bekaert-Sbert/a5a7d3c86f756f443aeae07ac8c64f8346914204
https://www.semanticscholar.org/paper/Towards-accelerating-polarization-path-tracing-of-Ohba-Yatagawa/676a79673126be1fc507f8d431dea0f0b7960311

fiber rendering
https://dl.acm.org/doi/pdf/10.1145/3023368.3023372

differential rendering
https://shuangz.com/projects/dtrt-sa19/diff_rendering.pdf
https://scontent.fiev6-1.fna.fbcdn.net/v/t39.8562-6/10000000_245993640840564_4920334379577755863_n.pdf?_nc_cat=101&ccb=1-7&_nc_sid=e280be&_nc_ohc=cbyx_6jcjcwQ7kNvwGc0E6P&_nc_oc=AdkdiCu8q5V-5kHy68qQ30B93WzBKdn0TTLqQ26CcXyokxz6qN1TBiQq8Bbqyw6QWF8&_nc_zt=14&_nc_ht=scontent.fiev6-1.fna&_nc_gid=6e26HQsWQwO2eUApHHnopg&oh=00_AffpCtHaVUn7eaQo_MIQ-y35IDViZZPJZpepCs9dXkzpAg&oe=68ED4AB1
https://shuangz.com/projects/psdr-aq-sg22/psdr-aq-sg22.pdf
https://inria.hal.science/hal-02497191/file/differentiable-pt-cov.pdf
https://shuangz.com/projects/psdr-pixel-sa22/psdr-pixel-sa22.pdf
https://shuangz.com/projects/psdr-sdf-sg24/psdr-sdf-sg24.pdf