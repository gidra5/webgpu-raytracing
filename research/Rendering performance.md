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

https://blog.demofox.org/2022/03/02/sampling-importance-resampling/

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

area light source
https://eheitzresearch.wordpress.com/415-2/

gradient domain
https://mediatech.aalto.fi/publications/graphics/GPT/
https://mediatech.aalto.fi/publications/graphics/GMLT/
https://diglib.eg.org/server/api/core/bitstreams/a365c243-3d09-4939-a185-3af4b4b531f1/content
https://files.is.tue.mpg.de/black/papers/OpenDR.pdf
https://www.semanticscholar.org/paper/Lossless-Basis-Expansion-for-Gradient%E2%80%90Domain-Fang-Hachisuka/3a6618b2b68f8b7f3a7ae3bfc70126ee761ed297
https://cs.uwaterloo.ca/~thachisu/lbegdr.pdf

metropolis
https://graphics.stanford.edu/papers/metro/
https://users.cg.tuwien.ac.at/zsolnai/gfx/adaptive_metropolis/

fiber rendering
https://dl.acm.org/doi/pdf/10.1145/3023368.3023372

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
