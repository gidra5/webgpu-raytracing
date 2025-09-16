## Scene
Scene is a tree of bounding boxes, where leaves are model instances. All child nodes' coordinates are in parent local coordinates, including bounds itself.

Model instances are a collection of model id, model matrix, that describes orientation and position against the bounding box, and other physical values like velocity, angular speed, etc.

Models are a tree of bounding boxes, where leaves are a list of triangles. All child nodes' coordinates are in parent local coordinates, including bounds itself. The origin of local coordinates is at center of mass.

Lower level bounding boxes also can store a pair of normals that have the lowest dot product to allow faster backface culling of the whole node. When looking at a triangle, we need to find an interpolated normal, that will have the lowest dot product. It does not really improve anything :(

Storing positions in local coordinates ensures that we dont accumulate large numerical errors due to float precision limitations. When computing final position, without local coordinates we may end up subtraction two large numbers, which will create larger numerical error that a sequence of subtractions of much smaller values.

All triangles have texture coordinates, base material id and normals associated with it.
All models have associated textures, that describe additional information about the model's triangles, like bumps, roughness, color, vertex position, etc.

backface intersection intuitively can be done by swapping order of edges. In that case u and v barycentric coords will be swapped, since they correspond to those edges.
Another approach is swapping direction of the ray, in with case only the intersection distance will need to be inverted back.

# Pipeline

Pipeline:
1. Project scene into clip spaces around the camera. Clip primitives outside of viewport cube.
Every clip space is frustrum with 90deg horizontal and vertical fov. The size is enough to contain camera frustrum. Optionally shift frustrums to cover the center.
2. Rasterize each frustrum along each major axis. Collect N first layers of fragments for the Z-axis. Collect N first and last fragments for X- and Y-axis.
3. Resolve fragments and collect interpolated fragment attributes - normal, uv, faceId, material properties.
4. Compute importance map - for each fragment measure variance and normalize into probability distribution.
5. Preprocess importance map into cdf.
6. For each hitpoint lookup the fragment and allocate more samples when high variance.
7. Run compute shader with fixed ray count. Sample importance map cdf to determine first ray origin.
8. For each fragment ray origin compute each of the components of bsdf.
9. For each hitpoint reproject prev frames values as "initial guess".
10. For each computed hitpoint splat the color values back into frame cache.
11. Allocate N threads and compute steady state between 4 points in the scene. Importance sample high and low variance points to even out the variance.
12. Rotate buffers for the next frame.
13. Reconstruct final frame from stored samples.

## Lighting

For global illumination 8 bounces is plenty, especially if russian roulette for samples is implemented.

We may improve convergence speed by reprojecting on every bounce. Additionally it does not need to be as precise (probably?), since it is an indirect color. A naive reprojection (adding to a current bounce color reprojected value) yields some kind of dust effect:

![[chrome_JdHyu08wpq_1755075985.jpg]]

Also accuracy of reprojection greatly affects the quality.

https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
## Filtering

We can filter out bad samples with exponential average, or based on some "quality" heuristic, like distance from the sample to target point.

conservative rendering:
https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-42-conservative-rasterization
https://github.com/andrewlowndes/perfect-antialiasing/tree/main

we can render multiple faces of the clip space cube, to fix low sample rate at grazing angles. Expensive memory-wise.
We could also fix edge sampling error, if we use the face coverage for a square we are sampling inside.
By computing the face coverage for a particular pixel square, we could improve our texture sampling accuracy, when pixel covers multiple faces, like on edges.

https://www.reddit.com/r/GraphicsProgramming/s/h1GI7rb6nq


https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2009.01674.x
https://jcgt.org/published/0003/04/04/paper.pdf
https://www.semanticscholar.org/paper/Real-time-multiply-recursive-reflections-and-using-Ganestam-Doggett/0c4cebce66ce22b9253b2674650eeeab4fae4879
https://abasilak.github.io/papers/journals/eg2020star/paper.pdf
http://graphics.cs.aueb.gr/graphics/docs/papers/IRT-i3D2016-av.pdf
https://abasilak.github.io/papers/journals/vc2020/paper.pdf
https://publications.lib.chalmers.se/records/fulltext/193772/193772.pdf
https://www.semanticscholar.org/paper/The-real-time-reprojection-cache-Nehab-Sander/c0d92df2423e643bb78c939b9136f7807c464f95
https://www.semanticscholar.org/paper/Accelerating-real-time-shading-with-reverse-caching-Nehab-Sander/ba59d37ae22053962d75762590a97e75fcb75977
https://www.semanticscholar.org/paper/Real-time-multiply-recursive-reflections-and-using-Ganestam-Doggett/0c4cebce66ce22b9253b2674650eeeab4fae4879?sort=relevance&page=2
https://www.semanticscholar.org/paper/Reflection-reprojection-using-temporal-coherence-Xie-Wang/0e7b582861fa33602801cec66d0908030f47249a
https://www.semanticscholar.org/paper/Generating-exact-ray-traced-animation-frames-by-Adelson-Hodges/987c5e122a777efe3fb997caa88dda5ae48477c7
https://www.semanticscholar.org/paper/Practical-approach-to-the-fast-Monte-Carlo-Gruzdev-Frolov/255f93658617156c20f54cb9f0dc6b4b8c84dcb6
https://cgg.mff.cuni.cz/~wilkie/Website/EGSR_14_files/WNDWH14HWSS.pdf

denoising
https://alain.xyz/blog/ray-tracing-denoising
https://www.reddit.com/r/GraphicsProgramming/s/Vm4LmWi3Dc
https://www.reddit.com/r/GraphicsProgramming/s/enCyTAwmm0

Importance Sampling: https://ameye.dev/notes/sampling-the-hemisphere/
https://www.reddit.com/r/GraphicsProgramming/s/lw9OdJUSkF
Assets: https://polyhaven.com/
https://gabrielgambetta.com/computer-graphics-from-scratch/05-extending-the-raytracer.html
You can check if neighboring pixels are occluded, and check that object first, assuming that for most rays they will hit the same object/triangle as the neighbor.

what we basically want is to find representative sample in the geometry buffer, the better the sample, the more stable the image will be.

https://wickedengine.net/2022/05/derivatives-in-compute-shader/
derivatives in compute shader, can be used for anisotropic filtering there.

https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/

https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/

Using matrices to compute further bounces:
https://bartwronski.com/2022/02/15/light-transport-matrices-svd-spectral-analysis-and-matrix-completion/
https://getcode.substack.com/p/massively-parallel-fun-with-gpus

Antialiasing:
https://www.iryoku.com/aacourse/
https://www.reddit.com/r/GraphicsProgramming/s/f26q2kQi56

depth of field:
https://blog.demofox.org/2018/07/04/pathtraced-depth-of-field-bokeh/

importance sampling advances
https://arxiv.org/pdf/2102.05407
https://math.arizona.edu/~tgk/mc/book_chap6.pdf
https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
https://jcgt.org/published/0014/01/08/paper.pdf
https://dl.acm.org/doi/abs/10.1145/1015706.1015750
https://arxiv.org/html/2504.05562v1
https://www.researchgate.net/publication/252576633_Fast_Filtering_and_Tone_Mapping_using_Importance_sampling
https://github.com/electronicarts/importance-sampled-FAST-noise

accelerating convergence:
https://en.wikipedia.org/wiki/Monte_Carlo_method
https://en.wikipedia.org/wiki/Series_acceleration
https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process
http://numbers.computation.free.fr/Constants/Miscellaneous/seriesacceleration.html

https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf

bloom
https://www.youtube.com/watch?v=QWqb5Gewbx8&ab_channel=AngeTheGreat

lens flare
https://resources.mpi-inf.mpg.de/lensflareRendering/pdf/flare.pdf
https://www.youtube.com/watch?v=IbJfZS0o2kg&ab_channel=GameDevelopersConference

panini projection
http://tksharpless.net/vedutismo/Pannini/
https://www.scribd.com/document/284463081/The-General-Panini-Projection
https://www.researchgate.net/publication/220795340_Pannini_A_New_Projection_for_RenderingWide_Angle_Perspective_Images

raytracing in one weekend series
https://raytracing.github.io/books/RayTracingInOneWeekend.html#diffusematerials
https://raytracing.github.io/books/RayTracingTheNextWeek.html#motionblur
http://raytracing.github.io/books/RayTracingTheRestOfYourLife.html

pbr book
https://www.pbr-book.org/4ed/contents

spatiotemporal upsampling
https://www.researchgate.net/publication/220792188_Spatio-temporal_upsampling_on_the_GPU
https://github.com/lukedan/ReSTIR-Vulkan?tab=readme-ov-file
https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf
https://github.com/jacquespillet/SVGF?tab=readme-ov-file
https://www.cg.tuwien.ac.at/sites/default/files/course/4411/attachments/08_next%20event%20estimation.pdf

[(PDF) Essential Ray Generation Shaders](https://www.researchgate.net/publication/354065227_Essential_Ray_Generation_Shaders)

virtualized geometry
https://discourse.threejs.org/t/virtually-geometric/28420

virtualized textures
https://discourse.threejs.org/t/virtual-textures/53353

depth function:
https://youtu.be/h1ocYFrtsM4?t=868

https://gamedevnotesblog.wordpress.com/category/algorithms/optimisation/

https://habr.com/en/articles/440488/
https://web.archive.org/web/20180822041342/http://ericpolman.com/2016/03/17/reflective-shadow-maps/
https://web.archive.org/web/20240717124558/https://ericpolman.com/2016/06/28/light-propagation-volumes/

https://bruop.github.io/tonemapping/

https://casual-effects.blogspot.com/2014/04/fast-terrain-rendering-with-continuous.html
https://www.decarpentier.nl/
https://briansharpe.wordpress.com/
https://rgl.epfl.ch/publications/Zeltner2020Specular
https://yehar.com/blog/?p=1495
https://www.realtimerendering.com/#books-small-table
https://larswander.com/writing/spectral-ray-tracing/
https://dl.acm.org/doi/10.1145/2601097.2601139
https://jo.dreggn.org/home/2010_atrous.pdf
https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-voxel-grids.html

fft and dft
https://paulbourke.net/miscellaneous/dft/

triangle intersect:
https://stackoverflow.com/questions/13163129/ray-triangle-intersection#:~:text=I%20have%20done%20a%20lot%20of%20benchmarks%2C,as%20fast%20as%20M%C3%B6ller%20and%20Trumbore's%20algorithm
https://www.researchgate.net/publication/41910471_Yet_Faster_Ray-Triangle_Intersection_Using_SSE4
https://www.researchgate.net/publication/352128555_Robust_Visibility_Surface_Determination_in_Object_Space_via_Plucker_Coordinates