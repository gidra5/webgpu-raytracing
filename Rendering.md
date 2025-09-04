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

## Reprojection

We can reuse data from previous frame using reprojection.
We calculate intersection point for a given pixel and project it back into screen coordinates sing previous frame's view matrix.

We can generate a projection matrix from view matrix using frustrum planes' normals following the [blog](https://jacco.ompf2.com/2024/01/18/reprojection-in-a-ray-tracer/). We can pack computation of d1 and d2 for each axis into one matrix, that will generate vector `dx1, dy1, dx2, dy2` when point to be projected multiplied with it. We can also compute `d1 + d2` by multiplying with another matrix. By composing and generating these two matrices on cpu, we can reduce actual projection to transform by mat3x4 + division of vec2, instead of multiplying vec4 by mat4x4 (inv view matrix) and division by scalar (to convert from homogeneous coords). That improves both projection speed and its precision.

Once we projected, the problem becomes how to precisely sample old frame data to reduce ghosting, dissoclusion artifacts and smearing.
The origin of the problem comes from rounding to nearest pixel to get the data. Since on high frame rates pixel coordinates change very slowly, a lot of frames will sample the same pixel position as the current frame, which ultimately freezes last frame in place, nullifying any projection results.

To avoid such rounding errors, we can reduce projection rate to a fraction of actual frame rate, such that current pixel coordinate and projected one are significantly different.
Another common technique to deal with accumulating errors is using exponential mean average (EMA), which boils down to simply mixing projected and current pixel colors with some fraction.

But there is also a question of disocclusion checks, which are equally significant to the quality of reprojection. A typical approach is to sample depth buffer, compute hit point from pixels ray direction and depth, and compare it with point that was projected. If they are too far from each other, we discard this projection attempt. We can avoid computing hit point by saving it into another buffer, and comparing with it instead. That improves stability of the result a lot, by avoiding computation errors and using the exact hit point used to compute the color.

Another technique that improves projection stability is sampling neighboring pixels and selection the one with least distance to the target point. At a glazing views computation error may be too significant, and pixels will shift, which creates a texture melting effect. For reference [reddit post](https://www.reddit.com/r/GraphicsProgramming/comments/1f6d4fg/temporal_back_reprojection_issue_at_grazing_angles/)

The things to look for a good reprojection:
1. On stale camera reprojection should be a perfect match.
2. Edges on closer shapes should not be false positives for disocclusion.
3. There should not be any flickering

A naive projection may not project the point to where it was traced from due to tiny float error.
Rounding will snap the value to where it must be, but it facilitates ghosting due to reprojection of small movements to the same pixel.
One could try jittering traced pixel to create significant enough change, that there will be no more need for pixel perfect projection. But naive implementation will accumulate jittering, which will only blur the picture.

test matrix - some invariants and identity. It should reproject points to the same coords if camera doesn't move

the best result so far:
![[chrome_cNkNWjSR9b_1754164912.mp4]]

Needed to calculate reprojection error in current frame, and use it to correct for any other reprojections. This allows to make perfect static image accumulation and improves accuracy in motion. 
Another thing that *almost* works is reprojecting sampled point once more. After first reprojection we sample geometry data, including the world-space point associated with the sample. We can reproject it again and somehow it fixes a lot of reprojection errors, allowing us to enter a decent accuracy territory.
Thus any such improvements are compared with the current sample's distance to desired point, and only accepted if it is better then the current sample.
Trying to sample only cells that are in the same object/face somehow does not work.

Similar accuracy can be achieved by approximating derivative of a projected point relative to its uv coords. We can use it to compute required uv change to eliminate the error in sampled position.
Essentially we want to solve the equation `(du, dv) * deltaUV = deltaP`, where `du` and `dv` are the approximated derivatives. This yields almose the same accuracy but with different numerical error distribution.

Jittering can improve reprojection in motion, since a lot of errors appear because the movements are very small, especially with high framerates.

There is also an reprojection error when moving parallel to projection plane. If we could estimate that error, we may additionally improve accuracy in such cases.

Making precise reprojection is extremely hard, especially when given limited precision and resolution.

diffuse, ao, shadows can be reprojected under any camera transform (rotation, translation). It does not depend on viewing angle, but global illumination depends on scene.
reflection can be reprojected under camera rotation. Since it depends on viewing angle translation will invalidate it. But rotation will just move the ray on screen, without affecting its direction.
Translation reprojection is non trivial, if possible.

doing jitter correctly: https://alextardif.com/TAA.html

Reprojection accuracy seems not to depend on accuracy of actual motion vectors, since computing it directly from camera position change does not improve accuracy?

The two can be combined to achieve even better result.

Further improvements seem to be bound by f32 calculations precision and geometry/frame buffer resolution.

precision can be improved using two f32 numbers to store lower and upper half of the numbers
https://www.youtube.com/watch?v=6OuqnaHHUG8&ab_channel=Creel
https://www.youtube.com/watch?v=5IL1LJ5noww&ab_channel=Creel
https://godotengine.org/article/emulating-double-precision-gpu-render-large-worlds/
It didn't work that well. Adding precision to most accuracy improving computation in reprojection didn't help. Adding accuracy to the reprojection itself helped a bit, but not nearly enough. The worst part is that performance considerably dropped, almost two times worse. It also added flicker, so maybe combining with regular reprojection it may yield better overall accuracy.
Making view inv and reprojection matricies full precision made reprojection better.
Performance normalized to a few additional ms for reprojection with double accuracy mainly after unrolling loops required for vectors and matrices.

Another guess was that interpolation was bad and didn't accommodate for perspective, giving more weight to closer points. That didn't improve anything, it seems.

Another idea was to only use view-space points for reprojection, without potentially worsening accuracy due to large world-space coordinates.

try to achieve exact reprojection for slightly offset world-space positions and then interpolate these instead. Tried multiple approaches, like add small offset to the current point and computing how much it moved in screen space. Then using these measures to approximate error due to movement in space by multiplying each error with corresponding coord on camera movement vector.

During raytracing you may compute the intersection point using barycentric coords and add some offset to avoid self intersection. That is kind of undesired for the geometry buffer and could actually introduce more errors due to intersection now not being exactly on the ray. Thus we need to actually return barycentrics and use it to compute what works best.

One brute-force technique you could use to eliminate disocclusion artifacts, or just noisy image in those parts, is rendering multiple layers of the scene.
We can additionally render the image at the second in-order intersection of the pixel ray. That way we get to accumulate rays also on immediately occluded surfaces.
We may also make backface intersections to try eliminating disocclusion from revealed parts of the object itself. But that may become not that useful due to extreme viewing angles that we get on such parts of the object. But it can have other uses, like generally improving global illumitation by reprojection.
These layers allow us to deeply represent the scene and opens up a lot of opportunities for long-term reuse.

One problem with this approach is performance. Since we basically render multiple images that all do a full raytrace of the scene, it quickly becomes unresponsive. One possible approach to fixing it might be rendering further layers at lower resolution. Since they are not immediately visible, it might be ok to have lower quality for them.

To implement that the initial naive "easy to integrate" approach was to trace a ray, and then trace another ray in the same direction from the hit point we get. Do that until we are at the required level.
That seems to be have a big problem with accuracy, probably because each time we change ray origin we accumulate numerical errors very quickly.
The harder to integrate approach is to pass another parameter to bvh intersection routine, that will be used in place to count necessary intersections.
A bit easier approach is to pass full interval to intersection test, and just update its min value to get the next closest intersection. It yields much better accuracy during reprojection and requires not too much changes.
Also sometimes multiple faces will be frontfacing and occluded by each other, so we need to check both nearest backface intersection and frontface and choose the one that is the closest.

With 8 layers if has reasonable coverage for simple scenes, which eliminates a lot of in-view disocclusion ghosting. Although logically you would expect 3 to be enough (one for backface and one for the next frontface intersection) but in reality there might be a lot more layers that would cover full depth of the scene and eliminate most of the ghosting.

A good news is that adding more layers at some point won't affect perf that much, because most of the rays are already missing the scene completely.
Bad news is that for even mildly complex scenery it is not really the case.

[reddit post with current progress](https://www.reddit.com/r/GraphicsProgramming/comments/1mpcrtr/temporal_reprojection_without_disocclusion/)

result so far:
![[gmAofKkLc6.mp4]]

depth peeling
https://www.microsoft.com/en-us/research/wp-content/uploads/2006/06/tr-2006-81.pdf
https://diglib.eg.org/server/api/core/bitstreams/229ef198-d130-4142-bccf-9ac7cf7499ff/content
https://highperformancegraphics.org/previous/www_2009/presentations/liu-bucket.pdf
https://gitea.yiem.net/QianMo/Real-Time-Rendering-4th-Bibliography-Collection/raw/commit/c4d6730d56a0d16f3baf7e588242c608fc72e379/Chapter%201-24/[1056]%20[HPG%202009]%20Efficient%20Depth%20Peeling%20via%20Bucket%20Sort.pdf
https://developer.download.nvidia.com/assets/gamedev/docs/OrderIndependentTransparency.pdf

depth peeling pipeline:
1. inputs
   1. clip space vertices
   2. depth min/max texture
   3. fragments depth stats texture array
   4. depth plane buckets texture array
2. Gather stats into depth min/max and stats array
3. Compute buckets from stats, such that equal amount of fragments is in each, clamped to min/max. Try computing bucket planes such that top n layers are guaranteed to be recognized.
4. run render pipeline on vertices, sort fragments into buckets
5. Resolve depth layers from buckets

https://selgrad.org/publications/2017_hpg_HBSS.pdf
https://research.nvidia.com/sites/default/files/pubs/2016-06_Deep-G-Buffers-for/Mara2016DeepGBuffer-extended-bright.pdf
https://research.nvidia.com/sites/default/files/pubs/2015-08_An-Adaptive-Acceleration/AcceleratedSSRT_HPG15.pdf
https://outerra.blogspot.com/2012/11/maximizing-depth-buffer-range-and.html
https://community.khronos.org/t/linearize-the-depth-buffer/72335/8

Render 6 frustrums with cubemap for each one. Thats view cube
Cubemaps store projections for each side of the frustrum.
Each view cube has up to n depth layers rendered. If layers on opposite sides start intersecting, we discard them.
Each layered view cube is saved for m frames. Oldest view keeps accumulated data, during render they're combined with reprojection.

The pipeline is as follows:
1. apply view-project to all vertices, clip. Compute shader.
2. Run a render pass for each side of each frustrum with input from compute shader. So 36 render passes. It will collect fragment depths into buckets via multiple framebuffer targets and will interpolate all attributes of the surfaces.
3. Run sorting pass in compute shader. Build an acceleration structure for screen space raytracing.

total data:
6 frustrums * 6 sides * n depth layers * m frames = 95681.25 * 1024 * 1024 = 100329062400 bytes (m=4, n=8)
frame = 6 attributes (9*4+4+2) * width * height
42 * 1920 * 1080 = 83.056640625 * 1024 * 1024 = 87091200 bytes 


Per each sample compute n values - coverage by a face being hit at a subsample i from 0 to n. 1-that is coverage by other faces.
Use that during interpolation. Given face id, take corresponding coverage values in neighbourhood of sample, bilinear interpolate that, and use as a weight for the sampled value.
## Lighting

For global illumination 8 bounces is plenty, especially if russian roulette for samples is implemented.

We may improve convergence speed by reprojecting on every bounce. Additionally it does not need to be as precise (probably?), since it is an indirect color. A naive reprojection (adding to a current bounce color reprojected value) yields some kind of dust effect:

![[chrome_JdHyu08wpq_1755075985.jpg]]

Also accuracy of reprojection greatly affects the quality.
## Filtering

We can filter out bad samples with exponential average, or based on some "quality" heuristic, like distance from the sample to target point.

conservative rendering:
https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-42-conservative-rasterization
https://github.com/andrewlowndes/perfect-antialiasing/tree/main

we can render multiple faces of the clip space cube, to fix low sample rate at grazing angles. Expensive memory-wise.
We could also fix edge sampling error, if we use the face coverage for a square we are sampling inside.
By computing the face coverage for a particular pixel square, we could improve our texture sampling accuracy, when pixel covers multiple faces, like on edges.

https://www.reddit.com/r/GraphicsProgramming/s/h1GI7rb6nq

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



```
Sn = 1/n*sum(k=1, n; f_k) 
   = 1/n*((n-1)*Sn-1 + f_n)
   = (n-1)/n*Sn-1 + f_n/n
   = Sn-1 - 1/n*Sn-1 + f_n/n
   = Sn-1 + 1/n*(f_n - Sn-1)
dSn = Sn - Sn-1 = 1/n*sum(k=1, n; f_k) - 1/(n-1)*sum(k=1, n-1; f_k)
	= 1/(n*(n-1))*((n-1)*sum(k=1, n; f_k) - n*sum(k=1, n-1; f_k))
	= Sn-1 + 1/n*(f_n - Sn-1) - (Sn-2 + 1/(n-1)*(f_n-1 - Sn-2))
	
	= 1/(n*(n-1))*((n-1)*sum(k=1, n; f_k) - n*sum(k=1, n; f_k) + n*f_n)
	= 1/(n*(n-1))*(sum(k=1, n; -f_k) + n*f_n)
	= 1/(n*(n-1))*(n*f_n - sum(k=1, n; f_k))
	= 1/(n-1)*(n*f_n - 1/n*sum(k=1, n; f_k))
	= 1/(n-1)*(f_n - Sn)
		
	= 1/(n*(n-1))*((n-1)*sum(k=1, n-1; f_k) + (n-1)*f_n - n*sum(k=1, n-1; f_k))
	= 1/(n*(n-1))*((n-1)*sum(k=1, n-1; f_k) - n*sum(k=1, n-1; f_k) + (n-1)*f_n)
	= 1/(n*(n-1))*(sum(k=1, n-1; -f_k) + (n-1)*f_n)
	= 1/n*(f_n - Sn-1)
	
	= Sn-1 + 1/n*(f_n - Sn-1) - Sn-2 - 1/(n-1)*(f_n-1 - Sn-2)
	= dSn-1 + 1/n*(f_n - Sn-1) - 1/(n-1)*(f_n-1 - Sn-2)
	= dSn-1 + 1/((n-1)*n)*((n-1)*(f_n - Sn-1) - n*(f_n-1 - Sn-2))
	= dSn-1 + 1/((n-1)*n)*((n-1)*f_n - (n-1)*Sn-1 - n*f_n-1 + n*Sn-2)
	= dSn-1 + 1/((n-1)*n)*(n*f_n - f_n - n*Sn-1 + Sn-1 - n*f_n-1 + n*Sn-2)
	= dSn-1 + 1/((n-1)*n)*(n*df_n - f_n - n*(Sn-1 - Sn-2) + Sn-1)
	= dSn-1 + 1/((n-1)*n)*(n*df_n - f_n + n*dSn-1 + Sn-1)
	= dSn-1 + 1/((n-1)*n)*(n*df_n + n*dSn-1 + Sn-1 - f_n)
	
	
Sn = S + 1/n*sum(k=1, n; df_k)

dSn ~= dSn-1 + 1/(n-1)*dSn-1
	= n/(n-1)*dSn-1
```

triangle intersect:
https://stackoverflow.com/questions/13163129/ray-triangle-intersection#:~:text=I%20have%20done%20a%20lot%20of%20benchmarks%2C,as%20fast%20as%20M%C3%B6ller%20and%20Trumbore's%20algorithm
https://www.researchgate.net/publication/41910471_Yet_Faster_Ray-Triangle_Intersection_Using_SSE4
https://www.researchgate.net/publication/352128555_Robust_Visibility_Surface_Determination_in_Object_Space_via_Plucker_Coordinates