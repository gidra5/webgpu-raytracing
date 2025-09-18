
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
https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2009.01674.x
https://www.microsoft.com/en-us/research/wp-content/uploads/2006/06/tr-2006-81.pdf
https://diglib.eg.org/server/api/core/bitstreams/229ef198-d130-4142-bccf-9ac7cf7499ff/content
https://highperformancegraphics.org/previous/www_2009/presentations/liu-bucket.pdf
https://gitea.yiem.net/QianMo/Real-Time-Rendering-4th-Bibliography-Collection/raw/commit/c4d6730d56a0d16f3baf7e588242c608fc72e379/Chapter%201-24/[1056]%20[HPG%202009]%20Efficient%20Depth%20Peeling%20via%20Bucket%20Sort.pdf
https://developer.download.nvidia.com/assets/gamedev/docs/OrderIndependentTransparency.pdf


depth peeling pipeline:
1. Initialize 6 texture arrays of length L. Each array item is a bin responsible for some depth range. The fisrt 3 of them are used for the current iteration, the other 3 are the values from the previous iteration.
   2. prev depth min, start from 0 and increment in 1/L steps.
   3. prev depth max, start from 1/L and increment in 1/L steps.
   4. prev fragment count, all values are 0.
   5. current depth min, all values are 1
   6. current depth max, all values are 0
   7. current fragment count, all values are 0
   8. When reusing from prev frame:
      1. prev depth min, equal to prev frame's current depth min.
      2. prev depth max, equal to prev frame's current depth min from next bin, decremented by O. 
      3. prev fragment count, all values are 0.
9. We try to achieve the following shape of the layers:
   10. First half of the layers has 2 fragments, the min and max. 
   11. The rest of the layers will have other fragments, which will be discarded.
   12. All the layers should still cover the whole depth range.
13. Look at prev bin values, decide effective bins for current iteration.
   14. If total fragment count is 0, then bin ranges equal to the prev min/max values. Since we go a fragment at this position, but the count is 0, it means that we just started, otherwise we would have had at least one fragment in total.
   15. If total fragment count is not 0, then its a secondary pass. Min and max values should be filled.
   16. Find a bin with at least 1 fragment.
   17. If the bin has more than L fragments, then split it into L equal bins and use the ranges for the current iteration.
   18. if prev bin had 1 fragment, then we took min value from the current bin. Increment current bin's min by O. Decrement fragment count.
   19. If the bin has 0 fragments, find next bin with at least 1 fragment.
   20. If the bin has 1 fragment, use it as min value of the current bin, take max value from the min of the next bin in prev.
   21. If the bin has 2 fragments, use them as min and max values of the current bin.
   22. let N be the current fragment count. If current bin has M fragments, which is less than L - N, subdivide it into M equal bins.
   23. If current bin has more of equal to L - N fragments, subdivide it into L - N equal bins.
   24. Set last bin's max to 1.
25. Sort fragments into bins.
26. If current fragment's depth is within range of the bin I, add it to the min/max and increment fragment count for that bin. The range is not inclusive on upped bound.

keeping track of min and max of a bit harder than a single min value. But maybe it adds some performance by cutting down the number of passes necessary.

bad:
1. Gather stats into depth min/max
2. split into L equal sections the depth span. Or split by prev depth layers/iteration. For each section find closest fragment and fragment count in each bin.
3. Repeat until found enough closest fragments.
4. If fragment count larger than L, place last bucket before min depth
5. After first iteration choose max to be upper bound of the first bucket that had more than 2 fragments. Min should be the lower bound of the bucket where total fragment count of buckets exceeds L.
6. Use stencil test to discard fragments that are already known to be sorted enough (we found and sorted all fragments, or we found enough closest ones). That will save a lot of depth tests and rasterization work on later passes.
7. Compute and save fragment attributes on the final pass.
8. Reuse prev frame layer depths to better estimate initial bin boundaries.

A good estimation of depth layers
1. Take current fragment
2. Reproject into prev frame
3. Take prev frame depth for that pixel, reproject them into current frame
4. Compute offset such that prev frame fragments will reproject into current pixel based on derivatives.
5. Use as new frame's bins.

Can be a separate pass that estimates layers

https://selgrad.org/publications/2017_hpg_HBSS.pdf
https://research.nvidia.com/sites/default/files/pubs/2016-06_Deep-G-Buffers-for/Mara2016DeepGBuffer-extended-bright.pdf
https://research.nvidia.com/sites/default/files/pubs/2015-08_An-Adaptive-Acceleration/AcceleratedSSRT_HPG15.pdf
https://outerra.blogspot.com/2012/11/maximizing-depth-buffer-range-and.html
https://community.khronos.org/t/linearize-the-depth-buffer/72335/8
https://www.researchgate.net/publication/267453578_Fast_Data_Parallel_Radix_Sort_Implementation_in_DirectX_11_Compute_Shader_to_Accelerate_Ray_Tracing_Algorithms
https://www.youtube.com/watch?v=AzXEao-WKRc&ab_channel=DantheMan

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
Use that during interpolation. Given face id, take corresponding coverage values in neighborhood of sample, bilinear interpolate that, and use as a weight for the sampled value.


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

Lets say we had an object transform, such that point at (x,y) now appears at (u,v) along a direction w. We can reproject with a factor that account for attenuation due to closer proximity, by computing (u,v) by combining the scene transform, initial (x, y) and the direction w along which we measure change. In the same fasion as camera reprojection, we can compare face ids or distances to cutoff dissocclusion artifacts.