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

https://www.reddit.com/r/GraphicsProgramming/s/h1GI7rb6nq

https://www.semanticscholar.org/paper/Real-time-multiply-recursive-reflections-and-using-Ganestam-Doggett/0c4cebce66ce22b9253b2674650eeeab4fae4879
https://www.semanticscholar.org/paper/Generating-exact-ray-traced-animation-frames-by-Adelson-Hodges/987c5e122a777efe3fb997caa88dda5ae48477c7

Assets: https://polyhaven.com/
https://gabrielgambetta.com/computer-graphics-from-scratch/05-extending-the-raytracer.html
You can check if neighboring pixels are occluded, and check that object first, assuming that for most rays they will hit the same object/triangle as the neighbor.

what we basically want is to find representative sample in the geometry buffer, the better the sample, the more stable the image will be.

https://wickedengine.net/2022/05/derivatives-in-compute-shader/
derivatives in compute shader, can be used for anisotropic filtering there.

https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/

raytracing in one weekend series
https://raytracing.github.io/books/RayTracingInOneWeekend.html#diffusematerials
https://raytracing.github.io/books/RayTracingTheNextWeek.html#motionblur
http://raytracing.github.io/books/RayTracingTheRestOfYourLife.html

pbr book
https://www.pbr-book.org/4ed/contents

https://habr.com/en/articles/440488/

https://casual-effects.blogspot.com/2014/04/fast-terrain-rendering-with-continuous.html
https://yehar.com/blog/?p=1495
https://www.realtimerendering.com/#books-small-table
