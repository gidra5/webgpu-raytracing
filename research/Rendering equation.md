https://perso.crans.org/sylvainrey/Biblio%20Physique/Physique/Optique/%5BMax%20Born%5D%20Principles%20of%20Optics%20-%20Electromagnetic%20Theory%20of%20Propagation%2C%20Interference%20and%20Diffraction%20of%20Light.pdf?utm_source=chatgpt.com

To describe visual appearance of the scene we have to first define ground rules - what are we actually trying to simulate. Any rendered scene is characterized by scattering events, that happen on the way between a light source and camera sensor. Thus we need to consider when scattering happens and what happens during and in-between scattering.

This process can be view in multiple different perspectives:
* An integral over all paths from source to sensor
* A differential equation considering change of radiosity at a point.
* An explicit equation considering radiosity at a point incoming from a particular direction.
* Wave equation capturing wave-optics effects like diffraction.

Each formulation is suitable to a different rendering problem domain, thus a combination of such formulations if usually required for closed form solutions and efficient evaluation.

When it is known a function depends on more arguments than shown, assume it is explicitly passed through.

Ideally we want to evaluate such wave-optical effects, [like](https://imadr.me/pbr/): 
- Reflection / Refraction / Transmission
- Diffraction
- Interference
- Polarization
- Dispersion
- Fluorescence
- Phosphorescence
# Physical Formulations
https://www.youtube.com/watch?v=FS8NotZ3diY
https://en.wikipedia.org/wiki/Radiative_transfer
https://math.stanford.edu/~papanico/pubftp/TRANSPORT.pdf
https://habr.com/ru/articles/958088/
https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer
https://d38rqfq1h7iukm.cloudfront.net/media/papers/Jakob2010Radiative.pdf
https://hal.science/hal-00002848/file/RTVI_OC_Revised.pdf
https://arxiv.org/pdf/2001.10050
https://arxiv.org/pdf/2401.09511
https://arxiv.org/pdf/1412.4371

The process raytracing simulates is based on Radiative Transfer Equation (RTE). 
Let's consider a radiosity along fragment of the ray $x(s)$. From energy conservation we have that the change in radiance is equal to the energy source input and scattered radiance minus absorbed fraction of current radiance:
$$\partial_sL=L_e+\sigma_sS-\sigma_aL$$
Where
* $\partial_sL$ is the change in radiance along the ray.
* $L_e$ is the emission of radiance in direction of the ray.
* $S$ is the change in radiance due to scattering.
* $\sigma_{a}$, $\sigma_{s}$, $\sigma_{t}=\sigma_{a}+\sigma_{s}$ are the absorption, scattering, and extinction coefficients along the ray respectively.
We can split scattering term into in-scattering and out-scattering. Since the in-scattering at one point is part of out-scattering at the other, the coefficient for both of these terms is $\sigma_s$. We can think of it as the in-scattered energy completely replacing the out-scattered, keeping the overall flow of energy due to scattering zero. 
The out-scattering is proportional to the current radiance, which gives us:
$$S=S_{in}-L$$
With this we can group out-scattered and absorbed radiance into an extinction term:
$$\partial_sL=L_e+\sigma_s(S_{in}-L)-\sigma_aL=L_e+\sigma_sS_{in}-(\sigma_a+\sigma_s)L=L_e+\sigma_sS_{in}-\sigma_tL$$
The in-scattering term can be expressed as an integral over all incoming radiance:
$$S_{in}=\intop\nolimits_{S^2} p\left(\omega_{i}\to\omega\right)L\left(\omega_{i}\right)\mathrm{d}\omega_{i}$$
Where the function $p$ is often called phase function, and it describes what portion the incoming light from a particular direction is scattered into the current one.

The $p$ must obey normalization constraint:
$$\intop\nolimits_{S^2}p\left(\omega_{i}\to\omega\right)\mathrm{d}\omega_{}=1$$

The ray $x(s)$ in general depends on refractive index, spacetime metric/curvature, polarization, free-space light speed and frequency of the ray, besides the regular position and direction. Direction, refractive index, and frequency can be combined into a wave-vector. If we also introduce uncertainty, position and wave-vector also gain variance.
All of these parameters' evolution depend on each other, which means the ray's path is inherently defined by all of them, basically tracing a ray in the whole phase space, not only in regular space.

Solving RTE in terms of the ray parametrization we get the integral form:
$$L\left(x\right)=T(x\to x_{surf})L_{surf}( x_{surf})+\int_{0}^{t_{surf}}T\left( x\to x_{t}\right) [L_e+\sigma_sS_{in}]( x_t)\mathrm{d}t$$
Where $\boldsymbol x_{surf}$ and $\boldsymbol x_{t}$ are shorthand for $x_{t}=x(s+t)$ and $x_{surf}=x(s+t_{surf})$, and [transmittance](https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance) $T\left(x\to x_{t}\right)$ is the following:
$$T\left(x\to x_{t}\right)=e^{-\intop\nolimits_0^{t}\sigma_{t}\left(x_{u}\right)\mathrm{d}u}$$

$t_{surf}$ is the boundary condition for the surface hit of a ray and entirely depends on the actual scene.

The transmittance also satisfies some properties such as:
$$T(x\to x)=1$$
$$T(x\to z)=T(x\to y)T(y\to z)$$
$$T(x\to z)=T(z\to x)$$

The $L_{surf}$ term is the one expressed in a standard rendering equation. But it is usually simplified to only consider reflected light. The exact form is as follows:
$$L_{surf}\left(x\right)=L_{e} + (1-\sigma_{r})
Q_{surf}+\sigma_{r}\intop\nolimits_{S^2}f(x_{i} \to x)(n\cdot\omega_{i})L(x_{i})\mathrm{d}x_{i}$$
Where
* $\boldsymbol x$ is the ray origin
* $\omega$ is the ray direction
* $L$ the incoming radiance from the direction $\omega$
* $L_e$ the emitted radiance in the direction $\omega$
* $\boldsymbol n$ is the normal of the surface.
* $f$ is the bidirectional scattering distribution function - probability density for scattering from the direction $\omega_{i}$ in the direction $\omega$.
* $Q_{surf}$ is the re-emitted absorbed light in the direction $\omega$.
* $\sigma_{r}\in[0,1]$ is the absorption factor in the direction $\omega$.

The $f(\boldsymbol x,\omega_{i}\to\omega)$ must also obey normalization constraint:
$$\intop\nolimits_{S^2}f\left(\omega_{i}\to\omega\right)(\boldsymbol n\cdot\omega_{i})\mathrm{d}\omega_{}=1$$
We may further collapse recursive integral form above into a [path integral](https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf):
$$
L=\int_Pf(\bar x)d\mu(\bar x)
$$
Where:
* $P$ is the space of all paths between sensor and light source
* $\bar x$ is a single path
* $f(\bar x)$ is path throughput
* and $\mu$ is the path measure that encodes different differential terms.

The path throughput is calculated based on the intermediate nodes:
$$
f(\bar x)=\sum_{i=0}^{k-1} L_e(x_i\to x_{i+1})\left[
\prod_{j=i}^{k-1}T_v(x_j\to x_{j+1})f_{j+1}
\right]T_v(x_{k-1}\to x_k)$$
$$T_v(x_i\to x_j)=G(x_i\to x_j)T(x_i\to x_j)$$
$$L_e(x_i\to x_j)=\int_{x_i}^{x_j} L_e(x)T(x\to x_j)dx$$
Where $f_j$ is the interaction function corresponding to the scattering at $x_j$ and $G$ is the geometry term encoding cosine attenuation factors.

[Helmholtz principle](https://en.wikipedia.org/wiki/Helmholtz_reciprocity) also states that rays following the same path in opposite directions experience the same events. This allows us to choose which way do we measure light - from camera to source or the other way around.
There are only two functions that depend both on incoming and outgoing light directions - $p(\omega_{i}\to\omega)$ and $f(\omega\to\omega_{i})$. Thus we impose additional constraints on these functions:
$$\sigma_s(\omega_i)p\left(\omega_{i}\to\omega\right)=\sigma_s(\omega)p\left(\omega\to\omega_{i}\right)$$
$$\sigma_r(\omega_i)f\left(\omega_{i}\to\omega\right)=\sigma_r(\omega)f\left(\omega\to\omega_{i}\right)$$
## Ray parameters
### Polarization
https://en.wikipedia.org/wiki/Polarization_(waves)

Since light is a wave, it oscillates around the ray direction. We can view the oscillations in the plane perpendicular to the ray. We can describe a polarization state in this plane with two values, each containing a phase and amplitude. Complex numbers give a natural way of encoding the state as a 2D complex vector:
$$s=\left[
A_xe^{i\varphi_x}\atop
A_ye^{i\varphi_y}
\right]=\left[
a_x+ib_x\atop
a_y+ib_y
\right]=a+ib$$
In that plane we can arbitrarily choose an orthogonal basis, that will be used to describe polarization state. The polarization states along each basis are considered basis polarizations. In principle any two orthogonal polarization states can be chosen as the basis. These satisfy the following constraint:
$$\left<s_1, s_2\right>=\bar{s}_{1x}s_{2x}+\bar{s}_{1y}s_{2y}=0$$
Thus the basis can be chosen to simplify the particular computations.

The wave may also be unpolarized, which means that no single polarization can be distinguished in the ray. In that case we can describe it with statistical values describing variations and correlations in polarization state over time. It can be described with coherence matrix, averaged over time:
$$J=s\bar{s}^T=\left[
s_x\bar{s}_x\ s_x\bar{s}_y\atop
s_y\bar{s}_x\ s_y\bar{s}_y
\right]$$
Equivalently, we can represent this matrix as 4 parameters called Stokes vector:
$$S=\left[\array{J_{11}+J_{22} \cr J_{11}-J_{22}\cr J_{12}+J_{21}\cr i(J_{12}-J_{21})}\right]$$
This representation allows for easier visualization and computations. Additionally, the $S_1$ parameter represents total intensity of the ray, which is convenient for rendering.

The light may also be partially polarized, with a fraction of purely polarized light $p$. The other fraction is unpolarized light, that is described by the average coherence matrix. We may assume that unpolarized light has absolutely no correlations, which makes $S_{1,2,3}=0$, and allows us to split the stokes vector into polarized and unpolarized parts. Otherwise we need 3 more values describing which fractions of each parameter belong to unpolarized matrix' parameters. Or we can literally track two such vectors.

One useful basis is based on the plane of incidence, which is the plane defined by incoming propagation direction and surface normal. The term that is parallel to the plane is called *p-like*, and the perpendicular one is *s-like*. Refracted and reflected amount of polarized light from Fresnel equations are defined in terms of this basis.

With this the radiance function $L$ is a Stokes vector, and the RTE now involves matricies instead of simple coefficients:


### Wave equation
https://ssteinberg.xyz/2023/03/27/rtplt/
https://en.wikipedia.org/wiki/Wave_equation
https://dl.acm.org/doi/pdf/10.1145/3450626.3459791
We can also look at this as a wave propagation problem in an absorbing and emitting medium. Let's consider a function $\psi(x, t)$ in electromagnetic field, satisfying the wave equation:
$$\frac 1 {c^2} \frac {\partial^2\psi}{\partial t^2}+n \frac {\partial\psi}{\partial t}=\nabla^2\psi+S$$
It is *damped*, which describes absorption with rate $n$, and has a source $S$, describing emission. 
From the wave function $\psi$ we can define a Wigner distribution function (WDF):
$$W(x,k)=\frac 1 {(2\pi)^3}\int\bar{\psi}(x-\frac 1 2x')\psi(x+\frac 1 2x')e^{-ix'\cdot k}dx'$$
Where a new parameter $k$ is the wave-vector.

A [wave-vector](https://en.wikipedia.org/wiki/Wave_vector) encodes a direction and a frequency of the wave at some point. We can define it as follows:
$$k=\omega\frac {2\pi\eta} {\lambda}=\omega\ 2\pi\nu\ \eta$$
Where $\omega$ is the direction of propagation, $\nu$ is the frequency, $\eta$ is the refractive index of the medium.
With that, WDF describes the direction spread of $\omega$ for a particular frequency $\nu$ at a given position $x$. We can recover the wave function $\psi$ from it up to a global phase shift, which gives a complete description of light.

We can use a gaussian WDF with the following shape:
$$g_{\beta,\rho}(x, k;x_0, k_0)=\frac 1 {\pi^3}e^{\frac {q(x-x_0,k-k_0)} {\beta^2}}$$
$$q(x,k)=\beta^2(\beta|k|-\rho|x|)^2+|x|^2$$
Where $x_0$, $k_0$ are the mean position and wave-vector, $\beta$ is the initial spatial variance of the distribution, and $\rho$ is the correlation parameter, encoding polarizations state. In general, $\rho$ and $\beta$ can be matrices, encoding anisotropy.
It is also normalized: $\int g(x,k)dx\ dk=1$
This distribution represents a *generalized ray*, which allows us to apply regular raytracing approaches, while still getting wave-optics accurate result.

We can derive a corresponding wave function for a given $g_{\beta,\rho}$:
$$\psi_{\beta,\rho}(x;x_0,k_0)=\frac 1 {(\pi\beta^2)^{3/4}}e^{q'(x-x_0,k_0)}$$
$$q'(x,k)=ik\cdot x-\frac 1 {2\beta^2}(1-i\rho)|x|^2$$
With that the measured intensity is computed as follows:
$$L=\int W(x,k)W_D(x,k)dx\ dk$$
Where $W_D$ is the detector's WDF.
If we assume our detectors are classical photoelectric detectors, the $\rho$ is 0. Then detector's WDF is computed as follows:
$$W_D(x,k)=\int_D \alpha(x_0) g_{\beta,0}(x,k;x_0)dx_0$$
Where $\alpha$ is the detection efficiency, and $D$ is the spatial extent of the detector.

Substituting into $L$ and swapping order of integration we get the following expression:
$$L=\int \alpha(x_0) \int W(x,k)g_{\beta,0}(x,k;x_0)dx\ dk\ dx_0$$

We can then apply the ordinary approach of measuring backwards by evolving $W_D$ under time-reversed dynamics. That approach can be characterized as *weakly local* (not a point, but a gaussian in phase space), *linear* (the "rays" do not interfere) and *complete* (fully describes wave-optics).

Consider the WDF $W_s$ of the light source. It interacts with the scene, until it reaches the detector. At that point the WDF transformed into $K\{W_s\}$ by the interaction kernel $K$ as follows:
$$K\{W_s\}(x,k)=\int K(x',k',x,k)W_s(x',k')dx'dk'$$
Where $K$ is a kernel representing the change in light distribution.

Since we want to apply this transform in reverse time, the directions change $k\to-k$, and phases get conjugated. Which means that we can express $L$ equivalently as follows:
$$L=\int \alpha(x_0) \int W(x,k)K^{-1}\{g_{\beta,0}\}(x,k;x_0)dx\ dk\ dx_0$$
Then we further integrate over $k_0$, $\beta$ and $\rho$.
For a given WDF $W$ and light source WDF $W_s$ we can compute measured light as follows:
$$L_s=\int W(x,k)W_s(x,k)dx\ dk=\int g_{\beta,\rho}(x,k)W_s(x,k)dx\ dk=\frac 1 {(2\pi)^3}\left|\int \psi_s(x)\bar{\psi}_{\beta,\rho}(x)dx\right|^2$$

The interaction kernels can be classified in two categories:
* Simple linear optics interactions. The same interactions that are simulated by classical raytracing following RTE.
* Diffractive interactions. These are the interactions that heavily depend on interference of the waves, such as scattering by rough surfaces.

Reflection/refraction and free-space propagation fall under simple interactions, which makes them easy to define:
$$K_{free}\{g_{\beta,\rho}(x_0, k_0)\}=g_{\beta',\rho'}(x_0+\bar zk_0, k_0)$$
$$K_{r}\{g_{\beta,\rho}(k_0)\}=Rg_{\beta,\rho}(reflect(k_0))$$
$$K_{t}\{g_{\beta,\rho}(k_0)\}=(1-R)g_{\beta,\rho}(refract(k_0))$$
$$\bar z = z /|k_0|$$
$$\beta'^2=\beta^2 + \bar z(2\rho+2\bar z\sigma_k)$$
$$\rho'=\rho+2\bar z\sigma_k$$
$$\sigma_k=\frac {1+\rho^2}{2\beta^2}$$
Where $z$ is the propagation distance.

With that, rays have the following state:
* Mean wave-vector + variance
* Mean position + variance
* Polarization state (Stokes vector)

In general, any interaction that happens between rays and the scene can depend on all of them. But primarily it depends on wavelength, direction and polarization.

Reflection/refraction distribution s for angular extent
### Index of Refraction
Any material's optical response is fundamentally described by an index of refraction (IoR), which is a complex number $\eta(x, \omega, \lambda)=n+ik$ encoding both refraction ratio $n$ and extinction coefficient $k$. The real and complex parts are not independent, they follow [Kramers–Kronig relations](https://en.wikipedia.org/wiki/Kramers%E2%80%93Kronig_relations), since it is the result of a physical process, which makes it a [linear response function](https://en.wikipedia.org/wiki/Linear_response_function). That also means we can derive imaginary part from real, and vice versa. The absorption coefficient used in RTE can be expressed in terms of $k$:
https://en.wikipedia.org/wiki/Refractive_index#Complex_refractive_index
$$\sigma_a=\frac{4\pi k}{\lambda}$$
It is not unphysical if we also include explicit surface reemission, as long as we scale it down proportional to absorption coefficient. That way, whatever reemission happens due to volumetric absorption, it is not double counted.

IoR is [related](https://en.wikipedia.org/wiki/Refractive_index#Relative_permittivity_and_permeability) to a physical measures called electric permittivity and permeability:
$$\eta=\sqrt {\varepsilon\mu}$$
For non-magnetic materials $\mu$ can be ignored, simplifying relation to:
$$\eta=\sqrt {\varepsilon}$$
Since light is an electromagnetic wave, it directly interacts with electrons in the material. When there are free electrons, like in conductors, they absorb the wave and create current, which then dissipates into heat or gets reemitted. Thus, materials that absorb light are called conductors, and non-absorbing ones are dielectrics.

Permittivity, just like IoR, is a complex number, generally modeled as follows:
$$\varepsilon=\varepsilon'-i\sigma\lambda\kappa$$
Where $\sigma$ is conductivity, and $\kappa$ is a [constant](https://en.wikipedia.org/wiki/Relative_permittivity#Lossy_medium) depending on speed of light and permeability.

Following [Drude-Lorentz model](https://www.mdpi.com/2076-3417/11/21/9902), we can also describe permittivity as follows:
$$
\varepsilon=1-\frac {f_0\omega_p^2}{\omega(1-i\Gamma_0)}+\sum\frac {f_i\omega_p^2}{\omega_i^2-\omega^2-i\omega\Gamma_i}=1-\frac {f_0\omega_p^2\lambda}{2\pi-i2\pi\Gamma_0}+\sum\frac {f_i\omega_p^2\lambda^2}{(\omega_i\lambda)^2-i2\pi\lambda\Gamma_i-4\pi^2}
$$
Where $\Gamma_i$ is the damping constant related to the electron collision frequency, $f_i$ is the free-electron oscillator strength, $\omega_p$ is the plasma frequency, $\omega_i$ are the oscillation frequencies of the bound electrons.
We can compress all constants into single coefficients, and get the following relation in terms of wavelength:
$$\varepsilon=1-f_0\lambda+\sum\frac {f_i}{\lambda-\gamma_i}-\frac {f_i}{\lambda-i\Gamma_i}$$
That showcases that at large wavelengths permittivity is dominated by a linear term, but requires inverse proportional corrections at smaller values.

Fermat's principle - light travels shortest-time path

https://perso.crans.org/sylvainrey/Biblio%20Physique/Physique/Optique/%5BMax%20Born%5D%20Principles%20of%20Optics%20-%20Electromagnetic%20Theory%20of%20Propagation%2C%20Interference%20and%20Diffraction%20of%20Light.pdf?utm_source=chatgpt.com
### Birefringence
https://en.wikipedia.org/wiki/Birefringence
https://en.wikipedia.org/wiki/Huygens_principle_of_double_refraction
Dependance of refractive index on direction of the ray and its polarization.
### General relativity
https://docs.google.com/document/d/1Ueo_gLj2LiP7dUPGt_-ERMB3dszPQkrqaGAIKRV7omc/edit?tab=t.0

Gr describes relation between energy and space curvature.
For raytracing we only need to be able to evaluate metric tensor at a sample point, which allows us to follow geodesics.

If we introduce time dependance into RTE, we can simulate effects predicted by general relativity, like lensing, phase redshifts, time dilation and stretching. We may improve even further by tracing geodesics instead of regular rays, which would allow simulation of light bending in space.

https://en.wikipedia.org/wiki/Metric_tensor
First, lets look at a notion of a metric tensor. 
Suppose that $g$ is the metric tensor. We can think of it as a parametrization for the dot product:
$$a\cdot b=g(a, b)=a^TGb$$
Which implies a few things:
1. It is a n by n matrix, where n is the number of dimensions
2. It is completely described by a matrix $G$.
3. It inherits all the properties of a dot product (bilinear, symmetric)
4. $G$ is symmetric

Note that under a coordinate transformation from $x_n$ to $x_n'$ the $G$ matrix also changes proportional to a jacobian $J$:
$$G'=J^TGJ$$
$$J=\left[\frac {\partial x_i}{\partial x_j'}\right]$$
The $G$ itself can be viewed as a collection of partial derivatives:
$$G=\left[\frac {\partial s}{\partial x_i}\cdot \frac {\partial s}{\partial x_j}\right]$$
Where $s$ is a higher dimensional parametrization of the surface. Something like $s(u,v)=[x(u,v),y(u,v),z(u,v)]$. 

(?) The dot product here corresponds to the geometry of infinitesimals, and usually assumed to be flat surface. That means if we "zoom in" close enough, it will look like a flat surface, which implies a standard dot product.

https://en.wikipedia.org/wiki/Einstein_field_equations
Einstein's field equations:
$$R_{\mu\nu}-\frac 1 2 Rg_{\mu\nu}+\Lambda g_{\mu\nu}=\kappa T_{\mu\nu}$$
$$R=g^{\mu\nu}R_{\mu\nu}$$
$$R_{\mu\nu}=?$$
$$T_{\mu\nu}=T^{\alpha\beta}g_{\alpha\mu}g_{\beta\nu}=?$$
$$\kappa=\frac {8\pi G}{c^4}$$
Stress-energy tensor $T$, ricci curvature tensor $R$.

GR RTE:
https://arxiv.org/pdf/1612.02828
$$
\frac {dL'}{ds}(\lambda_0)-\sigma_t'(\lambda_0)L'(\lambda_0)=V'(\lambda_0)
$$
$$L'(\lambda_0)=G^3\lambda_0^3L(\lambda_0G)$$
$$V'(\lambda_0)=G^2\lambda_0^2RV(\lambda_0G)$$
$$\sigma_t'(\lambda_0)=G\lambda_0R\sigma_t(\lambda_0G)R^{-1}$$

special relativity is the general relativity with minkowski metric
minkowski metric:
$$G=\left[\matrix{
1& 0& 0& 0\cr
0& 1& 0& 0\cr
0& 0& 1& 0\cr
0& 0& 0& -1\cr
}
\right]$$


general relativity renderers
https://iopscience.iop.org/article/10.3847/0004-637X/820/2/105/pdf
https://github.com/hungyipu/Odyssey
https://arxiv.org/pdf/1207.4234
https://itp.uni-frankfurt.de/~hees/publ/kolkata.pdf
https://arxiv.org/pdf/astro-ph/0406401
https://arxiv.org/pdf/2304.03804
https://www.researchgate.net/publication/362968273_Skylight_a_new_code_for_general-relativistic_ray-tracing_and_radiative_transfer_in_arbitrary_space-times
https://arxiv.org/html/2507.16165v1?utm_source=chatgpt.com
https://github.com/ABHModels/raytransfer?utm_source=chatgpt.com
https://arxiv.org/pdf/2407.10431


special relativity
https://www.linkedin.com/pulse/rendering-relativity-webgl-javascript-dmitry-lavrov?utm_source=chatgpt.com
https://github.com/freemeson/specRelTrace?utm_source=chatgpt.com

# BSDF

https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
We define BSDF as the function that describes radiance transfer across a surface boundary. It describes how much light is reflected or exits from inside the object between an incoming and outgoing directions.

https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf

### Reflection and Refraction
chatgpt'd
Dielectrics and conductors are the base for any other materials. So we assume that any material is a mixture of such materials and local geometric properties of the surface.

We assume polarization ratios are $w_s$ and $w_p$, such that $w_s+w_p=1$.
First we compute the incidence angle $\theta_i$:
$$\cos\theta_{i}=\left|\omega_{i}\cdot n\right|$$
Then we can compute refraction angle $\theta_t$ with Snell's law:
$$\sin\theta_{t}= \frac{\eta_i}{\eta_o}\sin\theta_i$$

Then we can compute reflected and refracted components for s- and p- polarized light:
$$r_{s} = \frac{\left(\eta_{i}\cos\theta_{i}-\eta_{o}\cos\theta_{t}\right)}{\eta_{i}\cos\theta_{i}+\eta_{o}\cos\theta_{t}},t_{s}=\frac{\left(2\eta_{i}\cos\theta_{i}\right)^{}}{\eta_{i}\cos\theta_{i}+\eta_{o}\cos\theta_{t}}$$
$$
r_{p} = \frac{\left(\eta_{o}\cos\theta_{i}-\eta_{i}\cos\theta_{t}\right)}{\eta_{o}\cos\theta_{i}+\eta_{i}\cos\theta_{t}},

t_{p}=\frac{\left(2\eta_{i}\cos\theta_{i}\right)^{}}{\eta_{o}\cos\theta_{i}+\eta_{i}\cos\theta_{t}}
$$
Which the are combined to get the $R$ as follows:
$$R\left(\lambda,\theta_{i}\right)=w_s|r_s|^2+w_p|r_p|^2$$
$$T(\lambda,\theta_{i})=\frac{Real(\eta_o\cos\theta_t)}{Real(\eta_i\cos\theta_i)} (w_s|t_s|^2+w_p|t_p|^2)$$
We may also compute $T$ simply by conservation of energy from $R$:
$$T(\lambda,\theta_{i})=1-R(\lambda, \theta_i)$$
The transmittance usable in BSDF needs to be scaled by refractive index:
$$T_{BSDF}=T\left|\frac{\eta_o}{\eta_i}\right|^2$$
If computed $\sin \theta_t\ge 1$ , then Total Internal Reflection occurs, since $R$ reduces to 1. In that case $\cos\theta_i$ is purely imaginary value, which means components of $R$ have the following form:
$$
\begin{aligned}
r_{s,p}^2 &=\left|\frac{a-ib}{a+ib}\right|^2=\left|\frac{\left(a-ib\right)^2}{a^2+b^2}\right|^2 \\
&=\frac{\left|a^2-2iab-b^2\right|^2}{\left(a^2+b^2\right)^2}\\
&=\frac{\left(a^2-b^2\right)^2+\left(2ab\right)^2}{\left(a^2+b^2\right)^2} \\
&=\frac{a^4-2a^2b^2+b^4+4a^2b^2}{\left(a^2+b^2\right)^2} \\ &=\frac{\left(a^2+b^2\right)^2}{\left(a^2+b^2\right)^2} \\
&=1
\end{aligned}$$

Thus, in a case of perfectly smooth surface, BSDF is as follows:
$$\begin{aligned}
f_{s}\left(x,\omega_{i}\to\omega_{o},\lambda\right)

&=R\left(\lambda,\omega_{i}, n\right)\delta(\omega_o-reflect\left(\omega_{i},n\right))\\

&+T_{BSDF}\left(\lambda,\omega_{i}, n\right)\delta(\omega_o-refract(\omega_{i}, \eta(x, \omega_{i}, \lambda), n))
\end{aligned}$$
# Phase function
chatgpt'd
https://miepython.readthedocs.io/en/v2.3.1/01_basics.html
https://drive.google.com/file/d/1xIU8YB-R6iS2JHanA9v9P-3WbmqALxfe/view

The $1/4\pi$ factor is omitted from phase functions to unclutter a bit.

Phase function is the basis for volumetric rendering, since it describes generically how the light scatters in a volume.
For phase function there isn't any universal and simple model. The most precise formulation for the phase function is given by Mie theory, which requires high computational resources.
We describe participating media by particle radius $a$ and complex refractive index $n$.
If we assume particles to be much smaller than wavelength, we get Rayleigh scattering:
https://en.wikipedia.org/wiki/Rayleigh_scattering
$$\cos\theta=\omega_i\cdot \omega$$
$$\sigma_s(\lambda)=\left(\frac{2\pi}{\lambda}\right)^4\frac{8\pi a^6|n^2-1|^2}{3|n^2+2|^2}$$
$$p(\omega_i\to\omega)=\frac{3}{2}(1+\cos^2\theta)$$
Can be extended for anisotropic volumes:
chatgpt'd
$$p(\omega_i\to\omega)=\frac{3}{4}\frac{1+\rho\cos^2\theta}{1+\rho/3},\ \rho=\frac{1-\delta}{1+\delta}$$
Where $\delta(\lambda)\in[0,6/7]$ is the depolarization factor that accounts for molecular anisotropy
Can be approximated similarly to HG with $g$ factor:
chatgpt'd
$$p(\omega_i\to\omega)=\frac{3}{2}(1+\cos^2\theta)(1+g\cos\theta)$$
For particles larger or at the order of wavelength we would need to evaluate Mie equations. Instead we can get arbitrarily close approximation with a weighted sum:
https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions
$$p(\omega_i\to\omega)=\sum_{i=1}^{n}w_ip_i(\theta-\varphi_i, g_i),\ \sum_{i=1}^{n}w_i=1$$
Where:
* $w_i$ is the weights for each phase function.
* $p_i$ are the constituent phase functions.
* $\varphi_i$ is the phase shift to allow off-ray preferred scattering direction.
* $g_i\in[-1,1]$ is the anisotropy factor. Must be equal mean cosine value of the distribution.

The $p_i$ can be chosen arbitrarily, as long as constraints on the phase function are respected. Common choices are:
* Henyey–Greenstein phase function
$$p_{HG}(\omega_i\to\omega)=\frac{1-g^2}{(1+g^2-2g\cos\theta)^{3/2}}$$
* Cornette–Shanks Phase Function:
Note that it is equivalent to rayleigh when $g=0$
$$p_{CS}(\omega_i\to\omega)=\frac{3(1+\cos^2\theta)}{2(1+g^2)}p_{HG}=\frac{p_{rayleigh}\ p_{HG}}{(1+g^2)}$$
* Xiao-Lei Fan:
https://cornercodes.com/2020/11/04/mie-phase-functions-comparison/
Faster to compute due to removal of square-roots. Better approximates Mie for low $g$. Not physically based, which causes a worse result for larger $g$ values.
$$p_{XLF}(\omega_i\to\omega)=p_{CS}(1+g^2-2g\cos\theta)^{1/2}+g\cos\theta$$
* von Mises–Fisher distribution:
https://persci.mit.edu/pub_pdfs/translucency.pdf
It is found that mixing it with the HG results in better approximations. Allows to approximate sharp peaks in scattering. $\kappa$ is the parameter that controls the sharpness of peaks and plays similar role to $g$, but they are not the same.
$$p_{vMF}(\omega_i\to\omega)=\frac{\kappa e^{\kappa\cos\theta}}{\sinh\kappa}$$
$$g_{vMF}=\coth\kappa-1/\kappa\ge 0$$
* van de Hulst approximations
https://en.wikipedia.org/wiki/Anomalous_diffraction_theory
chatgpt'd
These are the asymptotic approximations for Mie phase function when particle size $a\gg1$ and $n-1\ll1$.
$$p(\omega_i\to\omega)=\frac{1}{\pi a^2}\left(\frac{J_1(2ka\sin\frac{\theta}{2})}{ka\sin\frac{\theta}{2}}\right)^2$$
$$k=\frac{2\pi n}{\lambda}$$
$$x=ka$$
$$J_1(2z)=\sum\limits_{n=0}^{\infty}\frac{(-1)^{n}}{n!(n+1)!}z^{2n+1}$$
$$J_1(z)=\frac{1}{\pi}\intop_{0}^{\pi}\cos(\tau-z\sin\tau)d\tau$$
$$J_1(z) \backsim \sqrt{\frac{2}{\pi z}}\cos(z-\frac{3\pi}{4})\ \ (z\to\infty)$$
With the scattering and absorption coefficients:
$$\sigma_e=N\pi a^2(2-\frac{4\sin p}{p}-\frac{4(1-\cos p)}{p^2})$$
$$p=2x(n-1)$$
Where $N$ is number of particles per unit volume.
# Geometry
Different effects happen and contribute significantly at different geometric detail scales, or have implications that make impractical approaches used for other scales.

We can split geometric detail into following categories:
1. Scene. The highest level in the hierarchy, describing the whole world's composition from distinct objects. Allows for crude approximations and efficient evaluation.
2. Macro-scale. Defines explicit object geometry with a combination of primitives, which are shaded and traced against explicitly. 
3. Meso-scale. Significant geometric detail along the surface of an object. Usually mapped from object surface domain to a 2D domain defining geometric detail with less data and spatial variance.
4. Micro-scale. A detail, that is imperceptable at any given rendering resolution, but yields significant shading contributions due to self-shadowing and multiple bounces in vicinity of the surface.
5. Nano-geometry. A wavelength scale detail, that is insignificant to self-shadowing and multiple-bounce lighting, but introduces various dispersion effects due to wave optics.

Each scale requires distinct approaches to allow practical and universal shading of any surface.

The classification is applicable not only to surface detail, but also to volumetric detail.
## Meso-geometry
https://research.nvidia.com/sites/default/files/pubs/2016-02_Real-time-Rendering-of/ZirrKaplanyan_MultiscaleI3D2016.pdf
Meso-geometry is any geometry on the scale between micro and macro. It is detail that is impractical to describe with macro-geometry (wasteful and expensive), but not as fine as microgeometry, that we can treat it completely statistically.

While useful, overusing it may create too much visible artifacts due to it being limited to the primitive's plane\volume. While it is possible to trace against meso-geometry, it is often impractical. Since we don't recognize global texture of such geometry as distinct, we just need the pattern to be distinct locally, which allows this detail to be procedurally generated.

Many techniques to describe it rely on procedural or tabulated data, or textures.
### Surface detail
https://userpages.cs.umbc.edu/olano/papers/lean/lean.pdf
https://inria.hal.science/file/index/docid/967847/filename/LEADRmapping.pdf

https://www.youtube.com/watch?v=43Ilra6fNGc (why normal maps and height maps produce different shading? Seems to have rotated normals which causes incorrect shading)
https://learnopengl.com/Advanced-Lighting/Parallax-Mapping
We can represent additional surface detail on the single triangle with a variety of displacement maps, such as height/bump/parallax map. A similar effect is achieved with normal map, which is not strictly a displacement map, since it only *implies* displacement due to its effect on shading. Shell maps

### Thin geometry
Another class of meso-geometry is thin geometry like cloth, fur and hair, that adds further complications to the modelling.
### Fibers
https://shuangz.com/courses/cloth-sa12/cloth-sa12.pdf
A fiber, represents "building block" of any cloth. A fiber then woven against itself multiple times to create plies. The same way plies form yarns, and yarns form even deeper textiles. In some sense this process can be indefinite in both scaling up and down the fiber size.  Scaling up is both impractical and unrealistic, in a sense that we don't encounter usually such extreme cases. If needed they might be better modelled with explicit geometry.

fiber rendering
https://dl.acm.org/doi/pdf/10.1145/3023368.3023372
We can split how fibers are woven in the following categories: 
* migration - when there is $n$ fibers in a ply which twist around its center.
* loop - the fibers that were accidentally pulled out.
* hair - fibers that have open endpoints that stick outside.

cloth
https://s3.amazonaws.com/srmweb/publications/IrawanThesis.pdf
Woven cloth is constructed by interlacing two sets of parallel yarns, known as the warp and weft, at right angles to each other. In the process of weaving, warp yarns are raised or lowered and weft yarns (also known as fillings) are inserted in the space that resulted. Figure 2.1 shows a loom with the warp yarns before the weft yarns are inserted. The pattern in which the warp and weft are interleaved varies greatly, but the majority of fabrics are made in one of the three simplest weave patterns: plain weave, twill, and satin.

https://dl.acm.org/doi/pdf/10.1145/74333.74359
### Fur
A sparse field of independent fibers.
http://kunzhou.net/2013/fur-rendering-tvcg.pdf
### Hair
A not so sparse field of independent fibers.
https://www.pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair
https://www.cemyuksel.com/research/hairmesh_rendering/
https://www.cs.cornell.edu/~srm/publications/SG03-hair-lr.pdf
### Glint
glints
https://cseweb.ucsd.edu/~ravir/glints.pdf
https://rgl.epfl.ch/publications/Zeltner2020Specular
https://igg.unistra.fr/People/chermain/real_time_glint/
https://rgl.epfl.ch/publications/Loubet2020Slope
https://hal.science/hal-02364885/file/glint_ms.pdf
https://igg.unistra.fr/People/chermain/assets/pdf/Chermain2021ImportanceSampling.pdf
https://cs.uwaterloo.ca/sites/ca.computer-science/files/uploads/files/cs-2024-02.pdf
https://cs.uwaterloo.ca/sites/default/files/uploads/documents/cs-2024-02_0.pdf
https://ggx-research.github.io/publication/2023/06/09/publication-glints.html

more references
https://www.semanticscholar.org/paper/Importance-Sampling-of-Glittering-BSDFs-based-on-Chermain-Sauvage/f9f6ddb7b159264c9510a51db96321bdea68017f
### Scratch
https://rgl.epfl.ch/publications/Werner2017Scratch
https://inria.hal.science/hal-01321289/document
### Foam
https://hal.science/hal-04220006/file/micrograin_HAL.pdf
## Micro-geometry
While general RTE fully describes the radiance, it is unfeasible to render the micro details of objects. Besides unpracticality, such fine details are also imperceivable, since all of the detail is in a single pixel area, which is averaged in the final render. Thus it is a great place for statistical methods that describe microgeometry properties statistically.
In that case for every sample point $x$ we evaluate a statistical model of properties in an infinitesimal volume at that point, which simulates averaged result of fine details in both participating media and surface. 
There were developed two theories that give tools to handle both cases.
Together with broad scattering simulated in raytracing directly, it gives a complete description of radiance in the scene.
### Microfacet theory
https://d1qx31qr3h6wln.cloudfront.net/publications/microfacet-theory-non-uniform-heightfields_1.pdf
https://jcgt.org/published/0003/02/03/paper.pdf
The fresnel terms define reflection and transmission for ideal smooth surfaces. But that misses the imperfection of real world. Lets define a map from surface coords to world coords $H: R^2\to R^3$. If we assume that for a local patch $A$ the function $H$ is a heightmap, we can apply microfacet theory.

We define geometric surface properties as a combination of two functions:
* $D(x,h, n, t)$ - the Normal Distribution Function (NDF). The fraction of normals that is aligned with $h$.
* $G(x,\omega,h, n, t)\in[0,1]$ - the masking function. Describes a fraction of normals $h$ that is visible from direction $\omega$.
The parameters $n$ and $t$ are the geometric normal and tangent vectors, $h$ is a normal that would reflect/refract the $\omega_i$ into $\omega_o$, also called a half-vector.  Together these allow modelling a single successful scattering event at the surface from $\omega_i$ to $\omega_o$.

We also need to apply correction factors to first transform incident irradiance onto the microsurface and then transform the scattered radiance back to the macrosurface, because both irradiance and radiance are measured relative to a surface’s projected area.

They add up emitted light proportionally, over all possible normals, producing the following definition (some parameters omitted for compactness):
$$
\begin{aligned}
f &=\intop\nolimits_{H}\left|\frac{\omega_i\cdot m}{\omega_i\cdot n}\right|\frac{f_m}{{|\omega_o\cdot m|}}\left|\frac{\omega_o\cdot m}{\omega_o\cdot n}\right|D(m)G(m)dm\\

&=\intop\nolimits_{H}
\left|\frac{(\omega_i\cdot m)(\omega_o\cdot m)}{(\omega_o\cdot m) (\omega_i\cdot n)(\omega_o\cdot n)}\right|f_mD(m)G(m)dm\\

&=\intop\nolimits_{H}
\left|\frac{\omega_i\cdot m}{(\omega_i\cdot n)(\omega_o\cdot n)}\right|f_mD(m)G(m)dm\\

&=\frac{1}{|\omega_i\cdot n| |\omega_o\cdot n|}\intop\nolimits_{H}
|\omega_i\cdot m|f_mD(m)G(m)dm\\

\end{aligned}
$$
$$f_m=R(\omega_o, m)\delta(\omega_i - \omega_r)+ T(\omega_o, m)\delta(\omega_i-\omega_t)$$
where $\omega_r=reflect(\omega_o, m)$ and $\omega_t=refract(\omega_{o}, m, \eta)$.

These use two functions that compute refraction and reflection directions with the following formulas:
$$reflect(\omega_o, n) = \omega_o - 2(\omega_o\cdot n)n$$
$$k=1-\eta^2(1-(n\cdot \omega_o)^2)$$
$$refract(\omega_o, n, \eta) = \eta I - (\eta (n \cdot \omega_o) + \sqrt{k})\ n$$
https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml

We can apply change of variables theorem for delta-function and get the following expression for the $f_m$:
$$
f_m=R(\omega_o)\delta(m-h_r)\left\|\frac{\partial \omega_r}{\partial \omega_o}\right\|+T(\omega_o)\delta(m-h_t)\left\|\frac{\partial \omega_t}{\partial \omega_o}\right\|
$$
$$\left\|\frac{\partial \omega_r}{\partial \omega_o}\right\|=\frac{1}{4|\omega_o\cdot h_r|}$$
$$\left\|\frac{\partial \omega_t}{\partial \omega_o}\right\|=\frac{\eta_o^2|\omega_o\cdot h_t|}{(\eta_i(\omega_i\cdot h_t) + \eta_o(\omega_o\cdot h_t))^2}$$
$$h_r=\frac{\omega_i+\omega_o}{|\omega_i+\omega_o|}$$
$$h_t=-\frac{\eta_i\omega_i+\eta_o\omega_o}{|\eta_i\omega_i+\eta_o\omega_o|}$$
The $h_r$ and $h_t$ are the normals that would reflect/refract $\omega_i$ into $\omega_o$. Also note that for $h_r$ we have $\omega_i \cdot h_r = \omega_o \cdot h_r$ by definition which would allow us to cancel out the factor of $\omega_i \cdot m$ below.

With that we can eliminate the integral entirely by the definition of delta-function:
$$\int\delta(x-y)f(x)dx=f(y)$$
$$\begin{aligned}
f &=\frac{1}{|\omega_i\cdot n| |\omega_o\cdot n|}\intop\nolimits_{H}
|\omega_i\cdot m|
(R(\omega_o)\delta(m-h_r)\left\|\frac{\partial \omega_r}{\partial \omega_o}\right\|+T(\omega_o)\delta(m-h_t)\left\|\frac{\partial \omega_t}{\partial \omega_o}\right\|)
D(m)G(m)dm\\

&= \frac{1}{|\omega_i\cdot n| |\omega_o\cdot n|}(R(\omega_o)D(h_r)G(h_r) 
|\omega_i\cdot h_r| \left\|\frac{\partial \omega_r}{\partial \omega_o}\right\| +  T(\omega_o)D(h_t)G(h_t) 
|\omega_i\cdot h_t| \left\|\frac{\partial \omega_t}{\partial \omega_o}\right\|) \\


&= \frac{1}{|\omega_i\cdot n| |\omega_o\cdot n|}(R(\omega_o)D(h_r)G(h_r) 
|\omega_i\cdot h_r| \frac{1}{4|\omega_o\cdot h_r|} 
+ T(\omega_o)D(h_t)G(h_t) \frac{\eta_o^2 |\omega_i\cdot h_t| |\omega_o\cdot h_t|}{(\eta_i(\omega_i\cdot h_t) + \eta_o(\omega_o\cdot h_t))^2}) \\

&=\frac{1}{|\omega_i\cdot n| |\omega_o\cdot n|}(\frac{R(\omega_o)D(h_r)G(h_r)}{4}
+ T(\omega_o)D(h_t)G(h_t) 
\frac{\eta_o^2 |\omega_i\cdot h_t| |\omega_o\cdot h_t|}{(\eta_i(\omega_i\cdot h_t) + \eta_o(\omega_o\cdot h_t))^2}) \\

\end{aligned}$$

https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf

#### Normal Distribution Function
chatgpt'd
https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory

The constraints on $D$:
1. $D$ is not negative: $D\ge0$
2. $D$ produce the same (signed) projected area as the macrosurface for any direction $v$: $$\intop\nolimits_{H^2}(\boldsymbol v\cdot\omega)D\mathrm{d}\omega=(v\cdot n)$$
3. $D$ total area must be at least as large as the macrosurface: $$\intop\nolimits_{H^2}D\mathrm{d}\omega\ge1$$
4. Is zero outside hemisphere and at the boundary
5. Sometimes it is required that $D(h)=O(1/\cos^3\theta_h)$ or slower.

It is also useful to define a Visible Normal Distribution Function:
$$D_{\omega}(m)=\frac{G(\omega,m)\left<\omega\cdot m\right>D(m)}{\int_HG(\omega,m')\left<\omega\cdot m'\right>D(m')dm'}=\frac{\left<\omega\cdot m\right>}{\omega\cdot n}G(\omega,m)D(m)$$
#### Masking function
The constraints on $G$:
1. $G$ is smooth
2. As $n\cdot\omega\to0$, $G\to0$
3. Proper distribution of normals must project onto $\omega$ the same way as the macro surface. With that we expect that physically plausible distributions must satisfy:
$$\int_HD(m)G(\omega, m)\left<\omega\cdot m\right>dm=\left<\omega\cdot n\right>$$$\left<\omega\cdot m\right>=max(\omega\cdot m,0)$
Where $(x>0)$ is Heaviside function, that is 1 whenever the condition is true.
#### Smith's model
https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
https://jcgt.org/published/0003/02/03/paper.pdf

We can simplify computation of $G$ by making a single assumption that the masking is independent of normal. That means that there is no correlation between the height (or the normal) at one point of the microsurface and the height (or the normal) at any neighboring point, even the closest ones. The material conceptually turns from a connected surface into an opaque soup of little surface fragments that float in space. A consequence of this simplification is that masking becomes independent of the microsurface normal, which allows us to move $G$ from the integral above and solve for it:
$$G(\omega)=\frac{\left<\omega\cdot n\right>}{\int_HD(m)\left<\omega\cdot m\right>dm}$$
This is _Smith’s approximation_. Despite the rather severe simplification, it has been found to be in good agreement with both brute-force simulation of scattering on randomly generated surface microstructures and real-world measurements.

We can also express $G$ in terms of $\Lambda$, the expected number of occluding events:
$$G(\omega)=\frac{1}{1+\Lambda(\omega)}$$
$\Lambda$ arises naturally in the derivation of masking in the slope domain $P_2$. The exact definitions for $\Lambda$ are as follows:
$$\Lambda(\omega)=\int_{\cot\theta}^{\infty}\int_{-\infty}^{\infty}P_2(x, y)(x\tan\theta-1)dydx=\int_{\cot\theta}^{\infty}P(x)(x\tan\theta-1)dx$$
Where $P$ is the slope distribution in the view direction:
$$P(x)=\int_{-\infty}^{\infty}P_2(x, y)dy$$
Where $P_2$ is the slope distribution of the microfacets, related to the NDF as follows:
$$P_2(\bar{m})d\bar{m}=(m\cdot n)D(m)dm$$
$$D(m)=\frac{P_2(\bar{m})}{(m\cdot n)^4}$$
$$\bar{m}=-\frac{[m_x,m_y]}{m_z}=-\tan\theta_m[\cos\phi_m,\sin\phi_m]$$
#### Masking-shadowing function
https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
If we only account for a single scattering event, we should also account for shadowing of outgoing ray. If we assume independence of these two processes, we get:
$$G_s(\omega_i,\omega_o, m)=G(\omega_i, m)G(\omega_o, m)$$
While simple, it can underestimate visibility of peaks and valleys, which causes darkening at some angles.

If the heights are normally distributed, we can extend Smith's formulation to account for shadowing, allowing less conservative estimation:
$$G_s(\omega_i, \omega_o)=\frac{1}{1+\Lambda(\omega_i)+\Lambda(\omega_o)}$$

In particular, both guarantee reciprocity of the resulting BSDF.

#### Stretch invariance
https://jcgt.org/published/0003/02/03/paper.pdf
Some distributions allow for an easy extension to the anisotropic masking function, since they are invariant under stretching in the following sense:
$$P_2(\bar{m},\alpha)=\frac{1}{\lambda_x\lambda_y}P_2(\frac{\bar{m}}{\lambda},\frac{\alpha}{\lambda}),\text{ for any } \lambda>0$$
Intuitively it means that we can stretch the distribution however much we want, the shape will not change. In that case they can be expressed in terms of a single dimensional distribution $f$:
$$P_2(\bar{m},\alpha)=\frac{1}{\alpha_x\alpha_y}f(\left|\frac{\bar{m}}{\alpha}\right|)$$
When $\alpha_x=\alpha_y=\alpha$ we call it isotropic distribution. 
Consider the $\Lambda$ function with invariance and isotropic distribution assumed:
$$\begin{aligned}
\Lambda(\omega)&=\int_{\cot\theta}^{\infty}\int_{-\infty}^{\infty}P_2(x, y)(x\tan\theta-1)dydx \\&=\frac 1 c \int_{c}^{\infty}\int_{-\infty}^{\infty}P_2(x, y)(x-c)dydx \\&=\frac 1 c \int_{c}^{\infty}\int_{-\infty}^{\infty}xP_2(x, y)dydx- \int_{c}^{\infty}\int_{-\infty}^{\infty}P_2(x, y)dydx\\
c&=\cot \theta
\end{aligned}$$

We can reduce anisotropic distribution to isotropic with roughness $\alpha_y$ by stretching it along x-axis by $\alpha_y/\alpha_x$. Since $\Lambda$ only depends on $\omega$, we just need to transform it into stretched coordinates:
$$\omega'=[\frac{\alpha_x}{\alpha_y}\omega_x,\omega_y,\omega_z]$$
$${\tan\theta'}={\sqrt{(\frac{\alpha_x}{\alpha_y}\sin\phi)^2+\cos^2\phi}\tan\theta}$$
Then if we look at parameter a, that we derived above, we should express it in new coords:
$$a=\frac{1}{\alpha_y\tan\theta'}=\frac{1}{\alpha_y{\sqrt{(\frac{\alpha_x}{\alpha_y}\sin\phi)^2+\cos^2\phi}\tan\theta}}=\frac{1}{{\sqrt{(\alpha_x\sin\phi)^2+(\alpha_y\cos\phi)^2}\tan\theta}}=\frac{1}{\alpha\tan\theta}$$
In that case isotropic roughness $\alpha$ has the following value in terms of a roughness projected onto the outgoing direction $\omega_o$:
$$\alpha=\sqrt{(\alpha_x\sin\phi)^2+(\alpha_y\cos\phi)^2}=\frac{|[\alpha_x\omega_x, \alpha_y\omega_y]|}{\sin\theta}$$
#### Unaligned stretching
https://jcgt.org/published/0003/02/03/paper.pdf
The stretching operation does not need to be axis aligned. We can define a matrix $Q$ that would describe the rule for a norm computation:
$$|m|=\sqrt{m^TQm}$$
A standard Euclidean norm uses the unit matrix. Isotropic distributions then will be described with uniform scaling matrix. Anisotropic distributions can be described with non-uniform scaling. Unaligned stretching can be described with additional correlation parameters $r$. Thus we can describe $Q$ as follows:
$$Q=\left[
\matrix{
\alpha_x^2 & r\alpha_x\alpha_y \cr 
r\alpha_x\alpha_y & \alpha_y^2
}
\right]$$
#### Vertical Shearing and Non-Centered Distributions
https://jcgt.org/published/0003/02/03/paper.pdf
Since all the results are derived from slope distribution $P_2$, we can also introduce average slope $\widetilde{m}$ distinct from zero. That would allow us to accurately represent normal and bump maps in our equations, frequently used to add detail. The surface created by off-center the average slope is called meso-surface, being intermediate between macro and micro representation. 
Note that in the presence of meso-surface, the projected area of the micro-surface, as well as all other $\omega\cdot n$ factors, must be adjusted:
$$\intop\nolimits_{H^2}(\boldsymbol v\cdot\omega)D\mathrm{d}\omega=\frac{v\cdot \widetilde m}{n\cdot \widetilde m}$$
#### Generalized Trowbridge–Reitz model
https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
Lets consider a generic distribution of slopes, parametrized by power $\gamma$ and roughness $\alpha$:
$$f(r)=\frac{1}{\pi(1+\frac{r^2}{\gamma-1})^\gamma}$$
$$
P_2(\bar{m})=\frac{1}{\pi\alpha^2\left(1+\frac{\left|\bar{m}\right|^2}{\alpha^2(\gamma-1)}\right)^\gamma}
$$
 
From it we can derive NDF and masking functions:
https://research.nvidia.com/labs/rtr/student-beyond/publications/student-t-supplemental.pdf
https://chatgpt.com/g/g-p-68d44deb91288191b966e95e66e2b07c/c/68dd4308-a02c-8329-a027-532937397d31
$$D(m)=\frac{1}{\pi\alpha^2(m\cdot n)^4\left(1+\frac{\left|\bar{m}\right|^2}{\alpha^2(\gamma-1)}\right)^\gamma}$$
$$\begin{aligned}
&\Lambda(\omega)=\frac 1 a\int_{a}^{\infty}(x-a)P_2(x)dx\\
&P_2(x)=\frac {\sqrt{\gamma-1}\Gamma(\gamma-\frac 1 2)}{\alpha\sqrt\pi\Gamma(\gamma)}\left(1+\frac {x^2} {\alpha^2(\gamma-1)}\right)^{-(\gamma-\frac 1 2)}
\end{aligned}
$$
$$\begin{aligned}
&\Lambda(\omega)=-\frac {1}{2}+\frac {\Gamma(\gamma-\frac 3 2)}{2b\sqrt\pi\Gamma(\gamma-1)}(1+b^2)^{\frac 3 2-\gamma}+\frac {b\Gamma(\gamma-\frac 1 2)}{\alpha^2\sqrt\pi\Gamma(\gamma)}F_{2,1}(\frac 1 2,\gamma-\frac 1 2,\frac 3 2, -b^2)\\
&b=\frac {a}{\alpha\sqrt{\gamma-1}}
\end{aligned}$$
Works just fine, but hypergeometric term may be slow when implemented naively. For a more detailed derivation look at [[Generalized Trowbridge–Reitz lambda]]

Note that with $\gamma\to\infty$ it approaches normal distribution, which is the basis for Beckmann distribution. For $\gamma=2$ it results in regular Trowbridge–Reitz model.

chatgpt'd
Also note that for $\gamma=\frac \beta 2, \beta\in\mathbb{N}$ we can derive closed forms for the hypergeometric term.

Intuitively, $\gamma$ represents the proportion of steeper facets. Basically "what kind of roughness" the surface has. If a lot of facets are at extreme angles, practically all light will be trapped once it enters surface, because all its energy will dissipate while it bounces between facets.
The higher values usually represent more polished surfaces like glass or ceramics.
#### Generic slope distribution
https://diglib.eg.org/bitstream/handle/10.1111/cgf14590/v41i4pp105-116.pdf
We can describe any distribution as a linear combination of shifted/scaled Trowbridge–Reitz distributions. For a set of distributions $P_i$ with weights $w_i$ that sum to 1 and their corresponding NDFs $D_i$ and $\Lambda_i$, we can define combined distribution $P$, NDF $D$ and $\Lambda$ as follows:
$$P=\sum_{i}w_iP_i$$
$$D=\sum_{i}w_iD_i$$
$$\Lambda=\sum_{i}w_i\Lambda_i$$
While not physically motivated, it is useful to have for application of measured data.
### Microflake theory

https://rgl.epfl.ch/publications/Jakob2010Radiative
Consider an isolated particle illuminated by incident radiance $L(\omega)$. We can characterize the particle using three functions: 
1. $\sigma(\omega)$ is the area of the particle’s projection onto $\omega_{\bot}$. Probability of it hitting the particle.
2. $\alpha(\omega) \in [0,1]$ is the albedo of the particle when illuminated from direction $\omega$. Probability that light is scattered rather than absorbed, conditioned on having hit the particle
3. $p_p(\omega\to\omega')$ is the phase function exhibited by the particle when illuminated from direction $\omega$ and forms a probability density in the outgoing direction $\omega'$. Probability density for scattering to direction $\omega'$, conditioned on having scattered.

Note that the particle phase function $p_p$ is not necessary reciprocal. Rather the whole chain of events from hitting to scattering in a particular direction that is reciprocal:
$$f_p(\omega\to\omega')=\sigma(\omega)\alpha(\omega)p_p(\omega\to\omega')$$
$$f_p(\omega\to\omega')=f_p(\omega'\to\omega)$$
The properties of a volume containing many scattering particles depend on the characteristics of the particles—both their individual properties and the distribution of particle orientations that is present. We assume that particles are rotationally symmetric about some axis, so that their orientation can be entirely described by the direction of the axis. For the same reason we further assume that the particles are identical. A mixture of particle types can be accommodated easily by summing or integrating over the particles’ properties.

Under these two assumptions we can characterize the volume in the neighborhood of a particular point by two quantities: 
1. $\rho$ is the density of particles per unit volume. 
2. $D(m)$, a probability density on the sphere, gives the probability for a particle to be oriented in direction $m$.

With this we can define extinction and scattering coefficients $\sigma_t$, $\sigma_s$, and the phase function $p$:
$$
\sigma_t(\omega)=\rho\int_{S^2}\sigma(m, \omega)D(m)dm
$$
$$
\sigma_s(\omega)=\rho\int_{S^2}\alpha(m,\omega')\sigma(m, \omega')D(m)dm
$$
$$
p(\omega\to\omega')=\frac\rho{\sigma_s(\omega)}\int_{S^2}p_p(m, \omega'\to\omega)\alpha(m,\omega')\sigma(m, \omega')D(m)dm
$$

Now then lets proceed with defining these quantities for a particular type of particles. If we assume that particles are a planar, two-sided flakes, we can define $\sigma$ and $p_p$ as follows:
$$
\sigma(m, \omega)=a|\omega\cdot m|$$
$$p_p(\omega\to\omega')=f_m(m,\omega\to\omega') + f_m(-m, \omega\to\omega')$$
From which we can derive other relevant functions, except albedo. Free albedo allows us to adjust overall phase function to fit a particular model.

#### Generalized SGGX distribution
https://research.nvidia.com/sites/default/files/pubs/2015-08_The-SGGX-microflake/sggx.pdf
We can extend the GGX distribution and its generalization to be usable in microflake theory.
$$\sigma(\omega)=\sqrt {\omega^TS\omega}$$
$$D(m)=\frac 1 {\pi \sqrt S(m^TS^{-1}m)^2}$$
$$S=R[\alpha_x^2, \alpha_y^2, \alpha_z^2]R^T$$
$$S^{-1}=R[\frac 1{\alpha_x^2}, \frac 1{\alpha_y^2}, \frac 1{\alpha_z^2}]R^T$$
$$\bar{m}=R^Tm$$
We can try extend it just like the Generalized Trowbridge–Reitz: 
$$\begin{aligned}
D(m)&=\frac 1 {\pi \sqrt S(m^TS^{-1}m)^{\gamma}}\\
\end{aligned}$$
But note that this naive extension does not produce gaussian distribution in the limit $\gamma\to\infty$. To allow that we need to transform the denominator such that it is in the form $(1+\frac {x^2} a )^a$:
$$\begin{aligned}
D(m)&=\frac 1 {\pi \sqrt S(m^TS^{-1}m)^{2}}\\
&=\frac 1 {\pi \sqrt S (1 + m^TS^{-1}m - 1)^{2}}\\
&=\frac 1 {\pi \sqrt S (1 + \frac {m^TS^{-1}m - 1} {2-1})^{2}}\\
&=\frac 1 {\pi \sqrt S (1 + \frac {m^TS^{-1}m - 1} {\gamma-1})^{\gamma}}\\
\end{aligned}$$
Note that it also agrees with GTR when $S_{33}=1$.

With that last thing we need is to derive $\sigma(\omega)$:
$$
\begin{aligned}
\sigma(\omega)&=a\int_{S^2}|\omega\cdot m|D(m)dm\\
&=a\int_{S^2}\frac {|\omega\cdot m|} {\pi \sqrt S (1 + \frac {m^TS^{-1}m - 1} {\gamma-1})^{\gamma}}dm\\
\end{aligned}
$$

https://cseweb.ucsd.edu/~tzli/cse272/wi2023/lectures/11_microflake.pdf
https://onrendering.com/data/papers/ms16/ms16.pdf
https://cs.dartmouth.edu/~wjarosz/publications/seyb24from-small.pdf
### Multibounce microfacets
https://arxiv.org/pdf/2110.07145
https://sites.cs.ucsb.edu/~lingqi/publications/paper_mbbrdf_arxiv.pdf
https://d1qx31qr3h6wln.cloudfront.net/publications/Position_free_Smith.pdf
https://arxiv.org/pdf/2302.03408

We may consider full RTE at a surface boundary and its neighbouring volume. Given complex index of refraction we can compute absorption rate. If we interpreter surface as microflake volume, we may split flakes into front facing and backfacing and simulate light moving between under and above the surface together with absorption and refraction.

Microfacet theory assumes the facets form a single surface. If we relax this assumption such that facets can be positioned arbitrarily in micro-volume, then we basically get small plane-like dielectric flakes, which opens a possibility for modeling small-scale multi-bounces and subsurface scattering in thin surface volumes.

But its origins are in participating media rendering. It was developed to provide a similar framework to microfacets that allows deriving a valid phase function from some particle distribution and interaction that they have with light.

Fresnel equations and microfacets by themselves can't entirely approximate diffuse light, and I'm not even talking about approximating the rendering equation's output in its entirety. Diffuse light models absolute randomness in scattering distribution, making both $\omega_i$ and $\omega_o$ irrelevant.
When light bounces multiple times, it decorrelates $\omega_i$ and $\omega_o$ directions, making it more and more diffuse. And if the surface is extremely rough and reflective, a lot of bounces will happen, until the ray exits the surface, making it diffuse in nature.

We can evaluate the RTE over the microfacet's volume, bounded between upper and lower depth of the surface. The more bounces we simulate, the better the approximation becomes. Simulating it inside the volume via statistics is much cheaper than full raytracing per each facet, but still quite expensive considering the number of macro-surface intersections.

Following [this paper](https://eheitzresearch.wordpress.com/240-2/) we can simulate random walks in microfacet volumes, and evaluate the RTE at each step. We treat rays that exit the volume as contributing to overall BSDF, and others as part of the random walk.

visible normal distribution function $D_\omega$:
$$D_{\omega}(n)=\frac{(\omega\cdot n)D(n)}{\cos\theta(1+\Lambda(\omega))}$$
Generic phase function:
$$
p(\omega_i\to\omega, n)=\intop\nolimits_{H}f(m, \omega_i\to\omega)(\omega\cdot m)D_{\omega_i}(m)dm
$$

We still assume dielectric interactions for each microfacet, so the paper has only one relevant phase function for us:

$$p(\omega_i\to\omega, n)=\frac{RD_{\omega_i}(h_r)}{4|\omega_i\cdot h_r|} + (\omega\cdot n)\frac{\eta_o^2TD_{\omega_i}(h_t)}{(\eta_i(\omega_i\cdot h_t)+\eta_o(\omega_o\cdot h_t))^2}$$

There are some other models, like Oren-Nayar, Kulla–Conty or Burley, that can similarly restore energy from multiple bounces at the surface, but they are often heuristic or incomplete, which makes this model the most complete.
## Nano-geometry
A near-wavelength geometry detail, that is insignificant to BSDF, but contributes significant color-shifts due to wave-optics. Fine gratings of the surface distort BSDF per wavelength, which creates diffraction patterns.
### Diffraction

https://en.wikipedia.org/wiki/Diffraction

When a wave passes through a slit, it will create interference with itself, whenever measured at some distance from the slit. We can see it by applying [Huygens–Fresnel principle](https://en.wikipedia.org/wiki/Huygens%E2%80%93Fresnel_principle) - each point at the wavefront of planar wave can be treated as a spherical wave. When such wavefront passes through a slit, most of these spherical waves get reflected back, and only those that are between the corners get through. With that there is no compensation for the interference of two spherical waves at the sides of slit, which eventually reveals them at significant distances.

Once we considered how would a slit look, the same principles can be applied to arbitrary aperture. Further more, we can apply [Babinet’s Principle](https://en.wikipedia.org/wiki/Babinet%27s_principle), which basically means that the diffraction pattern for an object and an aperture of the same shape are the same. With that we can apply the diffraction results to any solid object.

[Kirchhoff's diffraction integral](https://en.wikipedia.org/wiki/Kirchhoff%27s_diffraction_formula) is the most general treatment of this phenomena. A bit simpler formular can be obtained by assuming far- and near-field interactions. In particular, far-field result is called Fraunhofer diffraction integral, and is exactly the Fourier transform of the wave function over the aperture. The near-field diffraction is describe by Fresnel diffraction integral and is useful for describing diffraction at nanoscale.

https://eugenedeon.com/
https://ssteinberg.xyz/2024fsdbsdf/steinberg2024_fsd_paper.pdf

Considering Babinet’s Principle, we can split the diffracted field into two parts, covering regions inside and outside aperture.

chatgpt'd
Happens due to wavelength-scale details in surface. For a thin layer, we get phase delay:
$$\delta(\lambda, d, \eta, \theta_t)=\frac{4\pi\ \eta\ d \cos \theta_t}{\lambda }$$
They scale polarized reflection and refraction as follows:

$$r'=\frac{r_1+r_2e^{2i\delta}}{1+r_1 r_2e^{2i\delta}}$$
where $r_1$ and $r_2$ are the entry and exit values for fresnel terms.

iridescence
https://hal.science/hal-01518344/file/paper-small%20%281%29.pdf

https://ssteinberg.xyz/2023rtplt/2023_rtplt_paper.pdf
https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-8-simulating-diffraction
microfacet diffraction
https://inria.hal.science/hal-01515948/file/paper.pdf
wave optics
https://cseweb.ucsd.edu/~ravir/waveoptics.pdf
https://backend.orbit.dtu.dk/ws/files/235458057/wptbsdf.pdf
diffraction shaders
https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/diff.pdf
# Layered Materials
https://www.pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Scattering_from_Layered_Materials
https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2014Comprehensive_2.pdf
https://hal.science/hal-01785457/document
https://arxiv.org/pdf/2110.07145
https://diglib.eg.org/items/fe183de1-de86-41b7-baf3-1efe8521c8c0
Until that point we only considered a uniform surface boundary. Having that foundation, we can extend it to multiple layers. 
Let us model each layer as a thin participating media of depth $d$, with upper boundary described by BSDF $f_l$, and a phase function $p_l$, each carrying the necessary parameters to be described with models for a single surface interface above. 
With this we can express radiance exiting a single layer as follows:
$$T=\left[R_l^{top}\ T_l^{bot}\atop T_l^{top}\ R_l^{bot}\right]$$
$$Q_l=\left[Q_l^{top}\atop Q_l^{bot}\right]$$
$$L_l=\left[L_l^{top}\atop L_l^{bot}\right]$$
$$L_l=TQ_l$$
Where
* $L_l^{top}$ and $L_l^{bot}$ are the radiance exiting the layer at the top and bottom
* $Q_l^{top}$ and $Q_l^{bot}$ are the radiance entering the layers, 
* $R^{top}$ and $R^{bot}$ as the reflected fraction of radiance from top and bottom.
* $T^{top}$ and $T^{bot}$ are the transmitted fractions between boundaries from top and bottom to the other side.
* $L_l$ is the vector of exiting radiance.
* $Q_l$ is the vector of entering radiance.
* $T$ is the Transfer matrix

We can compose two such layers using *adding equations*, which describe how two combine multiple transfer matrices into a single one:
$$R^{top}=R^{top}_1+T^{bot}_1(I-R^{top}_2R^{bot}_1)^{-1}R^{top}_2T^{top}_1$$
$$R^{bot}=R^{bot}_1+T^{top}_1(I-R^{bot}_1R^{top}_2)^{-1}R^{bot}_1T^{bot}_2$$
$$T^{top}=T^{top}_2(I-R^{bot}_1R^{top}_2)^{-1}T^{top}_1$$
$$T^{bot}=T^{bot}_1(I-R^{top}_2R^{bot}_1)^{-1}T^{bot}_2$$
Getting $T$ in general requires computing multiple bounces, which is expensive and often does not yield. We can get arbitrarily fine approximation by choosing finitely small $\Delta d$, where we can neglect multiple scattering, and apply *adding-doubling* algorithm to achieve desired layer depth.
The only issue with this approach is that it disregards the volumetric scattering by phase functions, and essentially replaces them by iteration of reflections and transmittance over the depth of the layer, which may have a significant impact for thick layers.

# Emission
https://www.taylorfrancis.com/books/edit/10.1201/9781003098690/phosphor-handbook-ru-shi-liu-xiaojun-wang?utm_source=chatgpt.com
https://www.cambridge.org/core/books/abs/classical-optics-and-its-applications/ewaldoseen-extinction-theorem/71F7EF2196FBAEF30C650A38E3C69FDF?utm_source=chatgpt.com
Generally emission is a distribution over the ray state.
We can split total emission $L_e$ by the nature of energy used for [emission](https://en.wikipedia.org/wiki/Luminescence):
* $L_h$ emission due to [temperature](https://en.wikipedia.org/wiki/Black-body_radiation) or its [change](https://en.wikipedia.org/wiki/Thermoluminescence).
* $L_a$ emission due to [earlier absorption of light](https://en.wikipedia.org/wiki/Photoluminescence), including [ionization](https://en.wikipedia.org/wiki/Radioluminescence).
* $L_m$ emission due to [mechanical action](https://en.wikipedia.org/wiki/Mechanoluminescence).
* $L_c$ emission due to [chemical reaction](https://en.wikipedia.org/wiki/Chemiluminescence).
* $L_s$ emission due to static electric field, including [electric current](https://en.wikipedia.org/wiki/Electroluminescence) and [Cherenkov radiation](https://en.wikipedia.org/wiki/Cherenkov_radiation).
These categories can be split further, based on particular scale (high/low energy interaction) or mechanism.
Thus we can split the total emission by its nature:
$$L_e = L_h+L_a+L_m+L_c+L_s$$
Almost all of the emission sources are defined as a distribution over ray state (wavelength, polarization, direction), and heavily depend on material properties and are time-varying.

###  Thermal emission

If we want to be even more physically accurate, we can define the $L_e$ and $Q$ functions based on thermal equilibrium or radiative equilibrium, which is "the total thermal radiation leaving an object is equal to the total thermal radiation entering it". Thus we can define them as follows:
$$L_h=B\left(\lambda, T\right)$$
Where $B(\lambda,T)$ is the blackbody radiance of the object, where $T$ is the temperature of the object. Since we assume equilibrium it is equal to the environment's thermal radiance, which we can assume anything. 
The $B(\lambda,T)$ itself is defined as:
$$B(\lambda,T)=\frac{2hc^2\lambda^{-5}}{e^{\frac{hc}{\lambda k_{B}T}}-1}=\frac{a\lambda^{-5}}{e^{b(\lambda T)^{-1}}-1}, a=2hc^2, b=\frac {hc}{k_B}$$

Of them only $L_h$ depends on temperature $T$ and its gradient.
### Chemical emission
### Mechanical emission
### Electric emission
### Photoluminescence

chatgpt'd

$$\int_0^{t} \int_0^{\infty} f_e(\lambda, \lambda_{in}, t, P, \omega) d\lambda_{in} dt$$

Of them only $L_a$ are due to almost instantaneous interaction.
Of them only $L_a$ is dependent on incoming light, and thus need to preserve energy.

We may also add physically accurate light emission for volumes and surfaces due to absorption of the incoming light. The definition of radiance due to photoluminescence:
$$L_p(x,\omega_{out},\lambda_{out})=\intop\nolimits_{S^2}\intop_{0}^{\infty}\eta_{PL}(x,\omega_{out},\lambda_{in}\to\lambda_{out})\sigma_{PL}(x,\omega_{out},\lambda_{in})L(x,\omega,\lambda_{in})d\lambda_{in}d\omega$$
Where:
* $\eta_{PL}(x,\lambda_{in}\to\lambda_{out})$ is the conversion rate at point $x$ from wavelength $\lambda_{in}$ to $\lambda_{out}$.
* $\sigma_{PL}(x,\lambda_{in})$ is the absorption rate at point $x$ for a wavelength $\lambda_{in}$.

$\eta_{PL}$ also has normalization constraint:
$$\intop_0^{\infty}\frac{\lambda_{in}}{\lambda_{out}}\eta_{PL}\left(\lambda_{in}\to\lambda_{out}\right)d\lambda_{out}\le1$$
https://inria.hal.science/hal-01818826/document
https://www.reddit.com/r/GraphicsProgramming/s/nAGtEgcWPm
### Total emission

We write down total re-emission for volumes and objects as follows:
$$Q_a=(1-\eta_v)B_{\lambda}(T) + \eta_v\intop\nolimits_{S^2}Ld\omega$$
In the same manner is defined a surface re-emission term:
$$Q_{surf}=(1-\eta_s)B_{\lambda}(T)+\eta_s\intop\nolimits_{S^2}(n\cdot\omega)f_eLd\omega$$



# Subsurface scattering
A particular class of lighting effect permits reformulation, that allows for cheaper simulations.

subsurface scattering
https://users.cg.tuwien.ac.at/zsolnai/wp/wp-content/uploads/2014/12/ssss.pdf
https://eugenedeon.com/pdfs/zv2020.pdf
# Camera
Overall
Integrate over "sensor" area
Sum over lenses
Integrate over aperture
Integrate over exposure time
Integrate over ray state (importance sample by photosensitivity)
Apply bloom (near-field diffraction pattern)
Convert collected intensities for each wavelength to rgb
### Spectrum to RGB

https://larswander.com/writing/spectral-ray-tracing/
Our definitions are wavelength-dependent, but our eyes have a different response for each of the wavelengths. Thus before displaying we need to compute the response for R, G, and B of our eyes.
Given some intensity distribution $L(\lambda)$, we need to compute spectral power distribution $S(\lambda)$ with the following formula:
chatgpt'd
$$S(\lambda)=\int_{A}\int_{\Omega} L(x,\omega,\lambda)(\omega\cdot n)d\omega dx$$
Where $A$ is the area of the pixel, $\Omega$ is aperture area, and $n$ is the direction of view.

Then we can compute RGB response with $\overline{r}\left(\lambda\right)$, $\overline{b}\left(\lambda\right)$, $\overline{b}\left(\lambda\right)$ functions that correspond to sensor response of each color sensor:
$$R=\int_0^{\infty}S(\lambda)\overline{r}\left(\lambda\right)d\lambda$$
$$G=\int_0^{\infty}S(\lambda)\overline{g}\left(\lambda\right)d\lambda$$
$$B=\int_0^{\infty}S(\lambda)\overline{b}\left(\lambda\right)d\lambda$$
The $\overline{r}\left(\lambda\right)$, $\overline{b}\left(\lambda\right)$, $\overline{b}\left(\lambda\right)$ functions itself are normalized to have equal area:
$$\int_0^{\infty}\overline{r}\left(\lambda\right)d\lambda=\int_0^{\infty}\overline{g}\left(\lambda\right)d\lambda=\int_0^{\infty}\overline{b}\left(\lambda\right)d\lambda$$
These function can be approximated through XYZ color space as a mixture of two-sided Gaussians $g$:
$$\tau(x,\mu,\tau_1,\tau_2)=\begin{cases}
    \tau_1 & \text{if } x<\mu \\
    \tau_2 & \text{otherwise}
\end{cases}$$
$$g(x,\mu,\tau_1,\tau_2)=e^{-\frac{\tau^2(x-\mu)^2}{2}}$$
$$\begin{aligned}\overline{x}\left(\lambda\right)&=1.056g(\lambda,599.8,0.0264,0.0323)\\
&+0.362g(\lambda,422,0.0624,0.0374)\\
&-0.065g(\lambda,501.1,0.049,0.0382)\end{aligned}$$
$$\overline{y}\left(\lambda\right)=0.821g(\lambda,568.8,0.0213,0.0247)+0.286g(\lambda,530.9,0.0613,0.0322)$$
$$\overline{z}\left(\lambda\right)=1.217g(\lambda,437,0.0845,0.0278)+0.681g(\lambda,459,0.0385,0.0725)$$
$$\left[\array{r\cr g\cr b}\right]=\left[\matrix{0.49 & 0.31 & 0.2\cr 0.17697 & 0.8124 & 0.01063\cr 0 & 0 & 0.99}\right]^{-1}\left[\array{x\cr y\cr z}\right]$$
![[chrome_VOI1ndezrZ_1758182559.png]]
https://youtu.be/wA1KVZ1eOuA?si=vBoEcSDCgD2pVAGd
https://en.wikipedia.org/wiki/CIE_1931_color_space
### RGB to spectrum
https://graphics.geometrian.com/research/spectral-primaries.html
We also need inverse transformations to transform an rgb material color into spectral distribution.
We can implement it as a function that measures spectral power distribution for a given rgb value, evaluated at given wavelength.

Note, that there is no unique spectrum corresponding to each rgb value. To resolve this issue we can additionally constrain it to be varying as little as possible. That is motivated by observation that many materials, especially natural, have smooth spectrum.

To construct such distribution from the rgb value, we should consider the effect of illuminating surface with that color with white light. The "white light" is standardized to be described by a $D_{65}$ distribution, the [standard daylight illuminant](https://en.wikipedia.org/wiki/Standard_illuminant#Illuminant_series_D). Thus, by definition, the white color must correspond to $D_{65}$'s distribution.

Now, given a distribution $S$ for some rgb value, the observed XYZ color for that rgb color under white light is computed as the sum over all wavelengths:
$$
\left[\array{X\cr Y\cr Z}\right]=\sum_{\lambda}
\left[\array{\bar x(\lambda)\cr \bar y(\lambda)\cr \bar z(\lambda)}\right]D_{65}(\lambda)S(\lambda)
$$
To then convert it to the linear rgb space we use the transformation formula:
$$
\left[\array{r\cr g\cr b}\right]=M^{-1}\left(\frac 1 {Y_{D_{65}}}
\left[\array{X\cr Y\cr Z}\right]\right)
$$

  
### Antialiasing
https://www.iryoku.com/aacourse/
https://www.reddit.com/r/GraphicsProgramming/s/f26q2kQi56
### Depth of field

Depth of Field (DoF) is lens artifact, caused by misplaced imaging plane relative to focus point.
Due to that we see a blurred image.
https://blog.demofox.org/2018/07/04/pathtraced-depth-of-field-bokeh/
### Motion blur
In addition to sensor area, aperture and wavelength integrals required for wavelength-to-rgb conversion for a camera, we also need to integrate over exposure time to get motion blur effects, and... well... total exposure.
https://raytracing.github.io/books/RayTracingTheNextWeek.html#motionblur
### Lens flare
https://resources.mpi-inf.mpg.de/lensflareRendering/pdf/flare.pdf
https://www.youtube.com/watch?v=IbJfZS0o2kg&ab_channel=GameDevelopersConference
### Bloom
https://www.youtube.com/watch?v=QWqb5Gewbx8&ab_channel=AngeTheGreat
### Tonemapping
https://bruop.github.io/tonemapping/
### Projections

panini projection
http://tksharpless.net/vedutismo/Pannini/
https://www.scribd.com/document/284463081/The-General-Panini-Projection
https://www.researchgate.net/publication/220795340_Pannini_A_New_Projection_for_RenderingWide_Angle_Perspective_Images

[(PDF) Essential Ray Generation Shaders](https://www.researchgate.net/publication/354065227_Essential_Ray_Generation_Shaders)

# Photometry
Analytic models are very useful, but often fail to fully capture the material's behavior. For that purpose we must include data-driven models that allow applying real-world measurements to the rendering and achieving faithful results.

god damn its so hard
BTFs
https://www.cemyuksel.com/research/stitchmeshes/
# Artistic parametrization
https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
Reformulation with a different set of parameters, that is much more artist-friendly.

