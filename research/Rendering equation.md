chatgpt'd
The standard Rendering Equation has the following form:

$$L\left(\omega_{o}\right)=\intop\nolimits_{H^2}\mathrm{f\left(\omega_{o},\omega_{i}\right)L\left(\omega_{i}\right)\cos\left(\omega_{i}\right)d}\omega_{i}$$
where $\omega_{i}$ ranges over the sphere of directions around some point $x$

It is a simplification of a more generic Radiative Transfer Equation (differential form):
$$\omega\cdot\nabla L\left(\omega\right)+\sigma_{t}\left(\omega\right)L\left(\omega\right)=V(\omega)$$
$$V(\omega)=Q_{e}(\omega)+\sigma_a(\omega)Q_a\left(\omega\right) +\sigma_{s}
(\omega)\intop\nolimits_{S^2} p\left(\omega_{i}\to\omega\right)L\left(\omega_{i}\right)\mathrm{d}\omega_{i}$$
Where
* $\boldsymbol x$ is the ray origin
* $\omega$ is the ray direction
* $L$ the incoming radiance from the direction $\omega$
* $Q_e$ is the emission of light in the direction $\omega$
* $Q_a$ is the re-emission of absorbed light in the direction $\omega$
* $p$ is the phase function - probability density for scattering from the direction $\omega_{i}$ in the direction $\omega$
* $\sigma_{a}$, $\sigma_{s}$, $\sigma_{t}=\sigma_{a}+\sigma_{s}$ are the absorption, scattering, and extinction coefficients respectively for a given point $\boldsymbol x$ along the direction $\omega$.
* $V$ is a source term for the light coming from the direction $\omega$.

The $p$ must obey normalization constraint:
$$\intop\nolimits_{S^2}p\left(\omega_{i}\to\omega\right)\mathrm{d}\omega_{}=1$$
Radiative Transfer Equation in integral form:
$$L\left(\boldsymbol x,\omega\right)=T(\boldsymbol x \to \boldsymbol x_{surf})L_{surf}(\boldsymbol x_{surf}, \omega)+\int_{0}^{t_{surf}}T\left(\boldsymbol x\to \boldsymbol x_{t}\right) V(\boldsymbol x,\omega)\mathrm{d}t$$
Where $\boldsymbol x_{surf}$ and $\boldsymbol x_{t}$ are shorthands for $\boldsymbol x_{t}=\boldsymbol x+t\omega$ and $\boldsymbol x_{surf}=\boldsymbol x+t_{surf}\omega$, and $T\left(x\to x_{t}\right)$ is the following:
$$T\left(x\to x_{t}\right)=e^{-\intop\nolimits_0^{t}\sigma_{t}\left(x_{u}, \omega\right)\mathrm{d}u}$$
The $L_{surf}$ term is the one expressed in a standard rendering equation. But it is usually simplified to only consider reflected light. The exact form is as follows:
$$L_{surf}\left(\omega\right)=L_{e}(\omega) + (1-\sigma_{r}
(\omega))Q_{surf}(\omega)+\sigma_{r}
(\omega)\intop\nolimits_{S^2}f(\omega_{i} \to \omega)(\boldsymbol n\cdot\omega_{i})L(\omega_{i})\mathrm{d}\omega_{i}$$
Where
* $\boldsymbol x$ is the ray origin
* $\omega$ is the ray direction
* $L$ the incoming radiance from the direction $\omega$
* $L_e$ the emitted radiance in the direction $\omega$
* $\boldsymbol n$ is the normal of the surface.
* $f$ is the bidirectional scattering distribution function - probability density for scattering from the direction $\omega_{i}$ in the direction $\omega$.
* $Q_{surf}$ is the re-emitted absorbed light in the direction $\omega$.
* $\sigma_{r}$  is the absorption factor in the direction $\omega$.

The $f(\boldsymbol x,\omega_{i}\to\omega)$ must also obey normalization constraint:
$$\intop\nolimits_{S^2}f\left(\omega_{i}\to\omega\right)(\boldsymbol n\cdot\omega_{i})\mathrm{d}\omega_{}=1$$
Since energy must be conserved, the following must hold:
$$\sigma_{r}\le1$$

Notice that the surface point is explicit and integration domain is the whole sphere of directions.
For simplicity sake, we could omit the parameters for each of the functions to simplify equations visually. If unclear assume we refer to the equations above.

All the equations above are also parametrized by time, ray origin and wavelength. Only parameters relevant for the un-ambiguation of the equation are written, others are implicitly passed through.
$t_{surf}$ is the boundary condition for the surface hit of a ray and entirely depends on the actual scene. 
# Reciprocity
chatgpt'd
It is a common assumption that it does not matter in which direction we measure light - from camera to light or the other way.
There are only two functions that depend both on incoming and outgoing light directions - $p(\omega_{i}\to\omega)$ and $f(\omega\to\omega_{i})$. Thus we impose additional constraints on these functions:
$$\sigma_s(\omega_i)p\left(\omega_{i}\to\omega\right)=\sigma_s(\omega)p\left(\omega\to\omega_{i}\right)$$
$$\sigma_r(\omega_i)f\left(\omega_{i}\to\omega\right)=\sigma_r(\omega)f\left(\omega\to\omega_{i}\right)$$
# BSDF

We define BSDF as the function that describes radiance transfer across a surface boundary. It describes how much light is reflected or exits from inside the object between an incoming and outgoing directions.

https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
### Dielectrics and conductors
chatgpt'd
We use the research dielectrics and conductors as the base for any other materials. So we assume that any material is a mixture of such materials and local geometric properties of the surface.

They are described by an index of refraction (IoR), which is a complex number $\eta(x, \omega, \lambda)=n+ik$. If $k$ is zero, then it is considered a dielectric, otherwise a conductor. $k$ represents absorption rate of the material.
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
$$f_{s}\left(x,\omega_{i}\to\omega_{o},\lambda\right)=R\left(\lambda,\omega_{i}, n\right)\delta(\omega_o-reflect\left(\omega_{i},n\right))+T_{BSDF}\left(\lambda,\omega_{i}, n\right)\delta(\omega_o-refract(\omega_{i}, \eta(x, \omega_{i}, \lambda), n))$$
### Absorption
chatgpt'd
The absorption rate can be expressed in terms of $k$:
$$\sigma_a=\frac{4\pi k}{\lambda}$$
It is not unphysical if we also include explicit surface reemission, as long as we scale it down proportional to absorption coefficient. That way, whatever reemission happens due to volumetric absorption, it is not double counted.

# Phase function
chatgpt'd
Phase function is the basis for volumetric rendering, since it describes generically how the light scatters in a volume.
For phase function there isn't any good "universal" model. The most precise formulation for the phase function is given by Mie theory, which requires high computational resources.
We describe participating media by particle radius $a$ and complex refractive index $n$.
If we assume particles to be much smaller than wavelength, we get Rayleigh scattering:
$$\sigma_s(\lambda)=\frac{8\pi^3a^6|n^2-1|^2}{3\lambda^4|n^2+2|^2}$$
$$p(\omega_i\to\omega)=\frac{3}{16\pi}(1+(\omega_i\cdot\omega)^2)$$

# Microgeometry
While general RTE fully describes the radiance, it is unfeasible to render the micro details of objects. Besides unpracticality, such fine details are also imperceivable, since all of the detail is in a single pixel area, which is averaged in the final render. Thus it is a great place for statistical methods that describe microgeometry properties statistically.
In that case for every sample point $x$ we evaluate a statistical model of properties in an infinitesimal volume at that point, which simulates averaged result of fine details in both participating media and surface. 
There were developed two theories that give tools to handle both cases.
Together with broad scattering simulated in raytracing directly, it gives a complete description of radiance in the scene.
## Microfacet theory

The fresnel terms define reflection and transmission for ideal smooth surfaces. But that misses the imperfection of real world. Lets define a map from surface coords to world coords $H: R^2\to R^3$. If we assume that for a local patch $A$ the function $H$ is a heightmap, we can apply microfacet theory.

We define geometric surface properties as a combination of two functions $D(x,h, \lambda, n, t)$ and $G(x,\omega_i,\omega_o,\lambda, n, t)$, the Normal Distribution Function, and masking-shadowing function. The parameters $n$ and $t$ are the geometric normal and tangent vectors, $h$ is a normal that would reflect/refract the $\omega_i$ into $\omega_o$, also called a half-vector.  Together these allow modelling a single scattering event at the surface.

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

There are some constraints on what $D$ and $G$ can be. The constraints on $D$:
1. $D$ is not negative: $D\ge0$
2. $D$  produce the same (signed) projected area as the macrosurface for any direction $v$: $\intop\nolimits_{H^2}(\boldsymbol v\cdot\omega)D\mathrm{d}\omega=(v\cdot n)$
3. $D$ total area must be at least as large as the macrosurface: $\intop\nolimits_{H^2}D\mathrm{d}\omega\ge1$
4. Is zero outside hemisphere and at the boundary
5. Sometimes it is required that $D(h)=O(1/\cos^3\theta_h)$ or slower.

The constraints on $G$:
1. $G(\omega_i, \omega_o)=G(\omega_o, \omega_i)$
2. $G\in\left\lbrack0,1\right\rbrack$
3. $G$ is smooth
4. As $n\cdot\omega\to0$, $G\to0$
5. Backfaces of microfacets are not visible frontside of macrosurfaces: $G=0$ if $(\omega_i\cdot m)(\omega_i\cdot n)\le0$ or $(\omega_o\cdot m)(\omega_o\cdot n)\le0$

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

Note that the G term is computationally complex and also misses the secondary bounces. Basically we would want to evaluate the complete RTE for each microfacet, instead of two generic functions.
## Microflake theory
particle density $\sigma_p$
albedo $\alpha$
NDF $D$
$$\sigma_a(\omega)=\sigma_p(1-\alpha)\intop\nolimits_{S^2}(m\cdot \omega)D(m)dm$$
$$\sigma_s(\omega)=\sigma_p\alpha\intop\nolimits_{S^2}(m\cdot \omega)D(m)dm$$
$$\rho(\omega_i\to\omega)=\frac{\alpha}{\sigma_s(\omega_i)}D(\frac{\omega_i+\omega}{|\omega_i+\omega|})D(-\frac{\omega_i+\omega}{|\omega_i+\omega|})$$
https://cseweb.ucsd.edu/~tzli/cse272/wi2023/lectures/11_microflake.pdf
## Multibounce microfacets

Fresnel equations and microfacets by themselves can't entirely approximate diffuse light, and I'm not even talking about approximating the rendering equation's output in its entirety. Diffuse light models absolute randomness in scattering distribution, making both $\omega_i$ and $\omega_o$ irrelevant.
When light bounces multiple times, it decorrelates $\omega_i$ and $\omega_o$ directions, making it more and more diffuse. And if the surface is extremely rough and reflective, a lot of bounces will happen, until the ray exits the surface, making it diffuse in nature.

We can evaluate the RTE over the microfacet's volume, bounded between upper and lower depth of the surface. The more bounces we simulate, the better the approximation becomes. Simulating it inside the volume via statistics is much cheaper than full raytracing per each facet, but still quite expensive considering the number of macrosurface intersections.

Following [this paper](https://eheitzresearch.wordpress.com/240-2/) we can simulate random walks in microfacet volumes, and evaluate the RTE at each step. We treat rays that exit the volume as contributing to overall BSDF, and others as part of the random walk.

visible normal distribution function $D_\omega$:
$$D_{\omega}(n)=\frac{(\omega\cdot n)D(n)}{\cos\theta(1+\Lambda(\omega))}$$
Generic phase function:
$$
p(\omega_i\to\omega, n)=\intop\nolimits_{H}f(m, \omega_i\to\omega)(\omega\cdot m)D_{\omega_i}(m)dm
$$

We still assume dielectric interactions for each microfacet, so the paper has only one relevant phase function for us:

$$p(\omega_i\to\omega, n)=\frac{RD_{\omega_i}(h_r)}{4|\omega_i\cdot h_r|} + (\omega\cdot n)\frac{\eta_o^2TD_{\omega_i}(h_t)}{(\eta_i(\omega_i\cdot h_t)+\eta_o(\omega_o\cdot h_t))^2}$$

### Relevant microfacet distributions and functions

Smith model and other stuff.

# Anisotropic effects

Materials may behave differently at some angles, when geometry is directionally correlated, elongating the specular highlight, for example when looking at machined surfaces.

# Layered Materials

We can describe surface of objects as a layered materials with depth $d$. Each material is modeled with its own BSDF, which are then composed into a single function.

# Diffraction
chatgpt'd
Happens due to wavelength-scale details in surface. For a thin layer, we get phase delay:
$$\delta(\lambda, d, \eta, \theta_t)=\frac{4\pi\ \eta\ d \cos \theta_t}{\lambda }$$
They scale polarized reflection and refraction as follows:

$$r'=\frac{r_1+r_2e^{2i\delta}}{1+r_1 r_2e^{2i\delta}}$$
where $r_1$ and $r_2$ are the entry and exit values for fresnel terms.

# Emission
chatgpt'd
If we want to be even more physically accurate, we can define the $L_e$ and $Q$ functions based on thermal equilibrium or radiative equilibrium, which is "the total thermal radiation leaving an object is equal to the total thermal radiation entering it". Thus we can define them as follows:
$$Q_a=B_{\lambda}\left(T\right)$$
$$Q_{surf}=B_{\lambda}\left(T\right)$$
Where $B_{\lambda}(T)$ is the blackbody radiance of the object, where $T$ is the temperature of the object. Since we assume equilibrium it is equal to the environment's thermal radiance, which we can assume anything. 
The $B_{\lambda}(T)$ itself is defined as:
$$B_{\lambda}(T)=\frac{2hc^2}{\lambda^5}\cdot\frac{1}{e^{\frac{hc}{\lambda k_{B}T}}-1}$$
### Photoluminescence

We may also add physically accurate light emission for volumes and surfaces due to absorption of the incoming light. The definition of radiance due to photoluminescence:
$$Q_{PL}(x,\omega_{out},\lambda_{out})=\intop\nolimits_{S^2}\intop_{0}^{\infty}\eta_{PL}(x,\omega_{out},\lambda_{in}\to\lambda_{out})\sigma_{PL}(x,\omega_{out},\lambda_{in})L(x,\omega,\lambda_{in})d\lambda_{in}d\omega$$
Where:
* $\eta_{PL}(x,\lambda_{in}\to\lambda_{out})$ is the conversion rate at point $x$ from wavelength $\lambda_{in}$ to $\lambda_{out}$.
* $\sigma_{PL}(x,\lambda_{in})$ is the absorption rate at point $x$ for a wavelength $\lambda_{in}$.

$\eta_{PL}$ also has normalization constraint:
$$\intop_0^{\infty}\frac{\lambda_{in}}{\lambda_{out}}\eta_{PL}\left(\lambda_{in}\to\lambda_{out}\right)d\lambda_{out}\le1$$
Generally it depends on the wavelength of the incoming and outgoing light, as in the definition above, but for simplification we could consider "single wavelength" definition:
$$Q_a(x,\omega)=\eta(x,\omega)\intop\nolimits_{S^2}L(x,\omega_{in})d\omega_{in}$$
In the same manner is defined a surface emission term:
$$Q_{surf}(x,\omega)=\eta(x,\omega)\intop\nolimits_{S^2}f_e(x,\omega,\omega_{in})L(x,\omega_{in})(n\cdot\omega_{in})d\omega$$
The $f_e$ term is responsible for "accepting" the radiance from a particular direction, that escapes outwards.
Basically we can define it as follows:
$$
f_e = \begin{cases}
    R(x,\omega) & \text{if } \omega\cdot n>0 \\
    T(x,\omega) & \text{otherwise}
\end{cases}
$$
### Total emission

We write down total re-emission for volumes and objects as follows:
$$Q_a=(1-\eta_v)B_{\lambda}(T) + \eta_v\intop\nolimits_{S^2}Ld\omega$$
In the same manner is defined a surface re-emission term:
$$Q_{surf}=(1-\eta_s)B_{\lambda}(T)+\eta_s\intop\nolimits_{S^2}(n\cdot\omega)f_eLd\omega$$



# Camera image rendering
Overall
Integrate over "sensor" area
Integrate over aperture
Integrate over exposure time
Integrate over wavelengths (importance sample by photosensitivity)
Convert collected intensities for each wavelength to rgb
### Eye photosensitivity

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

We also need inverse transformations to transform a material color into spectral reflectance distribution.
### Motion blur
In addition to sensor area, aperture and wavelength integrals required for wavelength-to-rgb conversion for a camera, we also need to integrate over exposure time to get motion blur effects, and... well... total exposure.
### Lens flare
https://resources.mpi-inf.mpg.de/lensflareRendering/pdf/flare.pdf
https://www.youtube.com/watch?v=IbJfZS0o2kg&ab_channel=GameDevelopersConference
### Bloom
https://www.youtube.com/watch?v=QWqb5Gewbx8&ab_channel=AngeTheGreat