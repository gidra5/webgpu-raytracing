The standard Rendering Equation has the following form:

$$L\left(\omega_{o}\right)=\intop\nolimits_{H^2}\mathrm{f\left(\omega_{o},\omega_{i}\right)L\left(\omega_{i}\right)\cos\left(\omega_{i}\right)d}\omega_{i}$$
where $\omega_{i}$ ranges over the sphere of directions around some point $x$

It is a simplification of a more generic Radiative Transfer Equation (differential form):
$$\omega\cdot\nabla L\left(\boldsymbol x,\omega\right)+\sigma_{t}\left(\boldsymbol x, \omega\right)L\left(\boldsymbol x,\omega\right)=V(\boldsymbol x,\omega)$$
$$V(\boldsymbol x,\omega)=Q_{e}(\boldsymbol x,\omega)+\sigma_a(\boldsymbol x,\omega)Q_a\left(\boldsymbol x,\omega\right) +\sigma_{s}
(\boldsymbol x, \omega)\intop\nolimits_{S^2} p\left(\boldsymbol x,\omega_{i}\to\omega\right)L\left(\boldsymbol x,\omega_{i}\right)\mathrm{d}\omega_{i}$$
Where
* $\boldsymbol x$ is the ray origin
* $\omega$ is the ray direction
* $L(\boldsymbol x,\omega)$ the incoming radiance to the point $\boldsymbol x$ from the direction $\omega$
* $Q_e(\boldsymbol x,\omega)$ is the emission of light at the point $\boldsymbol x$ in the direction $\omega$. A free parameter
* $Q_a(\boldsymbol x,\omega)$ is the re-emission of absorbed light at the point $\boldsymbol x$ in the direction $\omega$. A free parameter
* $p(\boldsymbol x,\omega_{i}\to\omega)$ is the phase function - probability density for scattering at the point $\boldsymbol x$ from the direction $\omega_{i}$ in the direction $\omega$. A free parameter
* $\sigma_{a}(\boldsymbol x,\omega)$, $\sigma_{s}(\boldsymbol x,\omega)$, $\sigma_{t}(\boldsymbol x,\omega)=\sigma_{a}(\boldsymbol x,\omega)+\sigma_{s}(\boldsymbol x,\omega)$ are the absorption, scattering, and extinction coefficients respectively for a given point $\boldsymbol x$ along the direction $\omega$. All must be non-negative values. A free parameter
* $V(\boldsymbol x,\omega)$ is a source term for the light coming from the point $\boldsymbol x$ in the direction $\omega$.

The $p(\boldsymbol x,\omega_{i}\to\omega)$ must obey normalization constraint:
$$\intop\nolimits_{S^2}p\left(x,\omega_{i}\to\omega\right)\mathrm{d}\omega_{}=1$$
Radiative Transfer Equation in integral form:
$$L\left(\boldsymbol x,\omega\right)=T(\boldsymbol x \to \boldsymbol x_{surf})L_{surf}(\boldsymbol x_{surf}, \omega)+\int_{0}^{t_{surf}}T\left(\boldsymbol x\to \boldsymbol x_{t}\right) V(\boldsymbol x,\omega)\mathrm{d}t$$
Where $\boldsymbol x_{surf}$ and $\boldsymbol x_{t}$ are shorthands for $\boldsymbol x_{t}=\boldsymbol x+t\omega$ and $\boldsymbol x_{surf}=\boldsymbol x+t_{surf}\omega$, and $T\left(x\to x_{t}\right)$ is the following:
$$T\left(x\to x_{t}\right)=e^{-\intop\nolimits_0^{t}\sigma_{t}\left(x_{u}, \omega\right)\mathrm{d}u}$$

Free parameters are also parametrized by time (so could change over time). $t_{surf}$ is also a free term, since it entirely depends on the actual scene. Thus it is time-dependent as well.

The $L_{surf}$ term is the one expressed in a standard rendering equation. But it is usually simplified to only consider reflected light. The exact form is as follows:
$$L_{surf}\left(\boldsymbol x,\omega\right)=L_{e}(\boldsymbol x,\omega) + (1-\sigma_{r}
(\boldsymbol x, \omega))Q_{surf}(\boldsymbol x,\omega)+\sigma_{r}
(\boldsymbol x, \omega)\intop\nolimits_{S^2}f(\boldsymbol x,\omega_{i} \to \omega)L(\boldsymbol x,\omega_{i})(\boldsymbol n(\boldsymbol x)\cdot\omega_{i})\mathrm{d}\omega_{i}$$
Where
* $\boldsymbol x$ is the ray origin
* $\omega$ is the ray direction
* $L(\boldsymbol x,\omega)$ the incoming radiance to the point $\boldsymbol x$ from the direction $\omega$
* $L_e(\boldsymbol x,\omega)$ the emitted radiance to the point $\boldsymbol x$ in the direction $\omega$
* $\boldsymbol n(\boldsymbol x)$ is the normal of the surface at the point $\boldsymbol x$. A free parameter
* $f(\boldsymbol x,\omega_{i}\to\omega)$ is the scattering function - probability density for scattering at the point $\boldsymbol x$ from the direction $\omega_{i}$ in the direction $\omega$. A free parameter
* $Q_{surf}(\boldsymbol x,\omega)$ is the re-emitted absorbed light at the point $\boldsymbol x$ in the direction $\omega$. A free parameter
* $\sigma_{r}(\boldsymbol x, \omega)$  is the absorption factor at the surface point $\boldsymbol x$ in the direction $\omega$. A free parameter

The $f(\boldsymbol x,\omega_{i}\to\omega)$ must also obey normalization constraint:
$$\intop\nolimits_{S^2}f\left(x,\omega_{i}\to\omega\right)(\boldsymbol n(\boldsymbol x)\cdot\omega_{i})\mathrm{d}\omega_{}=1$$
Since energy must be conserved, the following must hold:
$$\sigma_{r}(x_{t}, \omega)\le1$$

Notice that the surface point is explicit and integration domain is the whole sphere of directions.
For simplicity sake, we could omit the parameters for each of the functions to simplify equations visually. If unclear assume we refer to the equations above.

The $f$ function is the BSDF we usually use in computation.
# Reciprocity

It is a common assumption that it does not matter in which direction we measure light - from camera to light or the other way.
There are only two functions that depend both on incoming and outgoing light directions - $p(\boldsymbol x,\omega_{i}\to\omega)$ and $f(\boldsymbol x,\omega\to\omega_{i})$. Thus we impose additional constraints on these functions:
$$p\left(\boldsymbol x,\omega_{i}\to\omega\right)=p\left(\boldsymbol x,\omega\to\omega_{i}\right)$$
$$f\left(\boldsymbol x,\omega_{i}\to\omega\right)=f\left(\boldsymbol x,\omega\to\omega_{i}\right)$$
# BSDF

We define BSDF as the function that describes radiance transfer across a surface boundary. It describes how much light is reflected or exits from inside the object between an incoming and outgoing directions.
### Dielectrics and conductors

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

The absorption rate can be expressed in terms of $k$:
$$\sigma_a=\frac{4\pi k}{\lambda}$$
It is not unphysical if we also include explicit surface reemission, as long as we scale it down proportional to absorption coefficient. That way, whatever reemission happens due to volumetric absorption, it is not double counted.
# Microfacet theory

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

# Multibounce microfacets

Fresnel equations and microfacets by themselves can't entirely approximate diffuse light, and I'm not even talking about approximating the rendering equation's output in its entirety. Diffuse light models absolute randomness in scattering distribution, making both $\omega_i$ and $\omega_o$ irrelevant.
When light bounces multiple times, it decorrelates $\omega_i$ and $\omega_o$ directions, making it more and more diffuse. And if the surface is extremely rough and reflective, a lot of bounces will happen, until the ray exits the surface, making it diffuse in nature.

We can evaluate the RTE over the microfacet's volume, bounded between upper and lower depth of the surface. The more bounces we simulate, the better the approximation becomes. Simulating it inside the volume via statistics is much cheaper than full raytracing per each facet, but still quite expensive considering the number of macrosurface intersections.

Following (this paper)[https://eheitzresearch.wordpress.com/240-2/] we can simulate random walks in microfacet volumes, and evaluate the RTE at each step. We treat rays that exit the volume as contributing to overall BSDF, and others as part of the random walk.

We still assume dielectric interactions for each microfacet, so the paper has only one relevant phase function for us.

### Relevant microfacet distributions and functions

Smith model and other stuff.

# Anisotropic effects

Materials may behave differently at some angles, when geometry is directionally correlated, elongating the specular highlight, for example when looking at machined surfaces.

# Layered Materials

We can describe surface of objects as a layered materials with depth $d$. Each material is modeled with its own BSDF, which are then composed into a single function.

# Diffraction

Happens due to wavelength-scale details in surface. For a thin layer, we get phase delay:
$$\delta(\lambda, d, \eta, \theta_t)=\frac{4\pi\ \eta\ d \cos \theta_t}{\lambda }$$
They scale polarized reflection and refraction as follows:

$$r'=\frac{r_1+r_2e^{2i\delta}}{1+r_1 r_2e^{2i\delta}}$$
where $r_1$ and $r_2$ are the entry and exit values for fresnel terms.
# Emission

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


# Eye photosensitivity

Our definitions are wavelength-dependent, but our eyes have a different response for each of the wavelengths. Thus before displaying the radience field must be convolved with the response curves for R, G, and B of our eyes, to get corresponding intensities.

# Effects covered

Covered:
- Matte to glossy reflection (iso/anisotropic), polished/brushed metals.
- Clearcoat & general multilayer stacks (absorption, rough-on-rough, base BSDFs).
- Rough transmission (frosted/etched glass), thin colored films.
- Retroreflection (beads/corner cubes) via normalized vMF lobe.
- Subsurface scattering (diffusion or random-walk).
- Participating media: fog, haze, godrays, atmospheric scattering, Sea foam, breaking waves?
- Environment & area lights, soft shadows (via NEE/MIS).
- Blackbody light emission
- Photolumineshence
- Dusty, wet, shimmery materials

Not covered:
- Full **spectral dispersion**, thin-film interference color fringing, **polarization** (our base is RGB, unpolarized).    
- **Fluorescence/phosphorescence**, bioluminescence (wavelength-shift effects).    
- Coherent wave optics (diffraction, speckle).    
- Non-local texture appearance (true BTF/BRDF-field) unless provided as measured data.