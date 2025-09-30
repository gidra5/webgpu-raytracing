The standard Rendering Equation has the following form:

$$L\left(\omega_{o}\right)=\intop\nolimits_{H^2}\mathrm{f\left(\omega_{o},\omega_{i}\right)L\left(\omega_{i}\right)\cos\left(\omega_{i}\right)d}\omega_{i}$$
where $\omega_{i}$ ranges over the sphere of directions around some point $x$

It is a simplification of a more generic Radiative Transfer Equation (differential form):
https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer
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
$$L\left(\boldsymbol x,\omega\right)=T(\boldsymbol x \to \boldsymbol x_{surf})L_{surf}(\boldsymbol x_{surf}, \omega)+\int_{0}^{t_{surf}}T\left(\boldsymbol x\to \boldsymbol x_{t}\right) V(\boldsymbol x_t,\omega)\mathrm{d}t$$
Where $\boldsymbol x_{surf}$ and $\boldsymbol x_{t}$ are shorthands for $\boldsymbol x_{t}=\boldsymbol x+t\omega$ and $\boldsymbol x_{surf}=\boldsymbol x+t_{surf}\omega$, and trasmittance $T\left(x\to x_{t}\right)$ is the following:
https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance
$$T\left(x\to x_{t}\right)=e^{-\intop\nolimits_0^{t}\sigma_{t}\left(x_{u}, \omega\right)\mathrm{d}u}$$
It also satisfies some properties such as:
$$T(x\to x)=1$$
$$T(x\to z)=T(x\to y)T(y\to z)$$
$$T(x\to z)=T(z\to x)$$

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
* $\sigma_{r}\in[0,1]$  is the absorption factor in the direction $\omega$.

The $f(\boldsymbol x,\omega_{i}\to\omega)$ must also obey normalization constraint:
$$\intop\nolimits_{S^2}f\left(\omega_{i}\to\omega\right)(\boldsymbol n\cdot\omega_{i})\mathrm{d}\omega_{}=1$$

Notice that the surface point is explicit and integration domain is the whole sphere of directions.
For simplicity sake, we could omit the parameters for each of the functions to simplify equations visually. If unclear assume we refer to the equations above.

All the equations above are also parametrized by time, ray origin and wavelength. Only parameters relevant for the un-ambiguation of the equation are written, others are implicitly passed through.
$t_{surf}$ is the boundary condition for the surface hit of a ray and entirely depends on the actual scene. 
https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf
# Reciprocity
chatgpt'd
It is a common assumption that it does not matter in which direction we measure light - from camera to light or the other way.
There are only two functions that depend both on incoming and outgoing light directions - $p(\omega_{i}\to\omega)$ and $f(\omega\to\omega_{i})$. Thus we impose additional constraints on these functions:
$$\sigma_s(\omega_i)p\left(\omega_{i}\to\omega\right)=\sigma_s(\omega)p\left(\omega\to\omega_{i}\right)$$
$$\sigma_r(\omega_i)f\left(\omega_{i}\to\omega\right)=\sigma_r(\omega)f\left(\omega\to\omega_{i}\right)$$
# BSDF

https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
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
# Microgeometry
While general RTE fully describes the radiance, it is unfeasible to render the micro details of objects. Besides unpracticality, such fine details are also imperceivable, since all of the detail is in a single pixel area, which is averaged in the final render. Thus it is a great place for statistical methods that describe microgeometry properties statistically.
In that case for every sample point $x$ we evaluate a statistical model of properties in an infinitesimal volume at that point, which simulates averaged result of fine details in both participating media and surface. 
There were developed two theories that give tools to handle both cases.
Together with broad scattering simulated in raytracing directly, it gives a complete description of radiance in the scene.
## Microfacet theory

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

### Normal Distribution Function
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
### Masking function
The constraints on $G$:
1. $G$ is smooth
2. As $n\cdot\omega\to0$, $G\to0$
3. Proper distribution of normals must project onto $\omega$ the same way as the macro surface. With that we expect that physically plausible distributions must satisfy:
$$\int_HD(m)G(\omega, m)\left<\omega\cdot m\right>dm=\left<\omega\cdot n\right>$$$\left<\omega\cdot m\right>=max(\omega\cdot m,0)$
Where $(x>0)$ is Heaviside function, that is 1 whenever the condition is true.
### Smith's model
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
### Masking-shadowing function
https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
If we only account for a single scattering event, we should also account for shadowing of outgoing ray. If we assume independence of these two processes, we get:
$$G_s(\omega_i,\omega_o, m)=G(\omega_i, m)G(\omega_o, m)$$
While simple, it can underestimate visibility of peaks and valleys, which causes darkening at some angles.

If the heights are normally distributed, we can extend Smith's formulation to account for shadowing, allowing less conservative estimation:
$$G_s(\omega_i, \omega_o)=\frac{1}{1+\Lambda(\omega_i)+\Lambda(\omega_o)}$$

In particular, both guarantee reciprocity of the resulting BSDF.

### Stretch invariance
https://jcgt.org/published/0003/02/03/paper.pdf
Some distributions allow for an easy extension to the anisotropic masking function, since they are invariant under stretching in the following sense:
$$P_2(\bar{m},\alpha)=\frac{1}{\lambda_x\lambda_y}P_2(\frac{\bar{m}}{\lambda},\frac{\alpha}{\lambda}),\text{ for any } \lambda>0$$
Intuitively it means that we can stretch the distribution however much we want, the shape will not change. In that case they can be expressed in terms of a single dimensional distribution $f$:
$$P_2(\bar{m},\alpha)=\frac{1}{\alpha_x\alpha_y}f(\left|\frac{\bar{m}}{\alpha}\right|)$$
When $\alpha_x=\alpha_y=\alpha$ we call it isotropic distribution. 
Consider the $\Lambda$ function with invariance and isotropic distribution assumed:
$$\begin{aligned}
\Lambda(\omega)=\int_{\cot\theta}^{\infty}\int_{-\infty}^{\infty}P_2(x, y)(x\tan\theta-1)dydx &=
\tan\theta\int_{0}^{\infty}\int_{-\infty}^{\infty}P_2(x+\cot\theta, y)x dydx\\ 
&=
\tan\theta\int_{0}^{\infty}\int_{-\infty}^{\infty}\frac{1}{\alpha^2}f(\left|\frac{[x+\cot\theta,y]}{\alpha}\right|)xdydx\\
&=\frac{\tan\theta}{\alpha^2}
\int_{0}^{\infty}\int_{-\infty}^{\infty}f(\left|\frac{[x+\cot\theta,y]}{\alpha}\right|)xdydx
\end{aligned}$$

$$(x+\cot\theta)/\alpha=r\cos\phi$$
$$y/\alpha=r\sin\phi$$
$$\alpha r\cos\phi-\cot\theta\ge0$$
$$r\ge\frac{\cot\theta}{\alpha\cos\phi}=\frac{a}{\cos\phi}$$
$$\begin{aligned}
\int_{0}^{\infty}\int_{-\infty}^{\infty}f(\left|\frac{[x+\cot\theta,y]}{\alpha}\right|)xdydx &= \alpha^2\int_{-\pi/2}^{\pi/2}\int_{\frac{a}{\cos\phi}}^{\infty}f(r)(r\alpha\cos\phi-\cot\theta)r\ dr\ d\phi
\end{aligned}$$
$$\begin{aligned}
\int_{-\pi/2}^{\pi/2}\int_{\frac{a}{\cos\phi}}^{\infty}f(r)(\alpha r\cos\phi-\cot\theta)r\ dr\ d\phi 
&= \alpha\int_{-\pi/2}^{\pi/2}\cos\phi\int_{\frac{a}{\cos\phi}}^{\infty}f(r)r^2 drd\phi \\&- \cot\theta \int_{-\pi/2}^{\pi/2}\int_{\frac{a}{\cos\phi}}^{\infty}f(r)rdrd\phi\\
\end{aligned}$$
$$\begin{aligned}
\int f(r)rdr=r\int f(r)dr-\int\int f(r)drdr=rF_1(r)-\int F(r)dr=rF_1(r)-F_2(r)
\end{aligned}$$
$$\begin{aligned}
\int f(r)r^2dr&=r^2\int f(r)dr-\int\int f(r)dr2rdr\\&=r^2F(r)-2\int F(r)rdr\\&=r^2F(r)-2rF_2(r)+2\int F_2(r)dr\\&=r^2F_1(r)-2rF_2(r)+2F_3(r)
\end{aligned}$$
$$\begin{aligned}
\int_{-\pi/2}^{\pi/2}\cos\phi\int_{\frac{\cot\theta}{\alpha_x\cos\phi}}^{\infty}f(r)r^2 drd\phi
\end{aligned}$$
$$\begin{aligned}
\int_{-\pi/2}^{\pi/2}\int_{\frac{a}{\cos\phi}}^{\infty}f(r,\gamma)rdrd\phi
&=\int_{-\pi/2}^{\pi/2}(rF_1(r)-F_2(r))\Big|_{\frac{a}{\cos\phi}}^{\infty}d\phi\\
&=\int_{-\pi/2}^{\pi/2}\lim_{r\to\infty}(rF_1(r)-F_2(r))-(\frac{a}{\cos\phi}F_1(\frac{a}{\cos\phi})-F_2(\frac{a}{\cos\phi}))d\phi\\
&=\lim_{r\to\infty}(rF_1(r)-F_2(r))-\int_{-\pi/2}^{\pi/2}\frac{a}{\cos\phi}F_1(\frac{a}{\cos\phi})-F_2(\frac{a}{\cos\phi})d\phi\\
&=\lim_{r\to\infty}(rF_1(r)-F_2(r))-2\int_{0}^{\pi/2}\frac{a}{\cos\phi}F_1(\frac{a}{\cos\phi})-F_2(\frac{a}{\cos\phi})d\phi
\end{aligned}$$
$$\begin{aligned}
&r=\frac{a}{\cos\phi}\\
&\cos\phi=\frac{a}{r}\\
&\phi=\arccos(\frac{a}{r})\\
&d\phi=\frac{1}{\sqrt{1-(\frac{a}{r})^2}}d(\frac{a}{r})=\frac{1}{\sqrt{1-(\frac{a}{r})^2}}\frac{-a}{r^2}dr=\frac{-1}{r\sqrt{(\frac{r}{a})^2-1}}dr\\
\end{aligned}$$
$$\begin{aligned}
\int_{0}^{\pi/2}\frac{a}{\cos\phi}F_1(\frac{a}{\cos\phi})-F_2(\frac{a}{\cos\phi})d\phi
&=\int_{a}^{\infty}\frac{F_2(r)-rF_1(r)}{r\sqrt{(\frac{r}{a})^2-1}}dr\\
\end{aligned}$$
$$\begin{aligned}
&\Lambda(\omega)= \frac{1}{a}\int_{-\pi/2}^{\pi/2}\cos\phi\int_{\frac{a}{\cos\phi}}^{\infty}f(r)r^2 drd\phi - \int_{-\pi/2}^{\pi/2}\int_{\frac{a}{\cos\phi}}^{\infty}f(r)rdrd\phi \\&
a=\frac{1}{\alpha\tan\theta}
\end{aligned}$$
We can reduce anisotropic distribution to isotropic with roughness $\alpha_y$ by stretching it along x-axis by $\alpha_y/\alpha_x$. Since $\Lambda$ only depends on $\omega$, we just need to transform it into stretched coordinates:
$$\omega'=[\frac{\alpha_x}{\alpha_y}\omega_x,\omega_y,\omega_z]$$
$${\tan\theta'}={\sqrt{(\frac{\alpha_x}{\alpha_y}\sin\phi)^2+\cos^2\phi}\tan\theta}$$
Then if we look at parameter a, that we derived above, we should express it in new coords:
$$a=\frac{1}{\alpha_y\tan\theta'}=\frac{1}{\alpha_y{\sqrt{(\frac{\alpha_x}{\alpha_y}\sin\phi)^2+\cos^2\phi}\tan\theta}}=\frac{1}{{\sqrt{(\alpha_x\sin\phi)^2+(\alpha_y\cos\phi)^2}\tan\theta}}=\frac{1}{\alpha\tan\theta}$$
In that case isotropic roughness $\alpha$ has the following value in terms of a roughness projected onto the outgoing direction $\omega_o$:
$$\alpha=\sqrt{(\alpha_x\sin\phi)^2+(\alpha_y\cos\phi)^2}=\frac{|[\alpha_x\omega_x, \alpha_y\omega_y]|}{\sin\theta}$$
### Unaligned stretching
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
### Vertical Shearing and Non-Centered Distributions
https://jcgt.org/published/0003/02/03/paper.pdf
Since all the results are derived from slope distribution $P_2$, we can also introduce average slope $\widetilde{m}$ distinct from zero. That would allow us to accurately represent normal and bump maps, frequently used to add detail. The surface created by off-center the average slope is called meso-surface, being intermediate between macro and micro representation. 
Note that in the presence of meso-surface, the projected area of the micro-surface, as well as all other $\omega\cdot n$ factors, must be adjusted:
$$\intop\nolimits_{H^2}(\boldsymbol v\cdot\omega)D\mathrm{d}\omega=\frac{v\cdot \widetilde m}{n\cdot \widetilde m}$$
### Generalized Trowbridge–Reitz model
https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
Lets consider a generic distribution of slopes, parametrized by power $\gamma$ and roughness $\alpha$:
$$f(r)=\frac{\gamma-1}{\pi(1+r^2)^\gamma}$$
$$\begin{aligned}
F(r)&=\frac{\gamma-1}{\pi}\int (1+r^2)^{-\gamma}dr\\
&=\frac{\gamma-1}{\pi} r F_{2,1}(1/2,\gamma,3/2,-r^2)+C
\end{aligned}$$
$$\begin{aligned}
\int F(r)dr&=\frac{\gamma-1}{\pi} \int r F_{2,1}(1/2,\gamma,3/2,-r^2)dr\\
&=\frac{\gamma-1}{\pi}\left(r^2F_{2,1}(r)+C_1r+C_2+\frac{1}{2(\gamma-1)(1+r^2)^{\gamma-1}}\right)\\
&=\frac{\gamma-1}{\pi}r^2F_{2,1}(r)+C_1r+C_2+\frac{\gamma-1}{\pi}\frac{1}{2(\gamma-1)(1+r^2)^{\gamma-1}}\\
&=rF(r)+\frac{1}{2\pi(1+r^2)^{\gamma-1}}+C_2
\end{aligned}$$
$$
P_2(\bar{m})=\frac{\gamma-1}{\pi\alpha^2(1+\left|\frac{\bar{m}}{\alpha}\right|^2)^\gamma}
$$
From it we can derive NDF and masking functions:
$$D(m)=\frac{\gamma-1}{\pi\alpha^2(m\cdot n)^4(1+\left|\frac{\bar{m}}{\alpha}\right|^2)^\gamma}$$
$$\begin{aligned}
\int\frac{\gamma-1}{(1+r^2)^\gamma}rdr=\int\frac{\gamma-1}{(1+r^2)^\gamma}\frac 1 2 d(1+r^2)=-\frac {(1+r^2)^{1-\gamma}} 2
\end{aligned}$$
$$\begin{aligned}
\int\frac{(\gamma-1)r^2}{(1+r^2)^\gamma} dr&=r\int \frac{\gamma-1}{(1+r^2)^\gamma}r dr -\int \int \frac{\gamma-1}{(1+r^2)^\gamma}r drdr\\
&=-r\frac {(1+r^2)^{1-\gamma}} 2 +\frac 1 2\int (1+r^2)^{1-\gamma} dr
\end{aligned}$$
$$\begin{aligned}
\Lambda(\omega)&= \frac{1}{a}\int_{-\pi/2}^{\pi/2}\cos\phi\int_{\frac{a}{\cos\phi}}^{\infty}\frac{\gamma-1}{\pi(1+r^2)^\gamma}r^2 drd\phi - \int_{-\pi/2}^{\pi/2}\int_{\frac{a}{\cos\phi}}^{\infty}\frac{\gamma-1}{\pi(1+r^2)^\gamma}rdrd\phi \\
\end{aligned}$$
$$\begin{aligned}
\int_{\frac{a}{\cos\phi}}^{\infty}\frac{\gamma-1}{(1+r^2)^\gamma}r^2 dr
&= \frac 1 2 \int_{\frac{a}{\cos\phi}}^{\infty} (1+r^2)^{1-\gamma} dr - \frac 1 2 r(1+r^2)^{1-\gamma}\Big|_{\frac{a}{\cos\phi}}^{\infty}\\
&= \frac 1 2 \lim_{r\to\infty}(r(1+r^2)^{1-\gamma})- \frac 1 2 \frac{a}{\cos\phi}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}+\frac 1 2 \int_{\frac{a}{\cos\phi}}^{\infty} (1+r^2)^{1-\gamma} dr \\
&= \frac {\beta_1} 2- \frac 1 2 \frac{a}{\cos\phi}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}+\frac 1 2 \int_{\frac{a}{\cos\phi}}^{\infty} (1+r^2)^{1-\gamma} dr \\
\beta_1&=\lim_{r\to\infty}\frac{r}{(1+r^2)^{\gamma-1}}
\end{aligned}$$
$$\begin{aligned}
\int_{\frac{a}{\cos\phi}}^{\infty} (1+r^2)^{1-\gamma} dr
&=\frac{\gamma-1}{\pi} r F_{2,1}(1/2,\gamma,3/2,-r^2)\Big|_{\frac{a}{\cos\phi}}^{\infty}\\
&=\frac{\gamma-1}{\pi}(\lim_{r\to\infty} r F_{2,1}(1/2,\gamma,3/2,-r^2)- \frac{a}{\cos\phi} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))\\
&=\beta_2 - \frac{\gamma-1}{\pi}\frac{a}{\cos\phi} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2)\\
\beta_2&=\frac{\gamma-1}{\pi}\lim_{r\to\infty} r F_{2,1}(1/2,\gamma,3/2,-r^2)
\end{aligned}$$
$$\begin{aligned}
\frac{1}{a}\int_{-\pi/2}^{\pi/2}\cos\phi\int_{\frac{a}{\cos\phi}}^{\infty}\frac{\gamma-1}{(1+r^2)^\gamma}r^2 drd\phi=\\

\frac{1}{a}\int_{-\pi/2}^{\pi/2}\cos\phi(\frac {\beta_1} 2 - \frac{a}{2\cos\phi}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}+\frac {\beta_2} 2- \frac{\gamma-1}{2\pi}\frac{a}{\cos\phi} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi =\\ 

\frac{1}{a}\frac {\beta_1+\beta_2} 2\int_{-\pi/2}^{\pi/2}\cos\phi d\phi- \frac 1 2 \int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}+ \frac{\gamma-1}{2\pi}F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi =\\

\frac{\beta_1+\beta_2}{a}- \frac 1 2\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi- \frac{\gamma-1}{2\pi}\int_{-\pi/2}^{\pi/2} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi
\end{aligned}$$
$$\begin{aligned}
\int_{-\pi/2}^{\pi/2}(1+r^2)^{1-\gamma}\Big|_{\frac{a}{\cos\phi}}^{\infty}d\phi &= \lim_{r\to\infty}(1+r^2)^{1-\gamma}-\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi\\
&= \beta_3-\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi\\
\beta_3&=\lim_{r\to\infty}(1+r^2)^{1-\gamma}
\end{aligned}$$
[HGMfromEuler-arXiv.pdf](https://jvoight.github.io/articles/HGMfromEuler-arXiv.pdf)
$$\begin{aligned}
\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi &= 
\int_{-\pi/2}^{\pi/2}\cos^{2\gamma-2}\phi(\cos^2\phi+a^2)^{1-\gamma}d\phi \\
&= 2\int_{0}^{\pi/2}\cos^{2\gamma-2}\phi(\cos^2\phi+a^2)^{1-\gamma}d\phi\\
u=\cos^2\phi,\ &d\phi=\frac{du}{-2\sqrt{u(1-u)}}\\
\int_{0}^{\pi/2}\cos^{2\gamma-2}\phi(\cos^2\phi+a^2)^{1-\gamma}d\phi&=\int_{0}^{1}u^{\gamma-1}(u+a^2)^{1-\gamma}\frac{du}{-2\sqrt{u(1-u)}}\\
&=-\frac {a^{2(1-\gamma)}} 2\int_{0}^{1}u^{\gamma-\frac 3 2}(1-u)^{-\frac 1 2}(\frac u {a^2}+1)^{1-\gamma}du\\
&=-\frac {a^{2(1-\gamma)}} 2B(\gamma-\frac 1 2, \frac 1 2)F_{2,1}(\gamma-1,\gamma-\frac 1 2,\gamma,-\frac 1 {a^2})\\
&=-\frac {a^{2(1-\gamma)}\Gamma(\gamma-\frac 1 2)\Gamma(\frac 1 2)} {2\Gamma(\gamma)}F_{2,1}(\gamma-1,\gamma-\frac 1 2,\gamma,-\frac 1 {a^2})
\end{aligned}$$
$$\begin{aligned}
\Lambda(\omega)&= \frac{1}{a}\int_{-\pi/2}^{\pi/2}\cos\phi\int_{\frac{a}{\cos\phi}}^{\infty}\frac{\gamma-1}{\pi(1+r^2)^\gamma}r^2 drd\phi + \int_{-\pi/2}^{\pi/2}\int_{\frac{a}{\cos\phi}}^{\infty}\frac{\gamma-1}{\pi(1+r^2)^\gamma}r drd\phi \\
&=\frac{\beta_1+\beta_2}{a\pi}- \frac 1 {2\pi}\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi- \frac{\gamma-1}{2\pi^2} \int_{-\pi/2}^{\pi/2}F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi\\
&+ \frac {\beta_3} {2\pi}-\frac 1 {2\pi}\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi\\
&=\frac{\beta_1+\beta_2}{a\pi}- \frac 1 \pi\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi- \frac{\gamma-1}{2\pi^2} \int_{-\pi/2}^{\pi/2} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi\\
&+ \frac {\beta_3} {2\pi}\\
&=\frac{\beta_1+\beta_2}{a\pi}+ \frac {a^{2(1-\gamma)}\Gamma(\gamma-\frac 1 2)\Gamma(\frac 1 2)} {2\pi\Gamma(\gamma)}F_{2,1}(\gamma-1,\gamma-\frac 1 2,\gamma,-\frac 1 {a^2})- \frac{\gamma-1}{2\pi^2} \int_{-\pi/2}^{\pi/2} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi\\
&+ \frac {\beta_3} {2\pi}
\end{aligned}$$

Note that with $\gamma\to\infty$ it approaches normal distribution, which is the basis for Beckmann distribution. For $\gamma=1$ it results in regular Trowbridge–Reitz model.

glints
https://cseweb.ucsd.edu/~ravir/glints.pdf
https://rgl.epfl.ch/publications/Zeltner2020Specular
https://igg.unistra.fr/People/chermain/real_time_glint/
https://rgl.epfl.ch/publications/Loubet2020Slope
## Microflake theory
Microfacet theory assumes the facets form a single surface. If we relax this assumption such that facets can be positioned arbitrarily in micro-volume, then we basically get small plane-like dielectric flakes, which opens a possibility for modeling small-scale multi-bounces and subsurface scattering in thin surface volumes.

particle density $\sigma_p$
albedo $\alpha$
NDF $D$
$$\sigma_a(\omega)=\sigma_p(1-\alpha)\intop\nolimits_{S^2}(m\cdot \omega)D(m)dm$$
$$\sigma_s(\omega)=\sigma_p\alpha\intop\nolimits_{S^2}(m\cdot \omega)D(m)dm$$
$$\rho(\omega_i\to\omega)=\frac{\alpha}{\sigma_s(\omega_i)}D(\frac{\omega_i+\omega}{|\omega_i+\omega|})D(-\frac{\omega_i+\omega}{|\omega_i+\omega|})$$
https://cseweb.ucsd.edu/~tzli/cse272/wi2023/lectures/11_microflake.pdf
https://research.nvidia.com/sites/default/files/pubs/2015-08_The-SGGX-microflake/sggx.pdf
https://onrendering.com/data/papers/ms16/ms16.pdf
https://arxiv.org/pdf/2110.07145
## Multibounce microfacets
https://arxiv.org/pdf/2110.07145
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

# Layered Materials
https://www.pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Scattering_from_Layered_Materials
https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2014Comprehensive_2.pdf
https://hal.science/hal-01785457/document
https://arxiv.org/pdf/2110.07145
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
# Diffraction
chatgpt'd
https://eugenedeon.com/
https://ssteinberg.xyz/2024fsdbsdf/steinberg2024_fsd_paper.pdf
Happens due to wavelength-scale details in surface. For a thin layer, we get phase delay:
$$\delta(\lambda, d, \eta, \theta_t)=\frac{4\pi\ \eta\ d \cos \theta_t}{\lambda }$$
They scale polarized reflection and refraction as follows:

$$r'=\frac{r_1+r_2e^{2i\delta}}{1+r_1 r_2e^{2i\delta}}$$
where $r_1$ and $r_2$ are the entry and exit values for fresnel terms.

iridescence
https://hal.science/hal-01518344/file/paper-small%20%281%29.pdf

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
Sum over lenses
Integrate over aperture
Integrate over exposure time
Integrate over wavelengths (importance sample by photosensitivity)
Apply bloom (diffraction pattern)
Convert collected intensities for each wavelength to rgb
### Eye photosensitivity

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

We also need inverse transformations to transform a material color into spectral reflectance distribution.

### Antialiasing
https://www.iryoku.com/aacourse/
https://www.reddit.com/r/GraphicsProgramming/s/f26q2kQi56
### Depth of field
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

# Measurement fitting
god damn its so hard
# Artistic parametrization
https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
Reformulation with a different set of parameters, that is much more artist-friendly.
# more

fur and hair rendering
http://kunzhou.net/2013/fur-rendering-tvcg.pdf
https://www.pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair

subsurface scattering
https://users.cg.tuwien.ac.at/zsolnai/wp/wp-content/uploads/2014/12/ssss.pdf
