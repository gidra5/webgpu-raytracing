The standard Rendering Equation has the following form:

$$L\left(\omega_{o}\right)=\intop\nolimits_{H^2}\mathrm{f\left(\omega_{o},\omega_{i}\right)L\left(\omega_{i}\right)\cos\left(\omega_{i}\right)d}\omega_{i}$$
where $\omega_{i}$ ranges over the sphere of directions around some point $x$

It is a simplification of a more generic Radiative Transfer Equation (differential form):
$$\omega\cdot\nabla L\left(\boldsymbol x,\omega\right)+\sigma_{t}\left(\boldsymbol x, \omega\right)L\left(\boldsymbol x,\omega\right)=Q\left(\boldsymbol x,\omega\right)+\sigma_{s}\left(\boldsymbol x, \omega\right)\intop\nolimits_{S^2} p\left(\boldsymbol x,\omega_{i}\to\omega\right)L\left(\boldsymbol x,\omega_{i}\right)\mathrm{d}\omega_{i}$$
Where
* $\boldsymbol x$ is the ray origin
* $\omega$ is the ray direction
* $L(\boldsymbol x,\omega)$ the incoming radiance to the point $\boldsymbol x$ from the direction $\omega$
* $Q(\boldsymbol x,\omega)$ is the emission of light at the point $\boldsymbol x$ in the direction $\omega$. A free parameter
* $p(\boldsymbol x,\omega_{i}\to\omega)$ is the phase function - probability density for scattering at the point $\boldsymbol x$ from the direction $\omega_{i}$ in the direction $\omega$. A free parameter
* $\sigma_{a}(\boldsymbol x,\omega)$, $\sigma_{s}(\boldsymbol x,\omega)$, $\sigma_{t}(\boldsymbol x,\omega)=\sigma_{a}(\boldsymbol x,\omega)+\sigma_{s}(\boldsymbol x,\omega)$ are the absorption, scattering, and extinction coefficients respectively for a given point $\boldsymbol x$ along the direction $\omega$. All must be non-negative values. A free parameter
The $p(\boldsymbol x,\omega_{i}\to\omega)$ must obey normalization constraint:
$$\intop\nolimits_{S^2}p\left(x,\omega_{i}\to\omega\right)\mathrm{d}\omega_{}=1$$

Radiative Transfer Equation in integral form:
$$L\left(\boldsymbol x,\omega\right)=T(\boldsymbol x \to \boldsymbol x_{surf}, \omega)L_{surf}(\boldsymbol x_{surf}, \omega)+\int_{0}^{t_{surf}}T\left(\boldsymbol x\to \boldsymbol x_{t}, \omega\right)\left[\sigma_{s}\left(\boldsymbol x_{t}\right)\intop\nolimits_{S^2}p\left(\boldsymbol x_{t}, \omega_{i}\to\omega\right)L(\boldsymbol x_{t}, \omega_{i})\mathrm{d}\omega_{i}+Q\left(x_{t},\omega\right)\right]\mathrm{d}t$$
Where $\boldsymbol x_{surf}$ and $\boldsymbol x_{t}$ are shorthands for $\boldsymbol x_{t}=\boldsymbol x+t\omega$ and $\boldsymbol x_{surf}=\boldsymbol x+t_{surf}\omega$, and $T\left(x\to x_{t}\right)$ is the following:
$$T\left(x\to x_{t}, \omega\right)=e^{-\intop\nolimits_0^{t}\sigma_{t}\left(x_{u}, \omega\right)\mathrm{d}u}$$

Free parameters are also parametrized by time (so could change over time). $t_{surf}$ is also a free term, since it entirely depends on the actual scene. Thus it is time-dependent as well.

The $L_{surf}$ term is the one expressed in a standard rendering equation. But it is usually simplified to only consider reflected light. The exact form is as follows:
$$L_{surf}\left(\boldsymbol x,\omega\right)=L_{e}\left(\boldsymbol x,\omega\right)+\intop\nolimits_{S^2}f(\boldsymbol x,\omega, \omega_{i})L(\boldsymbol x,\omega_{i})(\boldsymbol n(\boldsymbol x)\cdot\omega_{i})\mathrm{d}\omega_{i}$$
Notice that the surface point is explicit and integration domain is the whole sphere of directions.
