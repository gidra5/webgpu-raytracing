$$f(r)=\frac{1}{\pi(1+\frac{r^2}{\gamma-1})^\gamma}$$
$$
P_2(\bar{m})=\frac{1}{\pi\alpha^2\left(1+\frac{\left|\frac{\bar{m}}{\alpha}\right|^2}{\gamma-1}\right)^\gamma}
$$
Idk where expression below comes from, but it seems to agree numerically. 
chatgpt'd
$$\begin{aligned}
&\Lambda(\omega)=\frac 1 a\frac {\sqrt{\gamma-1}\Gamma(\gamma-\frac 1 2)}{\alpha\sqrt\pi\Gamma(\gamma)}\int_{a}^{\infty}(x-a)\left(1+\frac {x^2} {\alpha^2(\gamma-1)}\right)^{-(\gamma-\frac 1 2)}dx\\
&\int_{a}^{\infty}(x-a)\left(1+\frac {x^2} {\alpha^2(\gamma-1)}\right)^{-(\gamma-\frac 1 2)}dx\\&=\int_{a}^{\infty}x\left(1+\frac {x^2} {\alpha^2(\gamma-1)}\right)^{-(\gamma-\frac 1 2)}dx-a\int_{a}^{\infty}\left(1+\frac {x^2} {\alpha^2(\gamma-1)}\right)^{-(\gamma-\frac 1 2)}dx\\&=(-\frac{\alpha^2(\gamma-1)(1+\frac {x^2}{\alpha^2(\gamma-1)})^{1-(\gamma-\frac 1 2)}}{2((\gamma-\frac 1 2)-1)})\Bigg|_{a}^{\infty}-a\int_{a}^{\infty}\left(1+\frac {x^2} {\alpha^2(\gamma-1)}\right)^{-(\gamma-\frac 1 2)}dx\\

&=(-\frac{\alpha^2(\gamma-1)(1+\frac {x^2}{\alpha^2(\gamma-1)})^{\frac 3 2-\gamma}}{2\gamma-3})\Bigg|_{a}^{\infty}-a(xF_{2,1}(\frac 1 2,\gamma-\frac 1 2,\frac 3 2, -\frac {x^2}{\alpha^2(\gamma-1)}))\Bigg|_{a}^{\infty}\\

&\lim_{x\to\infty}(1+\frac {x^2}{\alpha^2(\gamma-1)})^{\frac 3 2-\gamma}=\lim_{x\to\infty}\frac {(1+\frac {x^2}{\alpha^2(\gamma-1)})^{\frac 3 2}} {(1+\frac {x^2}{\alpha^2(\gamma-1)})^{\gamma}}=\begin{cases}
\infty  & \text{if } \gamma\lt\frac 3 2\\
1 & \text{if } \gamma=\frac 3 2\\
0 & \text{othewise}
\end{cases}\\
&\lim_{x\to\infty}xF_{2,1}(\frac 1 2,\gamma-\frac 1 2,\frac 3 2, -\frac {x^2}{\alpha^2(\gamma-1)})=\frac {\sqrt {\pi\alpha^2(\gamma-1)} \Gamma(\gamma-1)}{2\Gamma(\gamma-\frac 1 2)}+\begin{cases}
1 & \text{if } \gamma=1\\
0 & \text{othewise}
\end{cases}\\

&\int_{a}^{\infty}(x-a)\left(1+\frac {x^2} {\alpha^2(\gamma-1)}\right)^{-(\gamma-\frac 1 2)}dx\\&=\frac{\alpha^2(1-\gamma)}{2\gamma-3}(\begin{cases}
\infty  & \text{if } \gamma\lt\frac 3 2\\
1 & \text{if } \gamma=\frac 3 2\\
0 & \text{othewise}
\end{cases}-(1+\frac {a^2}{\alpha^2(\gamma-1)})^{\frac 3 2-\gamma})\\&-a(\frac {\sqrt {\pi\alpha^2(\gamma-1)} \Gamma(\gamma-1)}{2\Gamma(\gamma-\frac 1 2)}+\begin{cases}
1 & \text{if } \gamma=1\\
0 & \text{othewise}
\end{cases}-aF_{2,1}(\frac 1 2,\gamma-\frac 1 2,\frac 3 2, -\frac {a^2}{\alpha^2(\gamma-1)}))\\
&=\frac{\alpha^2(\gamma-1)}{2\gamma-3}(1+\frac {a^2}{\alpha^2(\gamma-1)})^{\frac 3 2-\gamma}-a\frac {\sqrt {\pi\alpha^2(\gamma-1)} \Gamma(\gamma-1)}{2\Gamma(\gamma-\frac 1 2)}+a^2F_{2,1}(\frac 1 2,\gamma-\frac 1 2,\frac 3 2, -\frac {a^2}{\alpha^2(\gamma-1)})\\


&\Lambda(\omega)=\frac 1 a\frac {\sqrt{\gamma-1}\Gamma(\gamma-\frac 1 2)}{\alpha\sqrt\pi\Gamma(\gamma)}\int_{a}^{\infty}(x-a)\left(1+\frac {x^2} {\alpha^2(\gamma-1)}\right)^{-(\gamma-\frac 1 2)}dx\\
&=\frac 1 a\frac {\sqrt{\gamma-1}\Gamma(\gamma-\frac 1 2)}{\alpha\sqrt\pi\Gamma(\gamma)}(\frac{\alpha^2(\gamma-1)}{2\gamma-3}(1+\frac {a^2}{\alpha^2(\gamma-1)})^{\frac 3 2-\gamma}\\&-a\frac {\sqrt {\pi\alpha^2(\gamma-1)} \Gamma(\gamma-1)}{2\Gamma(\gamma-\frac 1 2)}+a^2F_{2,1}(\frac 1 2,\gamma-\frac 1 2,\frac 3 2, -\frac {a^2}{\alpha^2(\gamma-1)}))\\
&=\frac {\alpha\sqrt{\gamma-1}\Gamma(\gamma-\frac 3 2)}{2a\sqrt\pi\Gamma(\gamma-1)}(1+\frac {a^2}{\alpha^2(\gamma-1)})^{\frac 3 2-\gamma}-\frac {1}{2}+\frac {\sqrt{\gamma-1}\Gamma(\gamma-\frac 1 2)}{\alpha\sqrt\pi\Gamma(\gamma)}aF_{2,1}(\frac 1 2,\gamma-\frac 1 2,\frac 3 2, -\frac {a^2}{\alpha^2(\gamma-1)})\\
&=\frac {\Gamma(\gamma-\frac 3 2)}{2b\sqrt\pi\Gamma(\gamma-1)}(1+b^2)^{\frac 3 2-\gamma}-\frac {1}{2}+\frac {b\Gamma(\gamma-\frac 1 2)}{\alpha^2\sqrt\pi\Gamma(\gamma)}F_{2,1}(\frac 1 2,\gamma-\frac 1 2,\frac 3 2, -b^2)\\
&b=\frac {a}{\alpha\sqrt{\gamma-1}}
\end{aligned}$$