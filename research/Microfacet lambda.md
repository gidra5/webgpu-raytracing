
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
[HGMfromEuler-arXiv.pdf](https://jvoight.github.io/articles/HGMfromEuler-arXiv.pdf)
$$F_{2,1}(a,b,c,z)=\frac{1}{B(b, c-b)}\int_{0}^{1}x^{b-1}(1-x)^{c-b-1}(1-zx)^{-a}dx$$
$$\int_{0}^{1}x^{b}(1-x)^{c}(1-zx)^a dx=F_{2,1}(-a,b+1,c+b+2,z)B(b+1, c+1)$$
$$\begin{aligned}
&\int_{-\pi/2}^{\pi/2} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi\\&=
\frac{1}{B(\gamma, 3/2-\gamma)}\int_{-\pi/2}^{\pi/2} \int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}(1+(\frac{a}{\cos\phi})^2x)^{-1/2}dxd\phi\\&=
\frac{1}{B(\gamma, 3/2-\gamma)}\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}\int_{-\pi/2}^{\pi/2} (1+(\frac{a}{\cos\phi})^2x)^{-1/2}d\phi dx
\end{aligned}$$
$$\begin{aligned}
&\int_{-\pi/2}^{\pi/2} (1+(\frac{a}{\cos\phi})^2x)^{-1/2}d\phi\\&=
\int_{-\pi/2}^{\pi/2} (1+\frac{a^2x}{\cos^2\phi})^{-1/2}d\phi\\&=
\int_{-\pi/2}^{\pi/2} (\frac{\cos^2\phi+ a^2x}{\cos^2\phi})^{-1/2}d\phi\\&=
\int_{-\pi/2}^{\pi/2} \cos\phi(\cos^2\phi+ a^2x)^{-1/2}d\phi\\&=
2\int_{0}^{\pi/2} \cos\phi(\cos^2\phi+ a^2x)^{-1/2}d\phi\\&=
2\int_{0}^{\pi/2} (1+ a^2x-\sin^2\phi)^{-1/2}d(\sin\phi)\\&=
2\int_{0}^{1} (1+ a^2x-u^2)^{-1/2}du\\&=
2\int_{0}^{1} (1+ a^2x)^{-1/2}(1-\frac{u^2}{1+ a^2x})^{-1/2}du\\&=
2\int_{0}^{1} (1-(\frac{u}{\sqrt{1+ a^2x}})^2)^{-1/2}\frac{du}{\sqrt{1+ a^2x}}\\&=
2\arcsin\left(\frac{u}{\sqrt{1+ a^2x}}\right)\Bigg|_0^1\\&=
2(\arcsin\left(\frac{1}{\sqrt{1+ a^2x}}\right)-\arcsin(0))\\&=
2\arcsin\left(\frac{1}{\sqrt{1+ a^2x}}\right)\\&=
2\frac{1}{\sqrt{1+ a^2x}}F_{2,1}\left(1/2,1/2,3/2,(\frac{1}{\sqrt{1+ a^2x}})^2\right)\\&=
2\frac{1}{\sqrt{1+ a^2x}}F_{2,1}\left(1/2,1/2,3/2,\frac{1}{1+ a^2x}\right)
\end{aligned}$$
https://en.wikipedia.org/wiki/Hypergeometric_function#Transformation_formulas
$$\begin{aligned}
&F_{2,1}\left(a,b,c,z\right)=(1-z)^{-a}F_{2,1}\left(a,c-b,c,\frac{z}{1-z}\right)\\
&F_{2,1}\left(a,b,c,z\right)=(1-z)^{-b}F_{2,1}\left(b,c-a,c,\frac{z}{1-z}\right)
\end{aligned}$$
$$\begin{aligned}
&F_{2,1}\left(1/2,1/2,3/2,\frac{1}{1+ a^2x}\right)\\&=
\left(1-\frac{1}{1+ a^2x}\right)^{-1/2}F_{2,1}\left(1/2,3/2-1/2,3/2,\frac{\frac{1}{1+ a^2x}}{\frac{1}{1+ a^2x}-1}\right)\\&=
\left(\frac{a^2x}{1+ a^2x}\right)^{-1/2}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)
\end{aligned}$$
https://physicsgg.me/wp-content/uploads/2019/11/handbook-of-mathematical-functions-by-m.-abramowitz-i.-stegun.pdf p. 559
$$\begin{aligned}
F_{2,1}\left(a,b,c,z\right)&=\frac{\Gamma(c)\Gamma(b-a)}{\Gamma(b)\Gamma(c-a)}(-z)^{-a}F_{2,1}\left(a,1-c+a,1-b+a,1/z\right)\\&+\frac{\Gamma(c)\Gamma(a-b)}{\Gamma(a)\Gamma(c-b)}(-z)^{-b}F_{2,1}\left(b,1-c+b,1-a+b,1/z\right)
\end{aligned}$$
$$\begin{aligned}
F_{2,1}\left(1/2,1,3/2,-z^2\right)=z^{-1}\arctan(z)
\end{aligned}$$$$\begin{aligned}
F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)=F_{2,1}\left(1/2,1,3/2,-(\frac{1}{a\sqrt x})^2\right)=a\sqrt x \arctan(\frac{1}{a\sqrt x})
\end{aligned}$$
$$\begin{aligned}
F_{2,1}\left(a,b,c,z\right)=F_{2,1}\left(b,a,c,z\right)
\end{aligned}$$
$$B(b,c-b)F_{2,1}(a,b,c,z)=\int_0^1x^{b-1}(1-x)^{c-b-1}(1-zx)^{-a}dx$$
$$B(b+1,c+1)F_{2,1}(-a,b+1,c+b+2,z)=\int_0^1x^{b}(1-x)^{c}(1-zx)^{a}dx$$
$$B(b,c)F_{2,1}(a,b,c+b,z)=\int_0^1x^{b-1}(1-x)^{c-1}(1-zx)^{-a}dx$$
$$\frac d {dz} F_{2,1}(a,b,c,z)=\frac{ab}{c}F_{2,1}(a+1,b+1,c+1,z)$$
$$\frac d {dz} F_{2,1}(a,b,c,z^2)=\frac{2abz}{c}F_{2,1}(a+1,b+1,c+1,z^2)$$
$$\frac d {dz} F_{2,1}(a,b,c,z^2)=\frac{2abz}{c}F_{2,1}(a+1,b+1,c+1,z^2)$$
$$\frac {d^n} {dz^n} (F_{2,1}(a,b,c,z))=\frac{2abz}{c}F_{2,1}(a+1,b+1,c+1,z^2)$$
$$\begin{aligned}
&F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)\\
&=\frac{\Gamma(3/2)\Gamma(1-1/2)}{\Gamma(1)\Gamma(3/2-1/2)}(-\frac{-1}{a^2x})^{-1/2}F_{2,1}\left(1/2,1-3/2+1/2,1-1+1/2,1/\frac{-1}{a^2x}\right)\\&+\frac{\Gamma(3/2)\Gamma(1/2-1)}{\Gamma(1/2)\Gamma(3/2-1)}(-\frac{-1}{a^2x})^{-1}F_{2,1}\left(1,1-3/2+1,1-1/2+1,1/\frac{-1}{a^2x}\right)\\
&=\frac{\Gamma(3/2)\Gamma(1/2)}{\Gamma(1)\Gamma(1)}a\sqrt{x}F_{2,1}\left(1/2,0,1/2,-a^2x\right)+\frac{\Gamma(3/2)\Gamma(-1/2)}{\Gamma(1/2)\Gamma(1/2)}a^2xF_{2,1}\left(1,1/2,3/2,-a^2x\right)\\
&=\frac {\pi a\sqrt{x}} 2 -a^2xF_{2,1}\left(1,1/2,3/2,-a^2x\right)
\end{aligned}$$
$$\begin{aligned}
&\int_{-\pi/2}^{\pi/2} (1+(\frac{a}{\cos\phi})^2x)^{-1/2}d\phi\\&=
2\frac{1}{\sqrt{1+ a^2x}}\left(\frac{a^2x}{1+ a^2x}\right)^{-1/2}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)\\&=
2\frac{1}{\sqrt{1+ a^2x}}\frac {\sqrt{1+a^2x}}{a\sqrt x}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)\\&=
\frac{2}{a\sqrt x}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)
\end{aligned}$$
$$x_n=\frac{\Gamma(x+n)}{\Gamma(x)}$$
$$x_{n+1}=\frac{\Gamma(x+n+1)}{\Gamma(x)}=\frac{(x+n)\Gamma(x+n)}{\Gamma(x)}=(x+n)x_n$$
$$\frac{\Gamma(x-n)}{\Gamma(x)}=\frac{(-1)^n}{(1-x)_n}$$
$$\frac{\Gamma(x-n)(1-x)_n}{\Gamma(x)}=(-1)^n$$
$$\frac{x_n}{(x+1)_n}=\frac {\frac{\Gamma(x+n)}{\Gamma(x)}} {\frac{\Gamma(x+1+n)}{\Gamma(x+1)}}=\frac {\frac{\Gamma(x+n)}{\Gamma(x)}} {\frac{(x+n)\Gamma(x+n)}{(x+1)\Gamma(x)}}=\frac {x+1} {x+n}$$
$$\begin{aligned}
&\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}\int_{-\pi/2}^{\pi/2} (1+(\frac{a}{\cos\phi})^2x)^{-1/2}d\phi dx\\
&=\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}(\frac{2}{a\sqrt x}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)) dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right) dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\sum_{n=0}^{\infty}\frac{(1/2)_n(1)_n}{(3/2)_nn!}\left(\frac{-1}{a^2x}\right)^n dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\sum_{n=0}^{\infty}\frac 3 {(1+2n)}(-1)^na^{-2n}x^{-n} dx\\
&=\frac{6}{a}\sum_{n=0}^{\infty}\frac {(-1)^n} {a^{2n}(1+2n)}\int_{0}^{1}x^{\gamma-3/2-n}(1-x)^{3/2-\gamma-1} dx\\
&=\frac{6}{a}\sum_{n=0}^{\infty}\frac {(-1)^n} {a^{2n}(1+2n)}B(\gamma-n-1/2,3/2-\gamma)\\
&=\frac{6}{a}\sum_{n=0}^{\infty}\frac {(-1)^n} {a^{2n}(1+2n)}B((\gamma-1/2)+(-n),1-(\gamma-1/2))\\
&=\frac{6}{a}\sum_{n=0}^{\infty}\frac {(-1)^n} {a^{2n}(1+2n)}\frac{\pi}{-n\sin(\pi (\gamma-1/2))B(-n,\gamma-1/2)}\\
&=\frac{6\pi}{a\cos(\pi\gamma)}\sum_{n=0}^{\infty}\frac {(-1)^n} {a^{2n}n(1+2n)}\frac{\Gamma(\gamma-1/2-n)}{\Gamma(-n)\Gamma(\gamma-1/2)}\\
&=\frac{-6\pi}{a\cos(\pi\gamma)}\sum_{n=0}^{\infty}\frac {1} {a^{2n}n(1+2n)}\frac{\Gamma(\gamma-1/2-n)\Gamma(n+1)}{\Gamma(0)\Gamma(1)\Gamma(\gamma-1/2)}\\
&=\frac{-6\pi}{a\cos(\pi\gamma)\Gamma(0)}\sum_{n=0}^{\infty}\frac {1} {a^{2n}(1+2n)}\frac{\Gamma(\gamma-1/2-n)\Gamma(n)}{\Gamma(\gamma-1/2)}\\
\end{aligned}$$
$$\begin{aligned}
&\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}\int_{-\pi/2}^{\pi/2} (1+(\frac{a}{\cos\phi})^2x)^{-1/2}d\phi dx\\
&=\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}(\frac{2}{a\sqrt x}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)) dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right) dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\sum_{n=0}^{\infty}\frac{(1/2)_n(1)_n}{(3/2)_nn!}\left(\frac{-1}{a^2x}\right)^n dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\sum_{n=0}^{\infty}\frac 3 {(1+2n)}(-1)^na^{-2n}x^{-n} dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\sum_{n=0}^{\infty}\int_0^1t^{2n} dt(-1)^na^{-2n}x^{-n} dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\int_0^1\sum_{n=0}^{\infty}t^{2n} (-1)^na^{-2n}x^{-n}dt dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\int_0^1\sum_{n=0}^{\infty}\left(\frac{-t^2}{a^2u}\right)^ndt dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\int_0^1\frac{1}{1+\frac{t^2}{a^2x}}dt dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}a\sqrt x\int_0^1\frac{1}{1+\left(\frac{t}{a\sqrt x}\right)^2}d(\frac{t}{a\sqrt x}) dx\\
&=6\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}\arctan(\frac{1}{a\sqrt x}) dx\\
\end{aligned}$$$$\begin{aligned}
&\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}\int_{-\pi/2}^{\pi/2} (1+(\frac{a}{\cos\phi})^2x)^{-1/2}d\phi dx\\
&=\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}(\frac{2}{a\sqrt x}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)) dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right) dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\sum_{n=0}^{\infty}\frac{(1/2)_n(1)_n}{(3/2)_nn!}\left(\frac{-1}{a^2x}\right)^n dx\\
&=\frac{2}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\sum_{n=0}^{\infty}\frac 3 {(1+2n)}(-1)^na^{-2n}x^{-n} dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\sum_{n=0}^{\infty}\int_0^1t^{2n} dt(-1)^na^{-2n}x^{-n} dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\int_0^1\sum_{n=0}^{\infty}t^{2n} (-1)^na^{-2n}x^{-n}dt dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\int_0^1\sum_{n=0}^{\infty}\left(\frac{-t^2}{a^2u}\right)^ndt dx\\
&=\frac{6}{a}\int_{0}^{1}x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\int_0^1\frac{1}{1+\frac{t^2}{a^2x}}dt dx\\
&=\frac{6}{a}\int_{0}^{1}\int_0^1x^{\gamma-3/2}(1-x)^{3/2-\gamma-1}\frac{\frac{a^2x}{t^2}}{1+\frac{a^2x}{t^2}}dx dt\\
&=\frac{6}{a}\int_{0}^{1}\frac{a^2}{t^2}\int_0^1x^{\gamma-1/2}(1-x)^{3/2-\gamma-1}(1+\frac{a^2x}{t^2})^{-1}dx dt\\
&=\frac{6}{a}\int_{0}^{1}\frac{a^2}{t^2}\frac 1 {B(\gamma+1/2,3/2-\gamma)}F_{2,1}(1, \gamma+1/2, 2, -a^2/t^2) dt\\
&=\frac{6}{aB(\gamma+1/2,3/2-\gamma)}\int_{0}^{1}\frac{a^2}{t^2}F_{2,1}(1, \gamma+1/2, 2, -\frac{a^2}{t^2}) dt\\
&=\frac{-6}{B(\gamma+1/2,3/2-\gamma)}\int_{0}^{1}F_{2,1}(1, \gamma+1/2, 2, -\frac{a^2}{t^2}) d(\frac{a}{t})\\
&=\frac{6}{B(\gamma+1/2,3/2-\gamma)}\int_{a}^{\infty}F_{2,1}(1, \gamma+1/2, 2, -t^2) dt\\
\end{aligned}$$
$$\begin{aligned}
&\int F_{2,1}(1, \gamma+1/2, 2, -t^2) dt\\&=
\int\sum_{n=0}^{\infty} \frac{(1)_n(\gamma+1/2)_n}{(2)_nn!}(-1)^n t^{2n} dt\\&=
\sum_{n=0}^{\infty} \frac{(1)_n(\gamma+1/2)_n}{(2)_nn!}(-1)^n \frac{t^{2n+1}} {2n+1}\\&=
t\sum_{n=0}^{\infty} \frac{(1)_n(\gamma+1/2)_n}{(2)_nn!} \frac{(1/2)_n} {(3/2)_n}(-t^2)^{n}\\&=
tF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -t^2)
\end{aligned}$$
$$\begin{aligned}
&\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}\int_{-\pi/2}^{\pi/2} (1+(\frac{a}{\cos\phi})^2x)^{-1/2}d\phi dx\\
&=\frac{6}{B(\gamma+1/2,3/2-\gamma)}\int_{a}^{\infty}F_{2,1}(1, \gamma+1/2, 2, -t^2) dt\\
&=\frac{6}{B(\gamma+1/2,3/2-\gamma)}(
tF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -t^2)|_a^{\infty})\\
&=\frac{6}{B(\gamma+1/2,3/2-\gamma)}(
\lim_{t\to\infty}tF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -t^2) - aF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -a^2))\\
&=\frac{6}{B(\gamma+1/2,3/2-\gamma)}(\beta_4 - aF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -a^2))\\
&\beta_4=\lim_{t\to\infty}tF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -t^2)
\end{aligned}$$
$$
\begin{aligned}
&\int_{-\pi/2}^{\pi/2} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi=\\&
\frac{1}{B(\gamma, 3/2-\gamma)}\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}\int_{-\pi/2}^{\pi/2} (1+(\frac{a}{\cos\phi})^2x)^{-1/2}d\phi dx=\\&
\frac{1}{B(\gamma, 3/2-\gamma)}\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}(
2\frac{1}{\sqrt{1+ a^2x}}F_{2,1}\left(1/2,1/2,3/2,\frac{1}{1+ a^2x}\right)) dx=\\&
\frac{1}{B(\gamma, 3/2-\gamma)}\int_{0}^{1}x^{\gamma-1}(1-x)^{3/2-\gamma-1}(
\frac{2}{a\sqrt x}F_{2,1}\left(1/2,1,3/2,\frac{-1}{a^2x}\right)) dx=\\&
\frac{1}{B(\gamma, 3/2-\gamma)}\frac{6}{B(\gamma+1/2,3/2-\gamma)}\int_{a}^{\infty}F_{2,1}(1, \gamma+1/2, 2, -t^2) dt=\\&
\frac{1}{B(\gamma, 3/2-\gamma)}\frac{6}{B(\gamma+1/2,3/2-\gamma)}(\beta_4 - aF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -a^2))=\\&
\frac{6}{B(\gamma, 3/2-\gamma)B(\gamma+1/2,3/2-\gamma)}(\beta_4 - aF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -a^2))\\
&\beta_4=\lim_{t\to\infty}tF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -t^2)

\end{aligned}
$$
$$\begin{aligned}
\int_{-\pi/2}^{\pi/2}(1+r^2)^{1-\gamma}\Big|_{\frac{a}{\cos\phi}}^{\infty}d\phi &= \lim_{r\to\infty}(1+r^2)^{1-\gamma}-\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi\\
&= \beta_3-\int_{-\pi/2}^{\pi/2}(1+(\frac{a}{\cos\phi})^2)^{1-\gamma}d\phi\\
\beta_3&=\lim_{r\to\infty}(1+r^2)^{1-\gamma}
\end{aligned}$$

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
&=\frac{\beta_1+\beta_2}{a\pi}+ \frac {a^{2(1-\gamma)}\Gamma(\gamma-\frac 1 2)\Gamma(\frac 1 2)} {2\pi\Gamma(\gamma)}F_{2,1}(\gamma-1,\gamma-\frac 1 2,\gamma,-\frac 1 {a^2})\\
&- \frac{\gamma-1}{2\pi^2} \int_{-\pi/2}^{\pi/2} F_{2,1}(1/2,\gamma,3/2,-(\frac{a}{\cos\phi})^2))d\phi+ \frac {\beta_3} {2\pi}\\
&=\frac{\beta_1+\beta_2}{a\pi}+ \frac {a^{2(1-\gamma)}\Gamma(\gamma-\frac 1 2)\Gamma(\frac 1 2)} {2\pi\Gamma(\gamma)}F_{2,1}(\gamma-1,\gamma-\frac 1 2,\gamma,-\frac 1 {a^2})\\
&- \frac{\gamma-1}{2\pi^2} 
\frac{6}{B(\gamma, 3/2-\gamma)B(\gamma+1/2,3/2-\gamma)}(\beta_4 - aF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -a^2))+ \frac {\beta_3} {2\pi}
\\
\beta_1&=\lim_{r\to\infty}\frac{r}{(1+r^2)^{\gamma-1}}\\
\beta_2&=\frac{\gamma-1}{\pi}\lim_{r\to\infty} r F_{2,1}(1/2,\gamma,3/2,-r^2)\\
\beta_3&=\lim_{r\to\infty}(1+r^2)^{1-\gamma}\\
\beta_4&=\lim_{t\to\infty}tF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -t^2)
\end{aligned}$$
$$\begin{aligned}
\Lambda(\omega)&= \eta_1+ \eta_2F_{2,1}(\gamma-1,\gamma-\frac 1 2,\gamma,-\frac 1 {a^2}) + \eta_3aF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -a^2)\\
\eta_1&=\frac{1}{2a\pi^2}\lim_{r\to\infty}{2\pi r(1+r^2)^{1-\gamma}+a\pi(1+r^2)^{1-\gamma}+2(\gamma-1)rF_{2,1}(1/2,\gamma,3/2,-r^2)}- \eta_3({2a\pi^2})rF_{3,2}(1,1/2,\gamma+1/2,2,3/2, -r^2)\\
\eta_2&=\frac {a^{2(1-\gamma)}\Gamma(\gamma-\frac 1 2)\Gamma(\frac 1 2)} {2\pi\Gamma(\gamma)}\\
\eta_3&=\frac{\gamma-1}{2\pi^2} 
\frac{6}{B(\gamma, 3/2-\gamma)B(\gamma+1/2,3/2-\gamma)}
\end{aligned}$$
