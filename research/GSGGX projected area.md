$$
\begin{aligned}
\int_{S^2}\frac {|\omega\cdot m|} {(1 + \frac {m^TS^{-1}m - 1} {\gamma-1})^{\gamma}}dm&=\int_{S^2}\frac {|\omega \cdot {R^Tm}|} {(1 + \frac {m^TQ^{-1}m - 1} {\gamma-1})^{\gamma}}dm\\
&=\int_{S^2}\frac {|R\omega \cdot {m}|} {(1 + \frac {m^TQ^{-1}m - 1} {\gamma-1})^{\gamma}}dm\\
&=\int_{S^2}\frac {|m_z|} {(1 + \frac {m^TU^TQ^{-1}Um - 1} {\gamma-1})^{\gamma}}dm\\
&=2\int_{H^2}\frac {m_z} {(1 + \frac {m^TU^TQ^{-1}Um - 1} {\gamma-1})^{\gamma}}dm
\end{aligned}$$
$$S=R^TQR$$
$$U^TR\omega=U^Tv=e_z, m=Um'$$
$$n=\frac {a_z} {m_z}m$$
$$\frac {n} {a_z}=\frac m {m_z}=\frac m {m\cdot [0,0,1]}$$
$$
\begin{aligned}
\int_{S^2}\frac {|\omega\cdot m|} {(1 + \frac {m^TS^{-1}m - 1} {\gamma-1})^{\gamma}}dm&=\int_{S^2}\frac {|\omega \cdot {R^Tm}|} {(1 + \frac {m^TQ^{-1}m - 1} {\gamma-1})^{\gamma}}dm\\&=\int_{S^2}\frac {|\omega \cdot {R^Tm}|} {(1 + \frac {\frac {m_x^2} {a_x^2}+\frac {m_y^2} {a_y^2}+\frac {m_z^2} {a_z^2} - 1} {\gamma-1})^{\gamma}}dm
\\&=(\gamma-1)^\gamma\int_{S^2}\frac {|\omega \cdot {R^Tm}|} {(\gamma-2 + \frac {m_x^2} {a_x^2}+\frac {m_y^2} {a_y^2}+\frac {m_z^2} {a_z^2} )^{\gamma}}dm
\\&=(\frac{\gamma-1}{\gamma-2})^\gamma\int_{S^2}\frac {|\omega \cdot {R^Tm}|} {(1 + \frac{\frac {m_x^2} {a_x^2}+\frac {m_y^2} {a_y^2}+\frac {m_z^2} {a_z^2}} {\gamma-2})^{\gamma}}dm
\end{aligned}$$

$$
\begin{aligned}
\int_{S^2}\frac {|\omega \cdot {R^Tm}|} {(1 + \frac{\frac {m_x^2} {a_x^2}+\frac {m_y^2} {a_y^2}+\frac {m_z^2} {a_z^2}} {\gamma-2})^{\gamma}}dm&=
\int_{-\pi}^{\pi}\int_{0}^{\pi}\frac {|\omega \cdot {R^T[\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta]}|} {(1 + \frac{\frac {\sin^2\theta\cos^2\phi} {a_x^2}+\frac {\sin^2\theta\sin^2\phi} {a_y^2}+\frac {\cos^2\theta} {a_z^2}} {\gamma-2})^{\gamma}}d\phi d\theta
\end{aligned}$$
$$
\begin{aligned}
&\int_{H^2}\frac {m_z} {(1 + \frac {m^TU^TQ^{-1}Um - 1} {\gamma-1})^{\gamma}}dm\\
&=\int_{0}^{\pi}\int_{0}^{\pi}\frac {\cos\theta} {\left(1 + \frac {\sin^2(\theta-\theta_v)\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2(\theta-\theta_v)} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}\sin\theta d\theta d\phi\\

&=\int_{0}^{\pi}\int_{-\theta_v}^{\pi-\theta_v}\frac {\cos(\theta+\theta_v)} {\left(1 + \frac {\sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}\sin(\theta+\theta_v) d\theta d\phi\\

&=\frac 1 2\int_{0}^{\pi}\int_{-\theta_v}^{\pi-\theta_v}\frac {1} {\left(1 + \frac {\sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}d(\sin^2(\theta+\theta_v)) d\phi\\
\end{aligned}$$
$$
\begin{aligned}
&u=\sin^2\theta\\
&\sqrt u=\sin\theta\\
&\arcsin\sqrt u+\theta_v=\theta+\theta_v\\
&\sin(\arcsin\sqrt u+\theta_v)=\sin(\theta+\theta_v)\\
&\sin(\arcsin\sqrt u+\theta_v)=\sqrt u\cos\theta_v+\sin\theta_v\sqrt {1-u}\\
&\cos\theta_vd(\sqrt u)+\sin\theta_vd(\sqrt {1-u})=(\frac {\cos\theta_v} {2\sqrt u}-\frac {\sin\theta_v} {2\sqrt {1-u}})du\\
\end{aligned}$$
$$
\begin{aligned}
&\int_{-\theta_v}^{\pi-\theta_v}\frac {1} {\left(1 + \frac {\sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}d(\sin^2(\theta+\theta_v))\\

&=\int_{-\theta_v}^{0}\frac {1} {\left(1 + \frac {\sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}d(\sin^2(\theta+\theta_v)) \\

&+ \int_{0}^{\pi-\theta_v}\frac {1} {\left(1 + \frac {\sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}d(\sin^2(\theta+\theta_v))\\

&=\int_{\sin^2\theta_v}^{0}\frac {1} {\left(1 + \frac {u\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {1-u} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}(\frac {\cos\theta_v} {2\sqrt u}-\frac {\sin\theta_v} {2\sqrt {1-u}})du \\

&+ \int_{0}^{\sin^2\theta_v}\frac {1} {\left(1 + \frac {u\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {1-u} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}(\frac {\cos\theta_v} {2\sqrt u}-\frac {\sin\theta_v} {2\sqrt {1-u}})du\\
&=2\int^{\sin^2\theta_v}_{0}\frac {1} {\left(1 + \frac {u\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {1-u} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}(\frac {\cos\theta_v} {2\sqrt u}-\frac {\sin\theta_v} {2\sqrt {1-u}})du \\
&=2\int^{\sin^2\theta_v}_{0}\frac {1} {\left(1 + \frac {u\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {1-u} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}\frac {\cos\theta_v} {2\sqrt u}du\\
&-2\int^{\sin^2\theta_v}_{0}\frac {1} {\left(1 + \frac {u\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {1-u} {a_z^2} - 1} {\gamma-1}\right)^{\gamma}}\frac {\sin\theta_v} {2\sqrt {1-u}}du \\
\end{aligned}$$
$$
\begin{aligned}
&\int^{\sin^2\theta_v}_{0}\frac {1} {\left(\gamma-2 + ub+\frac {1-u} {a_z^2}\right)^{\gamma}\sqrt u}du\\
&=\int^{\sin^2\theta_v}_{0}\frac {1} {\left(\gamma-2 + ub+\frac {1} {a_z^2}-\frac {u} {a_z^2}\right)^{\gamma}\sqrt u}du\\
&=(\gamma-2+\frac {1} {a_z^2})^{-\gamma}\int^{\sin^2\theta_v}_{0}\frac {1} {\left(1 + u\frac {(b-\frac {1} {a_z^2})}{\gamma-2+\frac {1} {a_z^2}}\right)^{\gamma}\sqrt u}du\\
\end{aligned}$$
$$
\begin{aligned}
&\int^{a}_{0}\frac {1} {\left(1 + uc\right)^{\gamma}\sqrt u}du\\
\end{aligned}$$
$$\begin{aligned}
&\sin^2\theta\left(\frac {\cos^2\phi} {a_x^2}+\frac {\sin^2\phi} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\\
&=(1-\cos^2\theta)\left(\frac {\cos^2\phi} {a_x^2}+\frac {1-\cos^2\phi} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\\
&=\frac {\cos^2\phi} {a_x^2}+\frac {1-\cos^2\phi} {a_y^2}-\cos^2\theta\left(\frac {\cos^2\phi} {a_x^2}+\frac {1-\cos^2\phi} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\\
&=\frac 1 {a_y^2}+\frac {\cos^2\phi} {a_x^2}-\frac {\cos^2\phi} {a_y^2}-\frac {\cos^2\theta\cos^2\phi} {a_x^2}-\frac {\cos^2\theta(1-\cos^2\phi)} {a_y^2}+\frac {\cos^2\theta} {a_z^2}\\
&=\frac 1 {a_y^2}+\cos^2\phi(\frac {1} {a_x^2}-\frac {1} {a_y^2})-\frac {\cos^2\theta\cos^2\phi} {a_x^2}-\frac {\cos^2\theta} {a_y^2}+\frac {\cos^2\theta\cos^2\phi} {a_y^2}+\frac {\cos^2\theta} {a_z^2}\\
&=\frac 1 {a_y^2}+\cos^2\phi(\frac {1} {a_x^2}-\frac {1} {a_y^2})+\cos^2\theta(\frac {1} {a_z^2}-\frac {1} {a_y^2})+\cos^2\theta\cos^2\phi(\frac {1} {a_y^2}-\frac {1} {a_x^2})\\
\end{aligned}$$
$$\begin{aligned}
&\sin^2\theta\left(\frac {\cos^2\phi} {a_x^2}+\frac {\sin^2\phi} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\\
&=\sin^2\theta\left(\frac {\cos^2\phi} {a_x^2}+\frac {\sin^2\phi} {a_y^2}\right)+\frac {1-\sin^2\theta} {a_z^2}\\
&=\frac {1} {a_z^2}-\frac {\sin^2\theta} {a_z^2}+\sin^2\theta\left(\frac {\cos^2\phi} {a_x^2}+\frac {\sin^2\phi} {a_y^2}\right)\\
&=\frac {1} {a_z^2}+\sin^2\theta\left(\frac {\cos^2\phi} {a_x^2}+\frac {\sin^2\phi} {a_y^2}-\frac {1} {a_z^2}\right)\\
\end{aligned}$$
$$\begin{aligned}
&\sin^2(\theta+\theta_v)=(\sin\theta\cos\theta_v+\sin\theta_v\cos\theta)^2\\
&=\sin^2\theta\cos^2\theta_v+\sin^2\theta_v\cos^2\theta+2\sin\theta\cos\theta_v\sin\theta_v\cos\theta\\
&=\sin^2\theta\cos^2\theta_v+\sin^2\theta_v(1-\sin^2\theta)+\sin\theta\cos\theta\sin2\theta_v\\
&=\sin^2\theta\cos^2\theta_v-\sin^2\theta\sin^2\theta_v+\sin^2\theta_v+\sin\theta\cos\theta\sin2\theta_v\\
&=\sin^2\theta(\cos^2\theta_v-\sin^2\theta_v)+\sin^2\theta_v+\sin\theta\cos\theta\sin2\theta_v\\
&=\sin^2\theta\cos2\theta_v+\sin^2\theta_v+\sin\theta\cos\theta\sin2\theta_v\\
\end{aligned}$$
$$
\begin{aligned}
&\int_{H^2}\frac {m_z} {(1 + \frac {m^TU^TQ^{-1}Um - 1} {\gamma-1})^{\gamma}}dm\\
&=(\gamma-1)^\gamma\int_{0}^{\pi}\int_{0}^{\pi}\frac {\cos\theta} {\left(\gamma-2 + \sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\right)^{\gamma}}\sin\theta d\theta d\phi\\

&=(\gamma-1)^\gamma\int_{0}^{\pi}\int_{-\theta_v}^{\pi-\theta_v}\frac {\cos(\theta+\theta_v)} {\left(\gamma-2 + \sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\right)^{\gamma}}\sin(\theta+\theta_v) d\theta d\phi\\

&=\frac {(\gamma-1)^\gamma} 2\int_{0}^{\pi}\int_{-\theta_v}^{\pi-\theta_v}\frac {1} {\left(\gamma-2 + \sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\right)^{\gamma}}\sin(\theta+\theta_v)d(\sin(\theta+\theta_v)) d\phi\\
\end{aligned}$$
$$
\begin{aligned}
&u=\sin\theta\\
&\arcsin u+\theta_v=\theta+\theta_v\\
&\sin(\arcsin u+\theta_v)=\sin(\theta+\theta_v)\\
&=u\cos\theta_v+\sin\theta_v(1-u)\\
&=u\cos\theta_v-\sin\theta_vu+\sin\theta_v\\
&=u(\cos\theta_v-\sin\theta_v)+\sin\theta_v\\
&=up+q\\
\end{aligned}$$
$$b=\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}$$
$$
\begin{aligned}
&\int_{-\theta_v}^{\pi-\theta_v}\frac {1} {\left(\gamma-2 + \sin^2\theta b+\frac {\cos^2\theta} {a_z^2}\right)^{\gamma}}\sin(\theta+\theta_v)d(\sin(\theta+\theta_v)) d\phi\\

&=p\int_{-\sin\theta_v}^{\sin\theta_v}\frac {1} {\left(\gamma-2 + u^2b+\frac {1-u^2} {a_z^2}\right)^{\gamma}}(up+q)du\\

&=p\int_{-\sin\theta_v}^{\sin\theta_v}\frac {1} {\left(\gamma-2+\frac {1} {a_z^2} + u^2(b-\frac {1} {a_z^2})\right)^{\gamma}}(up+q)du\\

&=p(\gamma-2+\frac {1} {a_z^2})^{-\gamma}\int_{-\sin\theta_v}^{\sin\theta_v}\frac {1} {\left(1 + u^2\frac {b-\frac {1} {a_z^2}}{\gamma-2+\frac {1} {a_z^2}}\right)^{\gamma}}(up+q)du\\

&=c\int_{-\sin\theta_v}^{\sin\theta_v}\frac {1} {\left(1 + u^2h\right)^{\gamma}}(up+q)du\\

&=cp\int_{-\sin\theta_v}^{\sin\theta_v}\frac {u} {\left(1 + u^2h\right)^{\gamma}}du+cq\int_{-\sin\theta_v}^{\sin\theta_v}\frac {1} {\left(1 + u^2h\right)^{\gamma}}du\\
\end{aligned}$$
$$h=\frac {\frac {a_z^2\cos^2(\phi-\phi_v)} {a_x^2}+\frac {a_z^2\sin^2(\phi-\phi_v)} {a_y^2}-1}{a_z^2(\gamma-2)+1}$$
$$
\begin{aligned}
&\int_{-\sin\theta_v}^{\sin\theta_v}\frac {u} {\left(1 + u^2h\right)^{\gamma}}du\\
&=\frac 1 2\int_{-\sin\theta_v}^{\sin\theta_v}\frac {1} {\left(1 + u^2h\right)^{\gamma}}d(u^2)\\
&=\int_{0}^{\sin\theta_v}\frac {1} {\left(1 + u^2h\right)^{\gamma}}d(u^2)\\
&=\int_{0}^{\sin^2\theta_v}\frac {1} {\left(1 + uh\right)^{\gamma}}du\\
&=\frac 1 h \int_{1}^{1+h\sin^2\theta_v}u^{-\gamma}du\\
&=\frac 1 h (\frac {(1+h\sin^2\theta_v)^{1-\gamma}}{1-\gamma}-\frac {1}{1-\gamma})\\
&=\frac {(1+h\sin^2\theta_v)^{1-\gamma}}{h(1-\gamma)}-\frac {1}{h(1-\gamma)}\\
&=\frac {(1+h\sin^2\theta_v)^{1-\gamma}-1}{h(1-\gamma)}\\
\end{aligned}$$
$$\int_{-\sin\theta_v}^{\sin\theta_v}\frac {1} {\left(1 + u^2h\right)^{\gamma}}du=2\sin\theta_vF_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v)$$
$$
\begin{aligned}
&\int_{-\theta_v}^{\pi-\theta_v}\frac {1} {\left(\gamma-2 + \sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\right)^{\gamma}}\sin(\theta+\theta_v)d(\sin(\theta+\theta_v))\\
&=cp\frac {(1+h\sin^2\theta_v)^{1-\gamma}-1}{h(1-\gamma)}+2cq\sin\theta_vF_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v)\\
&=p^2(\gamma-2+\frac {1} {a_z^2})^{-\gamma}\frac {(1+h\sin^2\theta_v)^{1-\gamma}-1}{h(1-\gamma)}+2p(\gamma-2+\frac {1} {a_z^2})^{-\gamma}\sin^2\theta_vF_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v)\\
&=\frac {(\cos\theta_v-\sin\theta_v)^2(\gamma-2+\frac {1} {a_z^2})^{-\gamma}}{h(1-\gamma)}(1+h\sin^2\theta_v)^{1-\gamma}-\frac {(\cos\theta_v-\sin\theta_v)^2(\gamma-2+\frac {1} {a_z^2})^{-\gamma}}{h(1-\gamma)}\\&+2(\cos\theta_v-\sin\theta_v)(\gamma-2+\frac {1} {a_z^2})^{-\gamma}\sin^2\theta_vF_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v)\\
&=b(1+h\sin^2\theta_v)^{1-\gamma}-b+cF_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v)\\
\end{aligned}$$
$$
\begin{aligned}
&\int_{0}^{\pi}\int_{-\theta_v}^{\pi-\theta_v}\frac {1} {\left(\gamma-2 + \sin^2\theta\left(\frac {\cos^2(\phi-\phi_v)} {a_x^2}+\frac {\sin^2(\phi-\phi_v)} {a_y^2}\right)+\frac {\cos^2\theta} {a_z^2}\right)^{\gamma}}\sin(\theta+\theta_v)d(\sin(\theta+\theta_v)) d\phi\\
&=\int_{0}^{\pi} b(1+h\sin^2\theta_v)^{1-\gamma}-b+cF_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v) d\phi\\
&=b\int_{0}^{\pi} (1+h\sin^2\theta_v)^{1-\gamma} d\phi+c\int_{0}^{\pi} F_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v) d\phi-b\\
&=b\int_{0}^{\pi} (1+h\sin^2\theta_v)^{1-\gamma} d\phi+c\int_{0}^{\pi} F_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v) d\phi-b\\
\end{aligned}$$

$$\begin{aligned}
&\int_{0}^{\pi} (1+h\sin^2\theta_v)^{1-\gamma} d\phi\\
&=\int_{0}^{\pi} (1+\frac {\frac {a_z^2\cos^2\phi} {a_x^2}+\frac {a_z^2\sin^2\phi} {a_y^2}-1}{a_z^2(\gamma-2)+1}\sin^2\theta_v)^{1-\gamma} d\phi\\
&=\int_{0}^{\pi} (1+\frac {\frac {a_z^2} {a_x^2}-(\frac {a_z^2} {a_x^2}-\frac {a_z^2} {a_y^2})\sin^2\phi-1}{a_z^2(\gamma-2)+1}\sin^2\theta_v)^{1-\gamma} d\phi\\
&=\int_{0}^{\pi} (1+\frac {(a_z^2-a_x^2)\sin^2\theta_v}{a_x^2a_z^2(\gamma-2)+a_x^2}-\frac {(\frac {a_z^2} {a_x^2}-\frac {a_z^2} {a_y^2})\sin^2\theta_v}{a_z^2(\gamma-2)+1}\sin^2\phi)^{1-\gamma} d\phi\\
&=(1+\frac {(a_z^2-a_x^2)\sin^2\theta_v}{a_x^2a_z^2(\gamma-2)+a_x^2})^{1-\gamma}\int_{0}^{\pi} (1-\frac{a_z^2(a_y^2-a_x^2)\sin^2\theta_v}{a_y^2a_x^2a_z^2(\gamma-2)+a_y^2a_x^2+a_y^2(a_z^2-a_x^2)\sin^2\theta_v}\sin^2\phi)^{1-\gamma} d\phi\\
&=2(1+\frac {(a_z^2-a_x^2)\sin^2\theta_v}{a_x^2a_z^2(\gamma-2)+a_x^2})^{1-\gamma}\int_{0}^{\pi/2} (1-d\sin^2\phi)^{1-\gamma} d\phi\\
&=(1+\frac {(a_z^2-a_x^2)\sin^2\theta_v}{a_x^2a_z^2(\gamma-2)+a_x^2})^{1-\gamma}\pi F_{2,1}(\frac 1 2,\gamma-1,1,d)\\
&=(1+\frac {(a_z^2-a_x^2)\sin^2\theta_v}{a_x^2a_z^2(\gamma-2)+a_x^2})^{1-\gamma}\pi F_{2,1}(\frac 1 2,\gamma-1,1,\frac{a_z^2(a_y^2-a_x^2)\sin^2\theta_v}{a_y^2a_x^2a_z^2(\gamma-2)+a_y^2a_x^2+a_y^2(a_z^2-a_x^2)\sin^2\theta_v})\\
\end{aligned}$$
$$\begin{aligned}
&\int_{0}^{\pi} F_{2,1}(\frac 1 2, \gamma,\frac 3 2,-h\sin^2\theta_v) d\phi\\
&=2\int_{0}^{\pi/2} F_{2,1}(\frac 1 2, \gamma,\frac 3 2,-\frac {\frac {a_z^2\cos^2\phi} {a_x^2}+\frac {a_z^2\sin^2\phi} {a_y^2}-1}{a_z^2(\gamma-2)+1}\sin^2\theta_v) d\phi\\
&=2\int_{0}^{\pi/2} F_{2,1}(\frac 1 2, \gamma,\frac 3 2,-
\frac {(a_z^2-a_x^2)\sin^2\theta_v}{a_x^2a_z^2(\gamma-2)+a_x^2}+\frac {(\frac {a_z^2} {a_x^2}-\frac {a_z^2} {a_y^2})\sin^2\theta_v}{a_z^2(\gamma-2)+1}\sin^2\phi) d\phi\\
&=2\int_{0}^{\pi/2} F_{2,1}(\frac 1 2, \gamma,\frac 3 2,a\sin^2\phi-b) d\phi\\
\end{aligned}$$

Try different order of integration.