# Intrinsic Hessian at the Optimistic Solution

We prove

$$
\operatorname{Hess}_{\mathcal S(\theta)}f(x^\star(\theta))
=\frac{1+0.3\theta}{1+\theta^2}I_k
\succeq0.01I_k,
\qquad \theta\in[-3,3].
$$

The identity uses the metric on \(\mathcal S(\theta)\) induced by the ambient
Euclidean space. Accordingly, \(I_k\) denotes the identity in any orthonormal
basis of \(T_{x^\star(\theta)}\mathcal S(\theta)\).

## 1. Restrict the Objective to the Manifold

Fix \(\theta\), abbreviate

$$
R=R(\theta)=\sqrt{1+\theta^2},
\qquad
b_\theta=1+0.3\theta,
$$

and write \(x=(Rv,0)\) with \(v\in\mathbb S^k\). On the minimizer manifold,
\(\lVert x\rVert=R\) and
\(u(\theta,x)=\langle v,a(\theta)\rangle\). Therefore

$$
\left.f(\theta,\cdot)\right|_{\mathcal S(\theta)}(Rv,0)
=2\sqrt{\epsilon^2+R^2}
+q_\theta\!\left(\langle v,a(\theta)\rangle\right).
$$

The first term is constant on the manifold.

## 2. Follow a Unit-Speed Geodesic

Let

$$
\eta=(\eta_y,0)\in T_{x^\star(\theta)}\mathcal S(\theta),
\qquad r=\lVert\eta\rVert.
$$

For \(r>0\), set \(w=\eta_y/r\). Since
\(x^\star(\theta)=(-Ra(\theta),0)\), we have
\(\langle w,a(\theta)\rangle=0\). The geodesic satisfying
\(x_\eta(0)=x^\star(\theta)\) and \(\dot x_\eta(0)=\eta\) is

$$
x_\eta(t)
=\left(
R\left[
-a(\theta)\cos\!\left(\frac{rt}{R}\right)
+w\sin\!\left(\frac{rt}{R}\right)
\right],0
\right).
$$

Consequently,

$$
u(\theta,x_\eta(t))
=-\cos\!\left(\frac{rt}{R}\right),
\qquad
u'(0)=0,
\qquad
u''(0)=\frac{r^2}{R^2}.
$$

The factor \(R^{-2}=(1+\theta^2)^{-1}\) appears because \(t\) measures
ambient arc length rather than angular displacement.

## 3. Compute the Riemannian Hessian

For the constants used in the experiment,

$$
q_\theta(u)
=b_\theta(u+1)
+\frac12\sin^2\!\left(2\pi(u+1)\right).
$$

The oscillatory term has zero first derivative at \(u=-1\), so
\(q_\theta'(-1)=b_\theta\). Using the second-derivative chain rule,

$$
\begin{aligned}
\left\langle
\eta,\operatorname{Hess}_{\mathcal S(\theta)}
f(x^\star(\theta))\,\eta
\right\rangle
&=
\left.\frac{d^2}{dt^2}f(\theta,x_\eta(t))\right|_{t=0}\\
&=
q_\theta''(-1)\bigl(u'(0)\bigr)^2
+q_\theta'(-1)u''(0)\\
&=
\frac{1+0.3\theta}{1+\theta^2}\lVert\eta\rVert^2.
\end{aligned}
$$

This holds for every tangent vector \(\eta\). Hence, in any orthonormal
tangent basis,

$$
\operatorname{Hess}_{\mathcal S(\theta)}f(x^\star(\theta))
=\frac{1+0.3\theta}{1+\theta^2}I_k.
$$

## 4. Prove the Uniform Lower Bound

For \(\theta\in[-3,3]\),

$$
\begin{aligned}
1+0.3\theta-0.01(1+\theta^2)
&=0.99+0.3\theta-0.01\theta^2\\
&\geq
\min\{0.99-0.9-0.09,\;0.99+0.9-0.09\}\\
&=0.
\end{aligned}
$$

The quadratic is concave, so its minimum over the interval is attained at an
endpoint. Dividing by \(1+\theta^2>0\) yields

$$
\frac{1+0.3\theta}{1+\theta^2}\geq0.01.
$$

Therefore

$$
\boxed{
\operatorname{Hess}_{\mathcal S(\theta)}f(x^\star(\theta))
\succeq0.01I_k
}
$$

with equality in the scalar lower bound at \(\theta=-3\).
