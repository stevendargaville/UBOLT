# The DSA preconditioner (`DSAPrecon`)

Diffusion synthetic acceleration is the scattering half of UBOLT's preconditioner, the
second half of Dargaville et al., JCP 518 (2024) 113342, Section 3:

    G^-1 = P . D_diff^-1 . R

The composite preconditioner's other stages attack the hyperbolic part of the transport
operator (a shell for removal, PCAIR for streaming) and nothing attacks the scattering,
so an optically thick problem with a scattering ratio near 1 degrades. DSA fills that
hole. This page holds the derivations and the reasoning behind each choice; the public
header (`include/ubolt/dsa.hpp`) keeps only the contract, and the measured iteration
counts behind the choices are in `TODO.md` (the Phase 4 DSA notes and the Phase 6 DSA
items) and `docs/dev/testing.md`.

## Where it sits

- It is a preconditioner, so it is NOT an `OperatorTerm`: it contributes nothing to the
  operator being solved. `TransportSolver` takes one optionally and drives it from a
  PCShell at composite index 2. It is caller-owned: create it, hand the solver a pointer,
  destroy it afterwards.
- One diffusion Mat and one inner KSP, values-refilled per group by `set_group()`. That
  is deliberately not folded into `TransportSolver::refresh()`: the solver has no group
  context, so the driver calls `set_group()` where it points the terms at the group's
  cross sections. Per-group cached Mats/KSPs would be a local change inside `DSAPrecon`
  if a sweep ever revisited groups.
- A multiplicative `PCComposite` builds the residual it hands later stages from pmat,
  which has no scattering in it; DSA needs the amat's residual, which is why the solver
  sets `PCSetUseAmat` when (and only when) a DSA is present (`TODO.md`, Phase 4 DSA
  notes).

## R and P

- **R** takes the angular residual to its 0th moment per (cell, basis) node,
  `phi(c) = sum_a w_a r(c, a)`, with the BC rows left out of the sum. A BC row carries the
  boundary condition (the identity, or the identity minus its mirrored angle), not a
  neutron balance, so it must not feed the diffusion solve. That is also why R is its own
  kernel and not `UboltAngularIntegral`, which is a gemm over every row.
- **P** broadcasts the scalar correction back isotropically,
  `delta_psi(c, a) = delta_phi(c) / sum_weights`, writing zero on the BC rows (the BC
  contract: the assembled operator already holds the boundary condition there).
- The `/ sum_weights` is what makes the two consistent: A applied to an isotropic
  `psi = phi / sum_w` gives `sigma_a phi / sum_w` on each angle plus streaming, and R
  hands back `sigma_a phi` - exactly D_diff's removal part.
- Either vacuum treatment works. Under ghost-flux (the default) the inflow rows are
  ordinary rows, so R sums them and P corrects them; under Dirichlet-cell they are BC
  rows and both halves skip them.
- Where the diffusion matrix is assembled in weak (volume- or mass-weighted) form, R's
  moment is multiplied by the same weight inside the restriction kernel: V per node on
  the plex, the lumped mass `m_i` on CG, nothing on the structured backends.

## The structured operator (StructuredFD1D/2D/3D)

A cell-centred finite difference `-div(D grad phi) + sigma_a phi` on a dof-1 twin of the
backend's DMDA (same grid, same decomposition), with `D = 1/(3 sigma_t)` and
`sigma_a = sigma_t - sigma_s(g, g)`:

- Interior faces take the harmonic mean `2 D_c D_n / (D_c + D_n)`, the flux-continuous
  face value, so a material interface needs no special casing.
- Vacuum faces are Marshak (Robin): eliminating the face value from
  `J = D_c (phi_c - phi_f) / (h/2)` and `J = phi_f / 2` leaves an outgoing current
  proportional to `phi_c` alone, `phi_c / (h (2 + h / (2 D)))` per unit volume.
- Reflective faces are zero Neumann: no current, no contribution.
- The Marshak face sits ON the domain boundary, which is where the ghost-flux vacuum
  treatment puts the transport boundary too; under Dirichlet-cell the transport
  boundary is the boundary cell's centre instead, and the same face serves both (its
  coefficient is insensitive - see `docs/dev/testing.md`).
- `DMCreateMatrix` is used here, unlike for the transport matrix: the objection there is
  upwind-specific (a DMDA star preallocates every point it reaches, where upwinding
  touches one neighbour per axis), and the diffusion operator IS the 3/5/7-point star.
- Assembly is on the host through `MatSetValuesStencil`: the matrix is n_angles times
  smaller than the transport one and refilled once per group, and the stencil call
  already knows the DMDA's patch-lexicographic numbering and ghost columns. A device COO
  path would need slot-map machinery of its own to buy nothing.
- The structured `create`s need the concrete SN set (for the per-axis cosines behind the
  consistent D's half-range current), not just the `AngularQuadrature` base.
- Face naming trap: in 2D bottom/top are the Y faces; in 3D (PETSc's box convention)
  bottom/top are Z, and the Y faces are front/back. Each builder spells its faces out
  for that reason.

## The discretisation-consistent D

First-order upwind streaming adds a numerical diffusion to the transport, and in thick
cells it dwarfs the physical `1/(3 sigma_t)`, so a correction built on the physical D is
consistent with the PDE but not with the discretisation. For a linear flux in an infinite
pure scatterer the upwind face current is exactly `-(D + m h) grad phi`, with

    m = sum_{Omega.n > 0} w (Omega . n) / sum_weights

the quadrature's half-range current (1/4 in the continuum) and h the cell width across
the face: the upwind jump term `m [[phi]]` written as a diffusion. So every face
coefficient, the Marshak faces' included, blends it in:

    D_face = (D^p + (m h)^p)^(1/p),   p = -dsa_consistent_d_power, 1.5 by default

- It is a property of the face, so it goes on after the harmonic mean, not per cell.
- p = 1 is the exact sum, right in both limits but it over-diffuses intermediate cells
  (tau = sigma_t h ~ 0.3-2). A 1D Fourier analysis of step-differenced source iteration
  plus this DSA puts the optimal D below `D + m h` there, and p = 1.5 tracks that optimum
  to within a few hundredths of spectral radius at every tau and scattering ratio. p < 1
  would add more than the sum, which nothing supports, so it is refused.
- On the DG0 plex, m is per face (off the backend's own ordinates and the face normal)
  and h is the centroid-to-centroid distance through the face; on a box both are the
  structured values.
- DG1 ignores the option: its interior penalty already floors at the same 1/4. CG ignores
  it too: SUPG's thick limit is the physical D (below).
- `-dsa_consistent_d 0` is the physical D.

## The DG0 plex operator (UnstructuredDG, order 0)

The cell-centred finite VOLUME sibling of the structured operator:

- A two-point flux across each face, `T_f = A_f / (d_c / D_c + d_n / D_n)`, d the
  centroids' normal distances to the face: current continuity between the two
  centroids, a series resistance. On a uniform box `d_c = d_n = h/2` and it IS the
  structured harmonic stencil.
- Marshak on vacuum boundary faces, `A_f / (2 + d_c / D_c)`; nothing on reflective ones.
- The consistent-D blend is applied to the face's effective D over the
  centroid-to-centroid length: `T_f = A_f / len * Blend(T_f len / A_f, m len)`.
- Volumes vary, so the matrix is assembled VOLUME-WEIGHTED - `V_c` times the
  per-unit-volume row the structured backends write - which keeps it SPD, and R's moment
  is scaled by V to match. On a box V is constant, so the plex operator is exactly V
  times the structured one and its correction reproduces the structured one to rounding
  with an exact inner solve (`verify_plexk` checks both).
- No DMDA twin: the matrix is sized off the backend's owned cells, whose global order is
  the transport rows' (`CheckPlexLayout`), and preallocated from its face CSR; host
  values go into that COO pattern once per group.

## The DG1 operator (UnstructuredDG, order 1): MIP

At DG1 the diffusion unknown lives in the DG1 space itself: D_diff is the modified
interior penalty (MIP) form of Wang & Ragusa (NSE 166, 2010) on the backend's
orthonormal modal basis, one unknown per (cell, basis) node:

    a(u, v) = sum_c int_c D grad u . grad v + sigma_a u v
            + sum_{interior f} int_f kappa [[u]][[v]] - {{D du/dn}}[[v]] - [[u]]{{D dv/dn}}
            + sum_{vacuum f}   int_f kappa u v - 1/2 D (du/dn) v - 1/2 u D (dv/dn)

- `kappa = max(C/2 (D_c/h_c + D_n/h_n), 1/4)` on an interior face and
  `max(C D_c/h_c, 1/4)` on a vacuum one, with h twice the centroid's distance to the face
  (the cell width on a box, a fraction of the height on a simplex, which only errs
  towards more penalty) and `C = -dsa_mip_penalty`, 4 by default. Below C ~ 2 the form
  loses coercivity.
- The 1/4 is the upwind jump penalty of an isotropic angular flux, so the thick limit is
  the transport's, and on a vacuum face it is Marshak's `phi / 2` again. Reflective faces
  are natural (zero Neumann).
- The face terms read the backend's exact face matrices (`int_f phi_i phi_j`, both
  cells' bases) and each side's constant basis gradients. Row (c, i) is the weak form
  tested with `phi_i`: every face is visited from both cells and each visit writes only
  its own cell's rows. With F the face matrices times `V_c A_f`, `m_i = F_own(i, 0)`, and
  g, gn this cell's and the neighbour's `grad phi . n`, an interior face adds
  `kappa F_own(i, j) - 1/2 D_c g_j m_i - 1/2 D_c g_i F_own(0, j)` to the own block and
  `-kappa F_up(i, j) - 1/2 D_n gn_j m_i + 1/2 D_c g_i F_up(0, j)` to the neighbour's; the
  neighbour's visit writes the transpose, so the matrix is symmetric (the derivation is
  also next to the code, `src/dsa_plexk.kokkos.cxx`).
- The rows are volume-weighted like DG0's and the transport's row (c, i) is
  `(1/V) int_c phi_i` times the balance, so R sums EVERY node's ordinates and scales by
  V, and P corrects every node, the slopes included. The restricted residual and the
  correction are then the same weak moments the transport rows are - which is what the
  cell-average (DG0-on-DG1) operator this replaced lacked; it did nothing on the quad box.
- The matrix carries block size n_basis, so GAMG aggregates a cell's nodes together.

## The CG-SUPG operator (UnstructuredCG)

The diffusion unknown lives where the transport's does, one per VERTEX, and D_diff is the
continuous P1/Q1 weak form on the backend's own element tables:

    a(u, v) = sum_e int_e D_e grad u . grad v + sigma_a u v + sum_{vacuum f} m_f int_f u v

with the CONSISTENT mass M, the face mass lumped the way the transport's weak boundary
is, `m_f` the face's (incoming) half-range current and nothing on a reflective face.

This is the transport operator restricted onto an isotropic flux, `m R A P`, exactly
(`verify_cgk` checks it to rounding): every odd-in-Omega part of A cancels over the
symmetric quadrature (the Galerkin streaming and the `tau Omega . G^T` removal and scatter
terms, tau being even), the SUPG term leaves `(1/W) sum_a w_a tau_a Omega_a Omega_a^T : K`,
which is `1/(3 sigma_t) I` wherever `tau = 1/sigma_t`, the removal less the scatter gives
`sigma_a M`, and the weak inflow gives `m_f`. So:

- `D_e = 1/(3 sigma_t)` per element: the thick limit of SUPG IS the physical diffusion,
  there is no numerical diffusion to blend in. In a thin cell the physical D does better
  than the SUPG tensor, and Marshak's 1/2 in place of `m_f` does worse.
- The transport rows are divided by the lumped mass `m_i` and these rows are the weak
  form, so R's moment is scaled by `m_i`; P is the isotropic broadcast, per vertex; R and
  P are the identity in space.
- D_diff is SPD, and every element touching an owned vertex is local (FEM overlap), so
  each owned row is complete on its rank.
- The cross sections, and so the voids, are per local ELEMENT.

## Voids

`D = 1/(3 sigma_t)` is not defined where `sigma_t = 0`. A cell (an element on CG) whose
group `Sigma_t` is at or below `-dsa_void_sigma_t` (0 by default: only a true void) is a
void, per GROUP (a material can be a void in some groups only). With no void cell the
arithmetic is the plain operator's, bit for bit.

### Bridging (the default)

The void cells stay in the diffusion operator as ordinary cells with no absorption and a
FREE-FLIGHT diffusion coefficient. `D = 1/(3 sigma_t)` is the random walk's `<mu^2>` times
the flight length `1/sigma_t`; in a void the flight is ended by the void's walls instead,
and the mean flight through a region is its mean chord, Cauchy's `L = 4 V / S` (the
classical Behrens void correction for streaming cavities in diffusion theory). So a void
cell takes

    D = 1 / (3 (sigma_t + 1/L))

(Wigner's rational combination of the collision and the wall; exactly `L / 3` in a true
void), with V the voids' total volume and S the area of their faces that END a flight:
faces onto a non-void cell and vacuum boundary faces. A reflective face mirrors the flight
on, so it is not in S. `-dsa_void_d` fixes D instead. Everything else is the unvoided
operator's (harmonic face D, the consistent-D blend, R and P over the void's nodes), so the
correction couples the regions a void separates, which the mask cannot.

- L is one aggregate over every void in the group, not per connected void; the counts
  are forgiving of it (within a factor ~3 of L / 3 costs at most one iteration).
- At DG1, every face touching a bridged void takes the weighted interior penalty of Ern,
  Stephansen & Zunino (IMA J Numer Anal 29, 2009): both averages weighted so that
  `{{D grad u}}` uses the face's HARMONIC D, `D_h = 2 D_c D_n / (D_c + D_n)`, and kappa
  computed off D_h. With the void's D ~100x the material's, the plain arithmetic MIP
  penalty welds the material's surface to the void; D_h is the DG1 sibling of the
  harmonic face D the DG0 operators use, and it stays SPD (both cells compute the same
  D_h).
- Bridging needs something to bridge and must stay nonsingular: a group that is void
  EVERYWHERE, or one with no vacuum face and no absorption anywhere (the bridged operator
  is then pure Neumann), falls back to the mask. So does a group where no face ends a
  flight (S = 0).

### Masking (`-dsa_void_bridge 0`)

- A void cell's diffusion rows are the identity (times V on the plex, so the plex matrix
  is still V times the structured one), decoupled from every neighbour in both
  directions, so the matrix stays SPD.
- R writes zero there and P corrects nothing there: the void's transport rows are left to
  the removal + PCAIR stages, which is where streaming is handled anyway.
- A face between a real cell and a void is a MARSHAK face for the real cell - exactly the
  vacuum boundary face (the MIP vacuum form at DG1): what crosses into the void is treated
  as leaked, the zero-incoming-current condition. Not zero Neumann: that would leave a
  non-absorbing island surrounded by void singular, where Marshak never is. Under the
  consistent D it is the BLENDED vacuum face on the real cell's side, and the void flag
  is tested before any blend, so no `m h` coupling ever reaches a masked cell.
- What the mask does NOT do is couple the regions a void separates.

### CG's differences (deliberate)

- The voids are per element, counted over OWNED elements, D staged over all local ones.
- CG never masks: the vertex a void shares with the material is the material's too.
  Unbridged, a void element keeps the SUPG operator's own tensor
  `(1/W) sum_a w_a tau_a Omega_a Omega_a^T` - the only D defined there - computed on the
  device with `UboltSUPGTau`'s own arithmetic. It degrades as `h / zeta` shrinks with the
  mesh, which is why bridging is the default here too.
- With no mask to fall back on, CG's bridge needs only a void and a face that ends a
  flight, and its singular guard (a vacuum face or some absorption) runs before it.

### The staged D

D per cell is the one channel a neighbour - on this rank or another, through the ghost
exchange - sees of a cell, so the void state rides it (`StageD` in
`src/dsa_operatork.hpp`):

- `0`: a masked void (no unknown to couple to: a Marshak face for the neighbour);
- `< 0`: a bridged void, `|v|` its D (DG1 needs to see one, for the weighted penalty);
- `> 0`: a real cell, its D.

No real cell's D can be zero or negative, so the encoding is unambiguous. The chord pass
stages a plain void flag through the same vectors beforehand (a second vector would cost
a second exchange per bridged group); it is restaged as a D before the assembly reads it.

### The singular guard

With every face reflective the diffusion operator is pure Neumann, so only absorption
keeps it nonsingular - the same constraint the transport operator itself carries
(all-reflect with a scattering ratio of exactly 1 has the constants in its kernel, see
`docs/dev/testing.md`). A face into a MASKED void is a Marshak face, so any masked void
is as good as a vacuum face; bridging is only chosen where a vacuum face or some
absorption holds it. The guard counts absorption in real cells only, unless the voids are
bridged. Every collective in the policy sits under a condition built from globally
reduced or globally identical values only, so every rank takes the same branches.

## Code layout

- `include/ubolt/dsa.hpp`: `DSAPrecon`, the public contract.
- `src/dsak.kokkos.cxx`: what every backend shares - R and P (the device kernels), the
  inner KSP (`dsa_` prefix, default KSPPREONLY + PCGAMG: the paper's single BoomerAMG
  V-cycle, with no hypre in the CI images), the void policy (census, bridge decision, D
  per unit, the mask), and the generic `create(..., const Discretisation &, ...)` that
  dispatches to the per-backend overloads.
- `src/dsa_operatork.hpp` (internal): the `DSAOperator` interface each backend
  implements (matrix, work vectors, weight, the void volume/surface census, the per-group
  refill) and the shared helpers (half-range current, blend, staged-D encoding).
- `src/dsa_structuredk.kokkos.cxx`, `src/dsa_plexk.kokkos.cxx` (DG0 and DG1),
  `src/dsa_cgk.kokkos.cxx`: one operator each.

Per group the host assembly costs two cell-sized device-to-host copies of the cross
sections (plus the SUPG tensor on CG when a void is unbridged), and a host-to-device copy
of the mask only when the mask changes; the structured backends stage D through host
vectors, so their ghost exchange never touches the device. Per apply there are no
transfers and no allocations: R (one launch, the weight folded in), the inner solve, P.
