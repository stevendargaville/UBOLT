#ifndef UBOLT_TYPES_HPP
#define UBOLT_TYPES_HPP

// All the Kokkos view typedefs used by libubolt live here
// See docs/dev/kokkos.md - don't declare ad-hoc Kokkos::View types elsewhere

#include <petscvec_kokkos.hpp>
#include <Kokkos_DualView.hpp>

using DefaultMemorySpace         = Kokkos::DefaultExecutionSpace::memory_space;
using PetscScalarKokkosView      = Kokkos::View<PetscScalar *, DefaultMemorySpace>;
using PetscScalar2DKokkosView    = Kokkos::View<PetscScalar **, DefaultMemorySpace>;
using PetscScalarConstKokkosView = Kokkos::View<const PetscScalar *, DefaultMemorySpace>;
// LayoutRight so a 1D Vec reshapes to (cells, angles) with angle contiguous
// (angle-fastest dof ordering)
using PetscScalar2DConstKokkosView = Kokkos::View<const PetscScalar **, Kokkos::LayoutRight, DefaultMemorySpace>;
using HostMirrorMemorySpace      = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;
using PetscScalarKokkosViewHost  = Kokkos::View<PetscScalar *, HostMirrorMemorySpace>;
using PetscScalarKokkosViewHostUnmanaged = Kokkos::View<PetscScalar *, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using PetscScalar2DKokkosViewHostUnmanaged = Kokkos::View<PetscScalar **, Kokkos::LayoutRight, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using KokkosTeamMemberType = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;

#endif
