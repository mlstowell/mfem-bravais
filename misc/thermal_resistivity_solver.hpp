// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_THERMAL_RESISTIVITY_SOLVER
#define MFEM_THERMAL_RESISTIVITY_SOLVER

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

class ThermalResistivity
{
public:
   ThermalResistivity(MPI_Comm & comm, ParMesh & pmesh,
                      ParFiniteElementSpace & L2FESpace,
                      Coefficient & kCoef,
                      int dim = 1, int order = 1,
                      double a = 1.0);
   ~ThermalResistivity();

   void ConductivityChanged();

   void SetConductivityCoef(Coefficient & kCoef);

   double GetResistivity(Vector * dR = NULL);

   void Visualize(ParGridFunction * dR = NULL);

   //ParFiniteElementSpace * GetL2FESpace() { return L2FESpace_; }

private:
   void InitSecondaryObjects();

   // void UpdateAndRebalance();

   void Solve(ParGridFunction * w = NULL);

   void CalcSensitivity(Vector & dF);

   double CalcResistivity(Vector * dR = NULL);

   // double EstimateErrors();

   MPI_Comm * commPtr_;
   int        myid_;
   int        numProcs_;
   int        dim_;
   int        order_;
   // int        n_;
   double     a_;

   ParMesh  * pmesh_;

   ParFiniteElementSpace * H1FESpace_;
   ParFiniteElementSpace * L2FESpace_;

   Coefficient           * kCoef_;
   Coefficient           * rkCoef_;
   Coefficient           * rCoef_;
   // Coefficient           * rkCoef1_;
   ParBilinearForm       * ak_;
   ParLinearForm         * gk_;
   // ParLinearForm         * gk1_;

   ParGridFunction * t_;
   ParGridFunction * rhs_;
   ParGridFunction * k_;
   ParGridFunction * w_;

   HypreParMatrix * A_;

   mutable Vector T_;
   mutable Vector RHS_;

   Vector errors_;

   Array<int> ess_bdr_;
   Array<int> ess_bdr2_;
   Array<int> ess_tdof_list_;
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif //MFEM_THERMAL_RESISTIVITY_SOLVER
