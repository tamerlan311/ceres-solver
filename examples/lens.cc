// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

// Data generated using the following octave code.
//   randn('seed', 23497);
//   m = 0.3;
//   c = 0.1;
//   x=[0:0.075:5];
//   y = exp(m * x + c);
//   noise = randn(size(x)) * 0.2;
//   y_observed = y + noise;
//   data = [x', y_observed'];

const int kNumObservations = 67;

template <typename T>
T opencv_distortion (const T& Rd, const T& k1, const T& k2, const T& k3)
{
    T Rd2 = Rd * Rd ;
    T Rd4 = Rd2 * Rd2 ;
    //T Rd6 = Rd4 * Rd2 ;

    T Rd3 = Rd * Rd * Rd ;
    T Rd5 = Rd3 * Rd2 ;
    T Rd7 = Rd4 * Rd3 ;

    //T Ru = T(1.0) + k1 * Rd2 + k2 * Rd4 + k3 * Rd6 ;
    T Ru = T(1.0) * Rd + k1 * Rd3 + k2 * Rd5 + k3 * Rd7 ;

    return Ru ;
}

template <typename T>
T ptlens_distortion (T Rd, T a, T b, T c)
{
    T Rd2 = Rd * Rd ;
    T Rd4 = Rd2 * Rd2 ;

    /* const float poly3 = a * ru2 * r + b * ru2 + c * r + d; */
    T Ru = a * Rd4 + b * Rd2 * Rd + c * Rd2 + (T(1.0) - a - b - c) * Rd ;
    //T Ru = a * Rd2 * Rd + b * Rd2 + c * Rd + (T(1.0) - a - b - c) ;

    return Ru ;
}

struct ExponentialResidual {
  ExponentialResidual(double x, double y)
      : x_(x), y_(y) {}

  template <typename T> bool operator()(const T* const m,
                                        const T* const c,
                                        T* residual) const {
    residual[0] = T(y_) - exp(m[0] * T(x_) + c[0]);
    return true;
  }

 private:
  const double x_;
  const double y_;
};

struct OpenCVSolver {
    OpenCVSolver (double Rd, double Ru)
        : m_Rd (Rd), m_Ru (Ru)
    {}

    template <typename T> bool operator()(const T* k1,
                                          const T* k2,
                                          const T* k3,
                                          T* residual) const
    {
        residual[0] = /*T*/(m_Ru - opencv_distortion(T(m_Rd), k1[0], k2[0], k3[0])) ;

        return true ;
    }

private:
    double m_Rd ;
    double m_Ru ;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  const double a = 0.007226 ;
  const double b = -0.023605 ;
  const double c = 0.016384 ;

  double k1 = 0.0;
  double k2 = 0.0;
  double k3 = 0.0;

  Problem problem;

  double Rd_max = 0.21218 ;
  for (double Rd = 0.0; Rd < Rd_max; Rd+=0.001) {
    problem.AddResidualBlock(
        new AutoDiffCostFunction<OpenCVSolver, 1, 1, 1, 1>(
            new OpenCVSolver(Rd, ptlens_distortion(Rd, a, b, c))),
        NULL,
        &k1, &k2, &k3);
  }

  Solver::Options options;
  //options.max_num_iterations = 25;
  //options.linear_solver_type = ceres::DENSE_QR;
  options.min_relative_decrease = 1e-32 ;
  options.function_tolerance = 1e-32 ;
  options.gradient_tolerance = 1e-32 ;
  options.parameter_tolerance = 1e-32 ;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "Initial k1: " << 0.0 << " k2: " << 0.0 << " k3: " << 0.0 << "\n";
  std::cout << "Final   k1: " << k1  << " k2: " << k2  << " k3: " << k3  << "\n";
  return 0;
}
