/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkGaussianKernelFunction_h
#define itkGaussianKernelFunction_h

#include "itkKernelFunctionBase.h"
#include "itkMath.h"
#include <cmath>

namespace itk
{
/** \class GaussianKernelFunction
 * \brief Gaussian kernel used for density estimation and nonparametric
 *  regression.
 *
 * This class encapsulates a Gaussian smoothing kernel for
 * density estimation or nonparametric regression.
 * See documentation for KernelFunctionBase for more details.
 *
 * \sa KernelFunctionBase
 *
 * \ingroup Functions
 * \ingroup ITKCommon
 */
template <typename TRealValueType = double>
class ITK_TEMPLATE_EXPORT GaussianKernelFunction : public KernelFunctionBase<TRealValueType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GaussianKernelFunction);

  /** Standard class type aliases. */
  using Self = GaussianKernelFunction;
  using Superclass = KernelFunctionBase<TRealValueType>;
  using Pointer = SmartPointer<Self>;

  using typename Superclass::RealType;
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(GaussianKernelFunction);

  /** Evaluate the function. */
  TRealValueType
  Evaluate(const TRealValueType & u) const override
  {
    constexpr TRealValueType negHalf{ -0.5 };
    return std::exp(negHalf * itk::Math::sqr(u)) * Math::one_over_sqrt2pi;
  }

protected:
  GaussianKernelFunction() {}
  ~GaussianKernelFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override
  {
    Superclass::PrintSelf(os, indent);
  }
};
} // end namespace itk

#endif
