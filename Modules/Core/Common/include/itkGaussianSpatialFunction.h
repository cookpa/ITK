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
#ifndef itkGaussianSpatialFunction_h
#define itkGaussianSpatialFunction_h

#include "itkSpatialFunction.h"
#include "itkFixedArray.h"
#include "itkFloatTypes.h"

namespace itk
{
/** \class GaussianSpatialFunction
 * \brief N-dimensional Gaussian spatial function class
 *
 * GaussianSpatialFunction implements a standard Gaussian curve in N-d.
 * m_Normalized determines whether or not the Gaussian is normalized
 * (whether or not the sum over infinite space is 1.0)
 *
 * m_Scale scales the output of the Gaussian to span a range
 * larger than 0->1, and is often set to the maximum value
 * of the output data type (for instance, 255 for uchars)
 *
 * \ingroup SpatialFunctions
 * \ingroup ITKCommon
 */
template <typename TOutput = double,
          unsigned int VImageDimension = 3,
          typename TInput = Point<SpacePrecisionType, VImageDimension>>
class ITK_TEMPLATE_EXPORT GaussianSpatialFunction : public SpatialFunction<TOutput, VImageDimension, TInput>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GaussianSpatialFunction);

  /** Standard class type aliases. */
  using Self = GaussianSpatialFunction;
  using Superclass = SpatialFunction<TOutput, VImageDimension, TInput>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(GaussianSpatialFunction);

  /** Input type for the function. */
  using typename Superclass::InputType;

  /** Output type for the function. */
  using typename Superclass::OutputType;

  /** Type used to store gaussian parameters. */
  using ArrayType = FixedArray<double, VImageDimension>;

  /** Evaluate the function at a given position. */
  OutputType
  Evaluate(const TInput & position) const override;

  /** Set/Get the scale factor to multiply the true value of the Gaussian. */
  /** @ITKStartGrouping */
  itkSetMacro(Scale, double);
  itkGetConstMacro(Scale, double);
  /** @ITKEndGrouping */
  /** Set/Get whether or not to normalize the Gaussian. Default is false. */
  /** @ITKStartGrouping */
  itkSetMacro(Normalized, bool);
  itkGetConstMacro(Normalized, bool);
  itkBooleanMacro(Normalized);
  /** @ITKEndGrouping */
  /** Set/Get the standard deviation in each direction. */
  /** @ITKStartGrouping */
  itkSetMacro(Sigma, ArrayType);
  itkGetConstMacro(Sigma, ArrayType);
  /** @ITKEndGrouping */
  /** Set/Get the mean in each direction. */
  /** @ITKStartGrouping */
  itkSetMacro(Mean, ArrayType);
  itkGetConstMacro(Mean, ArrayType);
  /** @ITKEndGrouping */
protected:
  GaussianSpatialFunction() = default;
  ~GaussianSpatialFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  ArrayType m_Sigma{ ArrayType::Filled(5.0) };

  ArrayType m_Mean{ ArrayType::Filled(10.0) };

  double m_Scale{ 1.0 };

  bool m_Normalized{ false };
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGaussianSpatialFunction.hxx"
#endif

#endif
