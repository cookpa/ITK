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
/*=========================================================================
 *
 *  Portions of this file are subject to the VTK Toolkit Version 3 copyright.
 *
 *  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 *
 *  For complete copyright, license and disclaimer of warranty information
 *  please refer to the NOTICE file at the top of the ITK source tree.
 *
 *=========================================================================*/
#ifndef itkBSplineCenteredResampleImageFilterBase_h
#define itkBSplineCenteredResampleImageFilterBase_h

#include "itkBSplineResampleImageFilterBase.h"
#include "itkProgressReporter.h"
#include <vector>

namespace itk
{

/**
 * \class BSplineCenteredResampleImageFilterBase
 * \brief Evaluates the Centered B-Spline interpolation of an image.  Spline order may be from 0 to 5.
 *
 * This class defines N-Dimension CenteredB-Spline transformation.
 * It is based on \cite unser1999, \cite unser1993 and \cite unser1993a.
 * Code obtained from bigwww.epfl.ch by Philippe Thevenaz
 *
 * Limitations:  Spline order must be between 0 and 5.
 *               Spline order must be set before setting the image.
 *               Uses mirror boundary conditions.
 *               Requires the same order of Spline for each dimension.
 * \ingroup ITKImageGrid
 */
template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT BSplineCenteredResampleImageFilterBase
  : public BSplineResampleImageFilterBase<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineCenteredResampleImageFilterBase);

  /** Standard class type aliases. */
  using Self = BSplineCenteredResampleImageFilterBase;
  using Superclass = BSplineResampleImageFilterBase<TInputImage, TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(BSplineCenteredResampleImageFilterBase);

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** OutputImagePixelType type alias support */
  using typename Superclass::OutputImagePixelType;

  /** OutputImageIterator type alias support */
  using typename Superclass::OutputImageIterator;

protected:
  void
  InitializePyramidSplineFilter(int SplineOrder) override;

  void
  Reduce1DImage(const std::vector<double> & in,
                OutputImageIterator &       out,
                unsigned int                inTraverseSize,
                ProgressReporter &          progress) override;

  void
  Expand1DImage(const std::vector<double> & in,
                OutputImageIterator &       out,
                unsigned int                inTraverseSize,
                ProgressReporter &          progress) override;

protected:
  BSplineCenteredResampleImageFilterBase() = default;
  ~BSplineCenteredResampleImageFilterBase() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  using IndexValueType = typename TInputImage::IndexValueType;
  using SizeValueType = itk::SizeValueType;

  // implemented
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBSplineCenteredResampleImageFilterBase.hxx"
#endif

#endif
