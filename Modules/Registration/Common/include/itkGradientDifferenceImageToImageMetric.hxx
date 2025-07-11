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
#ifndef itkGradientDifferenceImageToImageMetric_hxx
#define itkGradientDifferenceImageToImageMetric_hxx

#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkMath.h"

#include <iostream>
#include <iomanip>
#include <cstdio>

#include "itkSimpleFilterWatcher.h"
namespace itk
{

template <typename TFixedImage, typename TMovingImage>
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GradientDifferenceImageToImageMetric()
  : m_TransformMovingImageFilter(nullptr)
{
  for (unsigned int iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    m_MinFixedGradient[iDimension] = 0;
    m_MaxFixedGradient[iDimension] = 0;

    m_Variance[iDimension] = 0;
  }

  for (unsigned int iDimension = 0; iDimension < MovedImageDimension; ++iDimension)
  {
    m_MinMovedGradient[iDimension] = 0;
    m_MaxMovedGradient[iDimension] = 0;
  }

  this->m_DerivativeDelta = 0.001;
}

template <typename TFixedImage, typename TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  if (!this->GetComputeGradient())
  {
    itkExceptionMacro("Gradients must be calculated");
  }

  // Initialise the base class

  Superclass::Initialize();

  // Create the filter to resample the moving image

  m_TransformMovingImageFilter = TransformMovingImageFilterType::New();

  m_TransformMovingImageFilter->SetTransform(this->m_Transform);
  m_TransformMovingImageFilter->SetInterpolator(this->m_Interpolator);
  m_TransformMovingImageFilter->SetInput(this->m_MovingImage);

  m_TransformMovingImageFilter->SetDefaultPixelValue(0);

  m_TransformMovingImageFilter->SetSize(this->m_FixedImage->GetLargestPossibleRegion().GetSize());
  m_TransformMovingImageFilter->SetOutputOrigin(this->m_FixedImage->GetOrigin());
  m_TransformMovingImageFilter->SetOutputSpacing(this->m_FixedImage->GetSpacing());
  m_TransformMovingImageFilter->SetOutputDirection(this->m_FixedImage->GetDirection());

  // Compute the image gradients
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Compute the gradient of the fixed image

  m_CastFixedImageFilter = CastFixedImageFilterType::New();
  m_CastFixedImageFilter->SetInput(this->m_FixedImage);

  // Index of Sobel filters for each dimension
  for (unsigned int iFilter = 0; iFilter < FixedImageDimension; ++iFilter)
  {
    m_FixedSobelOperators[iFilter].SetDirection(iFilter);
    m_FixedSobelOperators[iFilter].CreateDirectional();

    m_FixedSobelFilters[iFilter] = FixedSobelFilter::New();

    m_FixedSobelFilters[iFilter]->OverrideBoundaryCondition(&m_FixedBoundCond);
    m_FixedSobelFilters[iFilter]->SetOperator(m_FixedSobelOperators[iFilter]);

    m_FixedSobelFilters[iFilter]->SetInput(m_CastFixedImageFilter->GetOutput());

    m_FixedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
  }

  ComputeVariance();

  // Compute the gradient of the transformed moving image

  m_CastMovedImageFilter = CastMovedImageFilterType::New();
  m_CastMovedImageFilter->SetInput(m_TransformMovingImageFilter->GetOutput());

  for (unsigned int iFilter = 0; iFilter < MovedImageDimension; ++iFilter)
  {
    m_MovedSobelOperators[iFilter].SetDirection(iFilter);
    m_MovedSobelOperators[iFilter].CreateDirectional();

    m_MovedSobelFilters[iFilter] = MovedSobelFilter::New();

    m_MovedSobelFilters[iFilter]->OverrideBoundaryCondition(&m_MovedBoundCond);
    m_MovedSobelFilters[iFilter]->SetOperator(m_MovedSobelOperators[iFilter]);

    m_MovedSobelFilters[iFilter]->SetInput(m_CastMovedImageFilter->GetOutput());

    m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
  }
}

template <typename TFixedImage, typename TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Variance: " << m_Variance << std::endl;

  os << indent << "MinMovedGradient: " << m_MinMovedGradient << std::endl;
  os << indent << "MaxMovedGradient: " << m_MaxMovedGradient << std::endl;
  os << indent << "MinFixedGradient: " << m_MinFixedGradient << std::endl;
  os << indent << "MaxFixedGradient: " << m_MaxFixedGradient << std::endl;

  itkPrintSelfObjectMacro(TransformMovingImageFilter);

  itkPrintSelfObjectMacro(CastFixedImageFilter);

  os << indent << "FixedSobelOperators: ";
  for (const auto & elem : m_FixedSobelOperators)
  {
    os << indent.GetNextIndent() << elem << std::endl;
  }

  os << indent << "FixedSobelFilters: ";
  if (m_FixedSobelFilters[0] != nullptr)
  {
    for (const auto & elem : m_FixedSobelFilters)
    {
      os << indent.GetNextIndent() << *elem << std::endl;
    }
  }
  else
  {
    os << "(empty)" << std::endl;
  }

  m_MovedBoundCond.Print(os, indent);
  m_FixedBoundCond.Print(os, indent);

  itkPrintSelfObjectMacro(CastMovedImageFilter);

  os << indent << "MovedSobelOperators: " << m_MovedSobelOperators << std::endl;

  os << indent << "DerivativeDelta: " << m_DerivativeDelta << std::endl;
}

template <typename TFixedImage, typename TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::ComputeMovedGradientRange() const
{
  unsigned int           iDimension = 0;
  MovedGradientPixelType gradient;

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    using IteratorType = itk::ImageRegionConstIteratorWithIndex<MovedGradientImageType>;

    IteratorType iterate(m_MovedSobelFilters[iDimension]->GetOutput(), this->GetFixedImageRegion());

    gradient = iterate.Get();

    m_MinMovedGradient[iDimension] = gradient;
    m_MaxMovedGradient[iDimension] = gradient;

    while (!iterate.IsAtEnd())
    {
      gradient = iterate.Get();

      if (gradient > m_MaxMovedGradient[iDimension])
      {
        m_MaxMovedGradient[iDimension] = gradient;
      }

      if (gradient < m_MinMovedGradient[iDimension])
      {
        m_MinMovedGradient[iDimension] = gradient;
      }

      ++iterate;
    }
  }
}

template <typename TFixedImage, typename TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::ComputeVariance() const
{
  unsigned int           iDimension = 0;
  SizeValueType          nPixels = 0;
  FixedGradientPixelType mean[FixedImageDimension];
  FixedGradientPixelType gradient;

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    using IteratorType = itk::ImageRegionConstIteratorWithIndex<FixedGradientImageType>;

    IteratorType iterate(m_FixedSobelFilters[iDimension]->GetOutput(), this->GetFixedImageRegion());

    // Calculate the mean gradients

    nPixels = 0;

    gradient = iterate.Get();

    mean[iDimension] = 0;

    m_MinMovedGradient[iDimension] = gradient;
    m_MaxMovedGradient[iDimension] = gradient;

    while (!iterate.IsAtEnd())
    {
      gradient = iterate.Get();
      mean[iDimension] += gradient;

      if (gradient > m_MaxFixedGradient[iDimension])
      {
        m_MaxFixedGradient[iDimension] = gradient;
      }

      if (gradient < m_MinFixedGradient[iDimension])
      {
        m_MinFixedGradient[iDimension] = gradient;
      }

      ++nPixels;
      ++iterate;
    }

    if (nPixels > 0)
    {
      mean[iDimension] /= nPixels;
    }

    // Calculate the variance

    iterate.GoToBegin();

    m_Variance[iDimension] = 0;

    while (!iterate.IsAtEnd())
    {
      gradient = iterate.Get();
      gradient -= mean[iDimension];

      m_Variance[iDimension] += gradient * gradient;

      ++iterate;
    }

    if (nPixels > 0)
    {
      m_Variance[iDimension] /= nPixels;
    }
  }
}

template <typename TFixedImage, typename TMovingImage>
auto
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::ComputeMeasure(
  const TransformParametersType & parameters,
  const double *                  subtractionFactor) const -> MeasureType
{
  unsigned int iDimension = 0;

  this->SetTransformParameters(parameters);
  m_TransformMovingImageFilter->UpdateLargestPossibleRegion();
  MeasureType measure{};

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    if (Math::AlmostEquals(m_Variance[iDimension], MovedGradientPixelType{}))
    {
      continue;
    }
    // Iterate over the fixed and moving gradient images
    // calculating the similarity measure

    MovedGradientPixelType movedGradient;
    FixedGradientPixelType fixedGradient;

    MovedGradientPixelType diff;

    using FixedIteratorType = itk::ImageRegionConstIteratorWithIndex<FixedGradientImageType>;

    FixedIteratorType fixedIterator(m_FixedSobelFilters[iDimension]->GetOutput(), this->GetFixedImageRegion());

    using MovedIteratorType = itk::ImageRegionConstIteratorWithIndex<MovedGradientImageType>;

    MovedIteratorType movedIterator(m_MovedSobelFilters[iDimension]->GetOutput(), this->GetFixedImageRegion());

    m_FixedSobelFilters[iDimension]->UpdateLargestPossibleRegion();
    m_MovedSobelFilters[iDimension]->UpdateLargestPossibleRegion();

    this->m_NumberOfPixelsCounted = 0;

    while (!fixedIterator.IsAtEnd())
    {
      // Get the moving and fixed image gradients

      movedGradient = movedIterator.Get();
      fixedGradient = fixedIterator.Get();

      // And calculate the gradient difference

      diff = fixedGradient - subtractionFactor[iDimension] * movedGradient;

      measure += m_Variance[iDimension] / (m_Variance[iDimension] + diff * diff);

      ++fixedIterator;
      ++movedIterator;
    }
  }

  return measure;
}

template <typename TFixedImage, typename TMovingImage>
auto
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
  const TransformParametersType & parameters) const -> MeasureType
{
  unsigned int iFilter = 0; // Index of Sobel filters for
                            // each dimension
  unsigned int iDimension = 0;

  this->SetTransformParameters(parameters);
  m_TransformMovingImageFilter->UpdateLargestPossibleRegion();

  // Update the gradient images

  for (iFilter = 0; iFilter < MovedImageDimension; ++iFilter)
  {
    m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
  }

  // Compute the range of the moved image gradients
  // NB: Ideally this should be a filter as the computation is only
  //     required if the moved gradient image has been updated.
  //     However for the moment we'll assume that this is the case
  //     whenever GetValue() is called.

  this->ComputeMovedGradientRange();

  // Compute the similarity measure

  MovedGradientPixelType subtractionFactor[FixedImageDimension];
  MeasureType            currentMeasure;

  for (iDimension = 0; iDimension < FixedImageDimension; ++iDimension)
  {
    subtractionFactor[iDimension] = 1.0;
  }

  // Compute the new value of the measure for this subtraction factor
  currentMeasure = this->ComputeMeasure(parameters, subtractionFactor);

  return currentMeasure;
}

template <typename TFixedImage, typename TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType &                derivative) const
{
  TransformParametersType testPoint = parameters;

  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  derivative = DerivativeType(numberOfParameters);

  for (unsigned int i = 0; i < numberOfParameters; ++i)
  {
    testPoint[i] -= this->m_DerivativeDelta;
    const MeasureType valuep0 = this->GetValue(testPoint);
    testPoint[i] += 2 * this->m_DerivativeDelta;
    const MeasureType valuep1 = this->GetValue(testPoint);
    derivative[i] = (valuep1 - valuep0) / (2 * this->m_DerivativeDelta);
    testPoint[i] = parameters[i];
  }
}

template <typename TFixedImage, typename TMovingImage>
void
GradientDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType &                   Value,
  DerivativeType &                Derivative) const
{
  Value = this->GetValue(parameters);
  this->GetDerivative(parameters, Derivative);
}
} // end namespace itk

#endif
