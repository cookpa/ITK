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
#ifndef itkAnnulusOperator_hxx
#define itkAnnulusOperator_hxx

#include "itkMath.h"
#include "itkSphereSpatialFunction.h"

#include <memory> // For make_unique.

namespace itk
{
/** Create the operator */
template <typename TPixel, unsigned int TDimension, typename TAllocator>
void
AnnulusOperator<TPixel, TDimension, TAllocator>::CreateOperator()
{
  const CoefficientVector coefficients = this->GenerateCoefficients();

  this->Fill(coefficients);
}

/** This function fills the coefficients into the corresponding
 *  neighborhood. */
template <typename TPixel, unsigned int TDimension, typename TAllocator>
void
AnnulusOperator<TPixel, TDimension, TAllocator>::Fill(const CoefficientVector & coeff)
{
  const std::slice temp_slice(0, coeff.size(), 1);

  typename Self::SliceIteratorType data(this, temp_slice);

  auto it = coeff.begin();

  // Copy the coefficients into the neighborhood
  for (data = data.Begin(); data < data.End(); ++data, ++it)
  {
    *data = *it;
  }
}

template <typename TPixel, unsigned int TDimension, typename TAllocator>
auto
AnnulusOperator<TPixel, TDimension, TAllocator>::GenerateCoefficients() -> CoefficientVector
{
  // Determine the initial kernel values...
  // values for a specified kernel
  double interiorV = m_InteriorValue;
  double annulusV = m_AnnulusValue;
  double exteriorV = m_ExteriorValue;

  if (m_Normalize)
  {
    const double bright = (m_BrightCenter ? 1.0 : -1.0);

    // Initial values for a normalized kernel
    interiorV = bright;
    annulusV = -1.0 * bright;
    exteriorV = 0.0;
  }

  // Compute the size of the kernel in pixels
  {
    SizeType r;

    const double outerRadius = m_InnerRadius + m_Thickness;
    for (unsigned int i = 0; i < TDimension; ++i)
    {
      r[i] = Math::Ceil<SizeValueType>(outerRadius / m_Spacing[i]);
    }
    this->SetRadius(r);
  }

  // Use a couple of sphere spatial functions...
  using SphereType = SphereSpatialFunction<TDimension>;
  auto innerS = SphereType::New();
  auto outerS = SphereType::New();

  innerS->SetRadius(m_InnerRadius);
  outerS->SetRadius(m_InnerRadius + m_Thickness);

  // Walk the neighborhood (this) and evaluate the sphere spatial
  // functions
  double       sumNotExterior = 0.0;
  double       sumNotExteriorSq = 0.0;
  unsigned int countNotExterior = 0;

  const typename SizeType::SizeValueType w = this->Size();

  const auto                     outside = std::make_unique<bool[]>(w);
  CoefficientVector              coeffP(w);
  OffsetType                     offset;
  typename SphereType::InputType point;

  for (unsigned int i = 0; i < w; ++i)
  {
    // get the offset from the center pixel
    offset = this->GetOffset(i);

    // convert to a position
    for (unsigned int j = 0; j < TDimension; ++j)
    {
      point[j] = m_Spacing[j] * offset[j];
    }

    // evaluate the spheres
    const bool inInner = innerS->Evaluate(point);
    const bool inOuter = outerS->Evaluate(point);

    // set the coefficients
    if (!inOuter)
    {
      // outside annulus
      coeffP[i] = exteriorV;
      outside[i] = true;
    }
    else if (!inInner)
    {
      // inside the outer circle but outside the inner circle
      coeffP[i] = annulusV;
      sumNotExterior += annulusV;
      sumNotExteriorSq += (annulusV * annulusV);
      ++countNotExterior;
      outside[i] = false;
    }
    else
    {
      // inside inner circle
      coeffP[i] = interiorV;
      sumNotExterior += interiorV;
      sumNotExteriorSq += (interiorV * interiorV);
      ++countNotExterior;
      outside[i] = false;
    }
  }

  // Normalize the kernel if necessary
  if (m_Normalize)
  {
    // Calculate the mean and standard deviation of kernel values NOT
    // the exterior
    auto         num = static_cast<double>(countNotExterior);
    const double mean = sumNotExterior / num;
    const double var = (sumNotExteriorSq - (sumNotExterior * sumNotExterior / num)) / (num - 1.0);
    const double std = std::sqrt(var);

    // convert std to a scaling factor k such that
    //
    //        || (coeffP - mean) / k || = 1.0
    //
    const double k = std * std::sqrt(num - 1.0);

    // Run through the kernel again, shifting and normalizing the
    // elements that are not exterior to the annulus.  This forces the
    // kernel to have mean zero and norm 1 AND forces the region
    // outside the annulus to have no influence.
    for (unsigned int i = 0; i < w; ++i)
    {
      // normalize the coefficient if it is inside the outer circle
      // (exterior to outer circle is already zero)
      if (!outside[i])
      {
        coeffP[i] = (coeffP[i] - mean) / k;
      }
    }
  }

  return coeffP;
}

} // namespace itk

#endif
