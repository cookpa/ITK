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

#ifndef itkLevelSetEquationAdvectionTerm_hxx
#define itkLevelSetEquationAdvectionTerm_hxx

#include "itkMath.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkCastImageFilter.h"

namespace itk
{
template <typename TInput, typename TLevelSetContainer>
LevelSetEquationAdvectionTerm<TInput, TLevelSetContainer>::LevelSetEquationAdvectionTerm()
{
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    m_NeighborhoodScales[i] = 1.0;
  }
  this->m_TermName = "Advection term";
  this->m_RequiredData.insert("BackwardGradient");
  this->m_RequiredData.insert("ForwardGradient");
  this->m_DerivativeSigma = LevelSetOutputRealType{};
  this->m_AutoGenerateAdvectionImage = true;
}

template <typename TInput, typename TLevelSetContainer>
void
LevelSetEquationAdvectionTerm<TInput, TLevelSetContainer>::SetAdvectionImage(AdvectionImageType * iImage)
{
  this->m_AdvectionImage = iImage;
  this->m_AutoGenerateAdvectionImage = this->m_AdvectionImage.IsNull();
  this->Modified();
}

template <typename TInput, typename TLevelSetContainer>
void
LevelSetEquationAdvectionTerm<TInput, TLevelSetContainer>::InitializeParameters()
{
  this->SetUp();

  if (this->m_AutoGenerateAdvectionImage)
  {
    this->GenerateAdvectionImage();
  }
}

template <typename TInput, typename TLevelSetContainer>
void
LevelSetEquationAdvectionTerm<TInput, TLevelSetContainer>::GenerateAdvectionImage()
{
  this->m_AdvectionImage = AdvectionImageType::New();
  this->m_AdvectionImage->SetRequestedRegion(this->m_Input->GetRequestedRegion());
  this->m_AdvectionImage->SetBufferedRegion(this->m_Input->GetBufferedRegion());
  this->m_AdvectionImage->SetLargestPossibleRegion(this->m_Input->GetLargestPossibleRegion());
  this->m_AdvectionImage->Allocate();

  AdvectionImagePointer gradientImage;

  if (Math::NotAlmostEquals(m_DerivativeSigma, LevelSetOutputRealType{}))
  {
    using DerivativeFilterType = GradientRecursiveGaussianImageFilter<InputImageType, AdvectionImageType>;

    auto derivative = DerivativeFilterType::New();
    derivative->SetInput(this->m_Input);
    derivative->SetSigma(this->m_DerivativeSigma);
    derivative->Update();

    gradientImage = derivative->GetOutput();
  }
  else
  {
    using DerivativeFilterType = GradientImageFilter<InputImageType>;

    auto derivative = DerivativeFilterType::New();
    derivative->SetInput(this->m_Input);
    derivative->UseImageSpacingOn();
    derivative->Update();

    using DerivativeOutputImageType = typename DerivativeFilterType::OutputImageType;
    using GradientCasterType = CastImageFilter<DerivativeOutputImageType, AdvectionImageType>;

    auto caster = GradientCasterType::New();
    caster->SetInput(derivative->GetOutput());
    caster->Update();

    gradientImage = caster->GetOutput();
  }

  /* copy negative gradient into the advection image. */
  ImageRegionIterator<AdvectionImageType> dit(gradientImage, this->m_Input->GetRequestedRegion());
  ImageRegionIterator<AdvectionImageType> ait(this->m_AdvectionImage, this->m_AdvectionImage->GetRequestedRegion());

  for (dit.GoToBegin(), ait.GoToBegin(); !dit.IsAtEnd(); ++dit, ++ait)
  {
    VectorType v = dit.Get();
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      v[j] *= -NumericTraits<LevelSetOutputRealType>::OneValue();
    }
    ait.Set(v);
  }
}

template <typename TInput, typename TLevelSetContainer>
auto
LevelSetEquationAdvectionTerm<TInput, TLevelSetContainer>::AdvectionSpeed(const LevelSetInputIndexType & iP) const
  -> VectorType
{
  return this->m_AdvectionImage->GetPixel(iP);
}

template <typename TInput, typename TLevelSetContainer>
auto
LevelSetEquationAdvectionTerm<TInput, TLevelSetContainer>::Value(const LevelSetInputIndexType & iP)
  -> LevelSetOutputRealType
{
  VectorType             advectionField = this->AdvectionSpeed(iP);
  LevelSetOutputRealType oValue{};

  LevelSetGradientType backwardGradient = this->m_CurrentLevelSetPointer->EvaluateBackwardGradient(iP);
  LevelSetGradientType forwardGradient = this->m_CurrentLevelSetPointer->EvaluateForwardGradient(iP);

  for (unsigned int dim = 0; dim < ImageDimension; ++dim)
  {
    const LevelSetOutputRealType component = advectionField[dim];

    if (component > LevelSetOutputRealType{})
    {
      oValue += component * backwardGradient[dim];
    }
    else
    {
      oValue += component * forwardGradient[dim];
    }
  }

  return oValue;
}

template <typename TInput, typename TLevelSetContainer>
auto
LevelSetEquationAdvectionTerm<TInput, TLevelSetContainer>::Value(const LevelSetInputIndexType & iP,
                                                                 const LevelSetDataType &       iData)
  -> LevelSetOutputRealType
{
  VectorType             advectionField = this->AdvectionSpeed(iP);
  LevelSetOutputRealType oValue{};

  for (unsigned int dim = 0; dim < ImageDimension; ++dim)
  {
    const LevelSetOutputRealType component = advectionField[dim];

    if (component > LevelSetOutputRealType{})
    {
      oValue += component * iData.BackwardGradient.m_Value[dim];
    }
    else
    {
      oValue += component * iData.ForwardGradient.m_Value[dim];
    }
  }

  return oValue;
}

} // namespace itk

#endif
