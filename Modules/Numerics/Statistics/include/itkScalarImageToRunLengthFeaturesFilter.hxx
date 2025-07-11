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
#ifndef itkScalarImageToRunLengthFeaturesFilter_hxx
#define itkScalarImageToRunLengthFeaturesFilter_hxx

#include "itkNeighborhood.h"
#include "itkMath.h"
#include "itkMakeUniqueForOverwrite.h"

namespace itk::Statistics
{
template <typename TImage, typename THistogramFrequencyContainer>
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::ScalarImageToRunLengthFeaturesFilter()
{
  this->SetNumberOfRequiredInputs(1);
  this->SetNumberOfRequiredOutputs(1);

  for (int i = 0; i < 2; ++i)
  {
    this->ProcessObject::SetNthOutput(i, this->MakeOutput(i));
  }

  this->m_RunLengthMatrixGenerator = RunLengthMatrixFilterType::New();
  this->m_FeatureMeans = FeatureValueVector::New();
  this->m_FeatureStandardDeviations = FeatureValueVector::New();

  // Set the requested features to the default value:
  // {Energy, Entropy, InverseDifferenceMoment, Inertia, ClusterShade,
  // ClusterProminence}
  const FeatureNameVectorPointer requestedFeatures = FeatureNameVector::New();
  // can't directly set this->m_RequestedFeatures since it is const!

  requestedFeatures->push_back(static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::ShortRunEmphasis));
  requestedFeatures->push_back(static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::LongRunEmphasis));
  requestedFeatures->push_back(static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::GreyLevelNonuniformity));
  requestedFeatures->push_back(static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::RunLengthNonuniformity));
  requestedFeatures->push_back(static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::LowGreyLevelRunEmphasis));
  requestedFeatures->push_back(static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::HighGreyLevelRunEmphasis));
  requestedFeatures->push_back(
    static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::ShortRunLowGreyLevelEmphasis));
  requestedFeatures->push_back(
    static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::ShortRunHighGreyLevelEmphasis));
  requestedFeatures->push_back(
    static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::LongRunLowGreyLevelEmphasis));
  requestedFeatures->push_back(
    static_cast<uint8_t>(itk::Statistics::RunLengthFeatureEnum::LongRunHighGreyLevelEmphasis));

  this->SetRequestedFeatures(requestedFeatures);

  // Set the offset directions to their defaults: half of all the possible
  // directions 1 pixel away. (The other half is included by symmetry.)
  // We use a neighborhood iterator to calculate the appropriate offsets.
  using NeighborhoodType = Neighborhood<typename ImageType::PixelType, ImageType::ImageDimension>;
  NeighborhoodType hood;
  hood.SetRadius(1);

  // select all "previous" neighbors that are face+edge+vertex
  // connected to the current pixel. do not include the center pixel.
  const unsigned int        centerIndex = hood.GetCenterNeighborhoodIndex();
  const OffsetVectorPointer offsets = OffsetVector::New();
  for (unsigned int d = 0; d < centerIndex; ++d)
  {
    const OffsetType offset = hood.GetOffset(d);
    offsets->push_back(offset);
  }
  this->SetOffsets(offsets);
  this->m_FastCalculations = false;
}

template <typename TImage, typename THistogramFrequencyContainer>
auto
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::MakeOutput(
  DataObjectPointerArraySizeType itkNotUsed(idx)) -> DataObjectPointer
{
  return FeatureValueVectorDataObjectType::New().GetPointer();
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::GenerateData()
{
  if (this->m_FastCalculations)
  {
    this->FastCompute();
  }
  else
  {
    this->FullCompute();
  }
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::FullCompute()
{
  const size_t numOffsets = this->m_Offsets->size();
  const size_t numFeatures = this->m_RequestedFeatures->size();
  auto **      features = new double *[numOffsets];
  for (size_t i = 0; i < numOffsets; ++i)
  {
    features[i] = new double[numFeatures];
  }

  // For each offset, calculate each feature

  using InternalRunLengthFeatureName = itk::Statistics::RunLengthFeatureEnum;

  {
    size_t offsetNum = 0;
    for (typename OffsetVector::ConstIterator offsetIt = this->m_Offsets->Begin(); offsetIt != this->m_Offsets->End();
         ++offsetIt, offsetNum++)
    {
      this->m_RunLengthMatrixGenerator->SetOffset(offsetIt.Value());
      this->m_RunLengthMatrixGenerator->Update();
      auto runLengthMatrixCalculator = RunLengthFeaturesFilterType::New();
      runLengthMatrixCalculator->SetInput(this->m_RunLengthMatrixGenerator->GetOutput());
      runLengthMatrixCalculator->Update();

      typename FeatureNameVector::ConstIterator fnameIt;
      {
        size_t featureNum = 0;
        for (fnameIt = this->m_RequestedFeatures->Begin(); fnameIt != this->m_RequestedFeatures->End();
             ++fnameIt, featureNum++)
        {
          features[offsetNum][featureNum] =
            runLengthMatrixCalculator->GetFeature(static_cast<InternalRunLengthFeatureName>(fnameIt.Value()));
        }
      }
    }
  }

  // Now get the mean and deviation of each feature across the offsets.
  this->m_FeatureMeans->clear();
  this->m_FeatureStandardDeviations->clear();
  const auto tempFeatureMeans = make_unique_for_overwrite<double[]>(numFeatures);
  const auto tempFeatureDevs = std::make_unique<double[]>(numFeatures);

  /*Compute incremental mean and SD, a la Knuth, "The  Art of Computer
    Programming, Volume 2: Seminumerical Algorithms",  section 4.2.2.
    Compute mean and standard deviation using the recurrence relation:
    M(1) = x(1), M(k) = M(k-1) + (x(k) - M(k-1) ) / k
    S(1) = 0, S(k) = S(k-1) + (x(k) - M(k-1)) * (x(k) - M(k))
    for 2 <= k <= n, then
    sigma = std::sqrt(S(n) / n) (or divide by n-1 for sample SD instead of
    population SD).
  */

  // Set up the initial conditions (k = 1)
  for (size_t featureNum = 0; featureNum < numFeatures; ++featureNum)
  {
    tempFeatureMeans[featureNum] = features[0][featureNum];
  }
  // Run through the recurrence (k = 2 ... N)
  for (size_t offsetNum = 1; offsetNum < numOffsets; ++offsetNum)
  {
    const int k = offsetNum + 1;
    for (size_t featureNum = 0; featureNum < numFeatures; ++featureNum)
    {
      const double M_k_minus_1 = tempFeatureMeans[featureNum];
      const double S_k_minus_1 = tempFeatureDevs[featureNum];
      const double x_k = features[offsetNum][featureNum];

      const double M_k = M_k_minus_1 + (x_k - M_k_minus_1) / k;
      const double S_k = S_k_minus_1 + (x_k - M_k_minus_1) * (x_k - M_k);

      tempFeatureMeans[featureNum] = M_k;
      tempFeatureDevs[featureNum] = S_k;
    }
  }
  for (size_t featureNum = 0; featureNum < numFeatures; ++featureNum)
  {
    tempFeatureDevs[featureNum] = std::sqrt(tempFeatureDevs[featureNum] / numOffsets);

    this->m_FeatureMeans->push_back(tempFeatureMeans[featureNum]);
    this->m_FeatureStandardDeviations->push_back(tempFeatureDevs[featureNum]);
  }

  auto * meanOutputObject =
    itkDynamicCastInDebugMode<FeatureValueVectorDataObjectType *>(this->ProcessObject::GetOutput(0));
  meanOutputObject->Set(this->m_FeatureMeans);

  auto * standardDeviationOutputObject =
    itkDynamicCastInDebugMode<FeatureValueVectorDataObjectType *>(this->ProcessObject::GetOutput(1));
  standardDeviationOutputObject->Set(this->m_FeatureStandardDeviations);

  for (size_t i = 0; i < numOffsets; ++i)
  {
    delete[] features[i];
  }
  delete[] features;
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::FastCompute()
{
  // Compute the feature for the first offset
  const typename OffsetVector::ConstIterator offsetIt = this->m_Offsets->Begin();
  this->m_RunLengthMatrixGenerator->SetOffset(offsetIt.Value());

  this->m_RunLengthMatrixGenerator->Update();
  auto runLengthMatrixCalculator = RunLengthFeaturesFilterType::New();
  runLengthMatrixCalculator->SetInput(this->m_RunLengthMatrixGenerator->GetOutput());
  runLengthMatrixCalculator->Update();

  using InternalRunLengthFeatureName = itk::Statistics::RunLengthFeatureEnum;
  this->m_FeatureMeans->clear();
  this->m_FeatureStandardDeviations->clear();
  for (typename FeatureNameVector::ConstIterator fnameIt = this->m_RequestedFeatures->Begin();
       fnameIt != this->m_RequestedFeatures->End();
       ++fnameIt)
  {
    this->m_FeatureMeans->push_back(
      runLengthMatrixCalculator->GetFeature(static_cast<InternalRunLengthFeatureName>(fnameIt.Value())));
    this->m_FeatureStandardDeviations->push_back(0.0);
  }

  auto * meanOutputObject =
    itkDynamicCastInDebugMode<FeatureValueVectorDataObjectType *>(this->ProcessObject::GetOutput(0));
  meanOutputObject->Set(this->m_FeatureMeans);

  auto * standardDeviationOutputObject =
    itkDynamicCastInDebugMode<FeatureValueVectorDataObjectType *>(this->ProcessObject::GetOutput(1));
  standardDeviationOutputObject->Set(this->m_FeatureStandardDeviations);
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::SetInput(const ImageType * image)
{
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(0, const_cast<ImageType *>(image));

  this->m_RunLengthMatrixGenerator->SetInput(image);
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::SetNumberOfBinsPerAxis(
  unsigned int numberOfBins)
{
  itkDebugMacro("setting NumberOfBinsPerAxis to " << numberOfBins);
  this->m_RunLengthMatrixGenerator->SetNumberOfBinsPerAxis(numberOfBins);
  this->Modified();
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::SetPixelValueMinMax(PixelType min,
                                                                                                PixelType max)
{
  itkDebugMacro("setting Min to " << min << "and Max to " << max);
  this->m_RunLengthMatrixGenerator->SetPixelValueMinMax(min, max);
  this->Modified();
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::SetDistanceValueMinMax(double min,
                                                                                                   double max)
{
  itkDebugMacro("setting Min to " << min << "and Max to " << max);
  this->m_RunLengthMatrixGenerator->SetDistanceValueMinMax(min, max);
  this->Modified();
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::SetMaskImage(const ImageType * image)
{
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(1, const_cast<ImageType *>(image));

  this->m_RunLengthMatrixGenerator->SetMaskImage(image);
}

template <typename TImage, typename THistogramFrequencyContainer>
const TImage *
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::GetInput() const
{
  if (this->GetNumberOfInputs() < 1)
  {
    return nullptr;
  }
  return static_cast<const ImageType *>(this->ProcessObject::GetInput(0));
}

template <typename TImage, typename THistogramFrequencyContainer>
const typename ScalarImageToRunLengthFeaturesFilter<TImage,
                                                    THistogramFrequencyContainer>::FeatureValueVectorDataObjectType *
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::GetFeatureMeansOutput() const
{
  return itkDynamicCastInDebugMode<const FeatureValueVectorDataObjectType *>(this->ProcessObject::GetOutput(0));
}

template <typename TImage, typename THistogramFrequencyContainer>
const typename ScalarImageToRunLengthFeaturesFilter<TImage,
                                                    THistogramFrequencyContainer>::FeatureValueVectorDataObjectType *
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::GetFeatureStandardDeviationsOutput() const
{
  return itkDynamicCastInDebugMode<const FeatureValueVectorDataObjectType *>(this->ProcessObject::GetOutput(1));
}

template <typename TImage, typename THistogramFrequencyContainer>
const TImage *
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::GetMaskImage() const
{
  if (this->GetNumberOfInputs() < 2)
  {
    return nullptr;
  }
  return static_cast<const ImageType *>(this->ProcessObject::GetInput(1));
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::SetInsidePixelValue(
  PixelType insidePixelValue)
{
  itkDebugMacro("setting InsidePixelValue to " << insidePixelValue);
  this->m_RunLengthMatrixGenerator->SetInsidePixelValue(insidePixelValue);
  this->Modified();
}

template <typename TImage, typename THistogramFrequencyContainer>
void
ScalarImageToRunLengthFeaturesFilter<TImage, THistogramFrequencyContainer>::PrintSelf(std::ostream & os,
                                                                                      Indent         indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "RequestedFeatures: " << this->GetRequestedFeatures() << std::endl;
  os << indent << "FeatureStandardDeviations: " << this->GetFeatureStandardDeviations() << std::endl;
  os << indent << "FastCalculations: " << this->GetFastCalculations() << std::endl;
  os << indent << "Offsets: " << this->GetOffsets() << std::endl;
  os << indent << "FeatureMeans: " << this->GetFeatureMeans() << std::endl;
}
} // namespace itk::Statistics

#endif
