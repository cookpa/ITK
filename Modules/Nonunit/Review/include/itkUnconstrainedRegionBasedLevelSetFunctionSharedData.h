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
#ifndef itkUnconstrainedRegionBasedLevelSetFunctionSharedData_h
#define itkUnconstrainedRegionBasedLevelSetFunctionSharedData_h

#include "itkRegionBasedLevelSetFunctionSharedData.h"

namespace itk
{
/** \class UnconstrainedRegionBasedLevelSetFunctionSharedData
 *
 * \brief Helper class used to share data in the ScalarChanAndVeseLevelSetFunction.
 *
 * This class holds cache data used during the computation of the LevelSet updates.
 *
 * Based on the paper \cite chan1999.
 *
 *
 * \author Mosaliganti K., Smith B., Gelas A., Gouaillard A., Megason S.
 *
 *  This code was taken from the Insight Journal paper \cite Mosaliganti_2009_c
 *  that is based on the papers \cite Mosaliganti_2009_a and
 *  \cite  Mosaliganti_2009_b.
 *
 * \ingroup ITKReview
 */
template <typename TInputImage, typename TFeatureImage, typename TSingleData>
class UnconstrainedRegionBasedLevelSetFunctionSharedData
  : public RegionBasedLevelSetFunctionSharedData<TInputImage, TFeatureImage, TSingleData>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(UnconstrainedRegionBasedLevelSetFunctionSharedData);

  using Self = UnconstrainedRegionBasedLevelSetFunctionSharedData;
  using Superclass = RegionBasedLevelSetFunctionSharedData<TInputImage, TFeatureImage, TSingleData>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  static constexpr unsigned int ImageDimension = TFeatureImage::ImageDimension;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  itkOverrideGetNameOfClassMacro(UnconstrainedRegionBasedLevelSetFunctionSharedData);

  using InputImageType = TInputImage;
  using typename Superclass::InputImagePointer;
  using typename Superclass::InputImageConstPointer;
  using typename Superclass::InputPixelType;
  using typename Superclass::InputRegionType;
  using typename Superclass::InputSizeType;
  using typename Superclass::InputSizeValueType;
  using typename Superclass::InputSpacingType;
  using typename Superclass::InputIndexType;
  using typename Superclass::InputIndexValueType;
  using typename Superclass::InputPointType;

  using FeatureImageType = TFeatureImage;
  using typename Superclass::FeatureImagePointer;
  using typename Superclass::FeatureImageConstPointer;
  using typename Superclass::FeaturePixelType;
  using typename Superclass::FeatureRegionType;
  using typename Superclass::FeatureSizeType;
  using typename Superclass::FeatureSizeValueType;
  using typename Superclass::FeatureSpacingType;
  using typename Superclass::FeatureIndexType;
  using typename Superclass::FeaturePointType;

  using typename Superclass::ListPixelType;
  using typename Superclass::ListImageType;
  using typename Superclass::ListImagePointer;
  using typename Superclass::ListImageConstPointer;
  using typename Superclass::ListRegionType;
  using typename Superclass::ListSizeType;
  using typename Superclass::ListSizeValueType;
  using typename Superclass::ListSpacingType;
  using typename Superclass::ListIndexType;
  using typename Superclass::ListIndexValueType;
  using typename Superclass::ListPointType;
  using typename Superclass::ListIteratorType;

  using typename Superclass::CentroidVectorType;
  using typename Superclass::SampleType;
  using typename Superclass::TreeGeneratorType;
  using typename Superclass::TreePointer;
  using typename Superclass::TreeType;
  using typename Superclass::KdTreePointer;

  using LevelSetDataType = TSingleData;
  using typename Superclass::LevelSetDataPointer;
  using typename Superclass::LevelSetDataPointerVector;
  using typename Superclass::LevelSetDataPointerVectorIterator;

  void
  PopulateListImage() override
  {
    ListPixelType L;

    for (unsigned int i = 0; i < this->m_FunctionCount; ++i)
    {
      L.push_back(i);
    }
    this->m_NearestNeighborListImage->FillBuffer(L);
  }

protected:
  UnconstrainedRegionBasedLevelSetFunctionSharedData()
    : Superclass()
  {}
  ~UnconstrainedRegionBasedLevelSetFunctionSharedData() override = default;
};
} // end namespace itk

#endif
