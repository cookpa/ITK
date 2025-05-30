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
#ifndef itkNearestNeighborExtrapolateImageFunction_h
#define itkNearestNeighborExtrapolateImageFunction_h

#include "itkExtrapolateImageFunction.h"
#include <algorithm> // For clamp.

namespace itk
{
/**
 * \class NearestNeighborExtrapolateImageFunction
 * \brief Nearest neighbor extrapolation of a scalar image.
 *
 * NearestNeighborExtrapolateImageFunction extrapolate image intensity at
 * a specified point, continuous index or index by copying the intensity
 * of the nearest neighbor within the image buffer.
 *
 * This class is templated
 * over the input image type and the coordinate representation type
 * (e.g. float or double).
 *
 * \ingroup ImageFunctions
 * \ingroup ITKImageFunction
 */
template <typename TInputImage, typename TCoordinate = float>
class ITK_TEMPLATE_EXPORT NearestNeighborExtrapolateImageFunction
  : public ExtrapolateImageFunction<TInputImage, TCoordinate>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(NearestNeighborExtrapolateImageFunction);

  /** Standard class type aliases. */
  using Self = NearestNeighborExtrapolateImageFunction;
  using Superclass = ExtrapolateImageFunction<TInputImage, TCoordinate>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(NearestNeighborExtrapolateImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** OutputType type alias support */
  using typename Superclass::OutputType;

  /** InputImageType type alias support */
  using typename Superclass::InputImageType;

  /** Dimension underlying input image. */
  static constexpr unsigned int ImageDimension = Superclass::ImageDimension;

  /** Index type alias support */
  using typename Superclass::IndexType;
  using IndexValueType = typename IndexType::IndexValueType;

  /** ContinuousIndex type alias support */
  using typename Superclass::ContinuousIndexType;

  /** Evaluate the function at a ContinuousIndex position
   *
   * Returns the extrapolated image intensity at a
   * specified position by returning the intensity of the
   * nearest neighbor within the image buffer.
   *
   */
  OutputType
  EvaluateAtContinuousIndex(const ContinuousIndexType & index) const override
  {
    IndexType nindex;

    const IndexType startIndex = this->GetStartIndex();
    const IndexType endIndex = this->GetEndIndex();

    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      nindex[j] = std::clamp(Math::RoundHalfIntegerUp<IndexValueType>(index[j]), startIndex[j], endIndex[j]);
    }
    return static_cast<OutputType>(this->GetInputImage()->GetPixel(nindex));
  }

  /** Evaluate the function at a ContinuousIndex position
   *
   * Returns the extrapolated image intensity at a
   * specified position by returning the intensity of the
   * nearest neighbor within the image buffer.
   *
   */
  OutputType
  EvaluateAtIndex(const IndexType & index) const override
  {
    IndexType nindex;

    const IndexType startIndex = this->GetStartIndex();
    const IndexType endIndex = this->GetEndIndex();

    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      nindex[j] = std::clamp(index[j], startIndex[j], endIndex[j]);
    }
    return static_cast<OutputType>(this->GetInputImage()->GetPixel(nindex));
  }

protected:
  NearestNeighborExtrapolateImageFunction() = default;
  ~NearestNeighborExtrapolateImageFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override
  {
    Superclass::PrintSelf(os, indent);
  }
};
} // end namespace itk

#endif
