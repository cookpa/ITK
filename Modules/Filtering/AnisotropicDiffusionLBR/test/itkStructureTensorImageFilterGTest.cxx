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

#include "itkStructureTensorImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkGTest.h"

// A constant image has zero gradients, so the tensor-trace maximum is zero; the
// adimensionizing rescale must not divide by it and poison the tensors with NaN
// (issue #6575, B23).
TEST(StructureTensorImageFilter, ConstantImageYieldsFiniteZeroTensors)
{
  using ImageType = itk::Image<float, 2>;
  auto image = ImageType::New();
  image->SetRegions(ImageType::RegionType{ ImageType::IndexType{}, ImageType::SizeType::Filled(16) });
  image->Allocate();
  image->FillBuffer(0.0F);

  using FilterType = itk::StructureTensorImageFilter<ImageType>;
  auto filter = FilterType::New();
  filter->SetInput(image);
  filter->SetRescaleForUnitMaximumTrace(true);
  filter->Update();

  using TensorImageType = FilterType::TensorImageType;
  itk::ImageRegionConstIterator<TensorImageType> it(filter->GetOutput(), filter->GetOutput()->GetBufferedRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto & tensor = it.Get();
    for (unsigned int c = 0; c < tensor.Size(); ++c)
    {
      ASSERT_TRUE(std::isfinite(tensor[c]));
      ASSERT_NEAR(tensor[c], 0.0F, 1e-20);
    }
  }
}
