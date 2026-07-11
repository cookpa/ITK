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

#include "itkN4BiasFieldCorrectionImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkGTest.h"

// A constant image has a degenerate histogram range; sharpening must degrade to an
// identity mapping instead of producing NaN bin indices (issue #6575, B2).
TEST(N4BiasFieldCorrectionImageFilter, ConstantImageProducesFiniteOutput)
{
  using ImageType = itk::Image<float, 2>;
  auto image = ImageType::New();
  image->SetRegions(ImageType::RegionType{ ImageType::IndexType{}, ImageType::SizeType::Filled(32) });
  image->Allocate();
  image->FillBuffer(1.0F);

  using FilterType = itk::N4BiasFieldCorrectionImageFilter<ImageType>;
  auto filter = FilterType::New();
  filter->SetInput(image);
  filter->SetMaximumNumberOfIterations(FilterType::VariableSizeArrayType(1, 2));
  filter->Update();

  itk::ImageRegionConstIterator<ImageType> it(filter->GetOutput(), filter->GetOutput()->GetBufferedRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ASSERT_TRUE(std::isfinite(it.Get()));
    ASSERT_NEAR(it.Get(), 1.0F, 1e-2) << " constant input must be (nearly) unchanged";
  }
}
